import argparse
import sys
import types
from pathlib import Path

import torch
import torchvision


def install_shim(module_name, **attrs):
    module = types.ModuleType(module_name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[module_name] = module
    return module


def roi_align_forward(input, roi, spatial_scale, pooled_height, pooled_width, sampling_ratio):
    return torchvision.ops.roi_align(
        input,
        roi,
        output_size=(pooled_height, pooled_width),
        spatial_scale=spatial_scale,
        sampling_ratio=sampling_ratio,
        aligned=False,
    )


def nms(boxes, scores, iou_threshold):
    return torchvision.ops.nms(boxes, scores, iou_threshold)


def zeros_union_boxes(box_pairs, pooling_size, padding=0):
    import numpy as np
    return np.zeros((len(box_pairs), 2, pooling_size, pooling_size), dtype=np.float32)


def zeros_bbox_overlaps(boxes, query_boxes):
    import numpy as np
    return np.zeros((len(boxes), len(query_boxes)), dtype=np.float32)


def write_placeholder_glove(path, tokens, dim=200):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    vectors = torch.randn(len(tokens), dim)
    vocabulary = {token: index for index, token in enumerate(tokens)}
    torch.save((vocabulary, vectors, dim), path)
    print(f"created placeholder {path}")


def load_state(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    for key, value in checkpoint.items():
        if isinstance(value, dict) and any(isinstance(item, torch.Tensor) for item in value.values()):
            return key, value
    raise RuntimeError(f"no tensor state_dict found in {path}")


def build_toy_entry(device):
    return {
        "features": torch.randn(2, 2048, device=device),
        "pair_idx": torch.tensor([[0, 1]], dtype=torch.long, device=device),
        "union_feat": torch.randn(1, 1024, 7, 7, device=device),
        "spatial_masks": torch.randn(1, 2, 27, 27, device=device),
        "boxes": torch.tensor(
            [[0.0, 0.1, 0.1, 0.5, 0.5], [0.0, 0.5, 0.5, 0.9, 0.9]],
            device=device,
        ),
        "labels": torch.tensor([1, 2], dtype=torch.long, device=device),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--forward", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))

    state_key, state = load_state(args.ckpt)
    obj_count = state["object_classifier.obj_embed.weight"].shape[0] + 1
    attention_count = state["a_rel_compress.weight"].shape[0]
    spatial_count = state["s_rel_compress.weight"].shape[0]
    contact_count = state["c_rel_compress.weight"].shape[0]
    obj_classes = ["__background__"] + [f"obj_{i}" for i in range(1, obj_count)]
    write_placeholder_glove(repo / "data" / "glove.6B.200d.pt", obj_classes[1:])
    write_placeholder_glove(repo / "checkpoints" / "glove.6B.200d.pt", obj_classes)

    install_shim(
        "fasterRCNN.lib.model._C",
        roi_align_forward=roi_align_forward,
        nms=nms,
        roi_pool_forward=None,
        roi_pool_backward=None,
        roi_align_backward=None,
    )
    install_shim("lib.draw_rectangles.draw_rectangles", draw_union_boxes=zeros_union_boxes)
    install_shim(
        "lib.fpn.box_intersections_cpu.bbox",
        bbox_overlaps=zeros_bbox_overlaps,
        bbox_intersections=zeros_bbox_overlaps,
    )

    from lib.supervised.sgg.sttran.sttran import STTran

    model = STTran(
        mode="predcls",
        attention_class_num=attention_count,
        spatial_class_num=spatial_count,
        contact_class_num=contact_count,
        obj_classes=obj_classes,
        enc_layer_num=1,
        dec_layer_num=3,
    )
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)

    print(f"repo={repo}")
    print(f"checkpoint={args.ckpt}")
    print(f"state_key={state_key}")
    print(f"objects={obj_count} attention={attention_count} spatial={spatial_count} contact={contact_count}")
    print(f"missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
    print(f"missing_sample={missing[:20]}")
    print(f"unexpected_sample={unexpected[:20]}")
    if missing or unexpected:
        raise SystemExit(2)

    if args.forward:
        if not torch.cuda.is_available():
            raise RuntimeError("The official STTran forward path hard-codes CUDA masks")
        device = torch.device("cuda:0")
        model.to(device).eval()
        entry = build_toy_entry(device)
        with torch.no_grad():
            output = model(entry)
        print(f"forward_device={device}")
        print(f"attention_distribution={tuple(output['attention_distribution'].shape)}")
        print(f"spatial_distribution={tuple(output['spatial_distribution'].shape)}")
        print(f"contacting_distribution={tuple(output['contacting_distribution'].shape)}")
        print("smoke forward ok")


if __name__ == "__main__":
    main()
