import torch

from usg_par.encoders.types import EncodedModality
from usg_par.model import USGParCore


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(7)

    batch = 1
    dim = 32
    num_queries = 6
    num_classes = 5
    num_predicates = 4

    feats = EncodedModality(
        feats_per_scale=[
            torch.randn(batch, 4, dim, device=device),
            torch.randn(batch, 16, dim, device=device),
            torch.randn(batch, 64, dim, device=device),
        ],
        feat_sizes=[(2, 2), (4, 4), (8, 8)],
        mask_features=torch.randn(batch, dim, 8, 8, device=device),
        context_tokens=torch.randn(batch, 12, dim, device=device),
    )

    model = USGParCore(
        modalities=("image",),
        dim=dim,
        num_queries=num_queries,
        num_predicates=num_predicates,
        num_scales=3,
        mask_decoder_layers=2,
        rpc_layers=1,
        relation_layers=1,
        top_k=8,
    ).to(device)

    class_embeddings = {"image": torch.randn(num_classes, dim, device=device)}
    with torch.no_grad():
        out = model({"image": feats}, class_embeddings)

    image_out = out.per_modality["image"]
    print("device", device)
    print("cls_logits", tuple(image_out.cls_logits.shape))
    print("pred_masks", tuple(image_out.pred_masks.shape))
    print("relation_logits", tuple(image_out.relation_logits.shape))
    print("smoke ok")


if __name__ == "__main__":
    main()
