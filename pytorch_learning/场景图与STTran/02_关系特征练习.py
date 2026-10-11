"""第二课：用小张量理解 STTran 的关系特征。

本脚本只使用人工框、人工特征和小维度线性层，不加载图片、检测器、
GloVe 或真实 STTran 权重。它展示数据组织与计算顺序，不评估识别效果。
"""

import torch
from torch import nn
from torch.nn import functional as F


def 创建联合框(boxes, pair_idx, im_idx):
    """boxes 每行：[帧号, x1, y1, x2, y2]；输出也是这个顺序。"""
    subject_boxes = boxes[pair_idx[:, 0]]
    object_boxes = boxes[pair_idx[:, 1]]
    # 一个关系对必须连接同一帧里的两个物体。
    assert torch.equal(subject_boxes[:, 0], object_boxes[:, 0])
    assert torch.equal(subject_boxes[:, 0].long(), im_idx)
    left_top = torch.minimum(subject_boxes[:, 1:3], object_boxes[:, 1:3])
    right_bottom = torch.maximum(subject_boxes[:, 3:5], object_boxes[:, 3:5])
    return torch.cat((im_idx.float().unsqueeze(1), left_top, right_bottom), dim=1)


def 创建教学空间布局(boxes, pair_idx, union_boxes, size=4):
    """教学版：把联合框划成 size×size 小格，以格子中心是否在框内填 0/1。

    官方 draw_union_boxes 会保留边缘的部分覆盖值；本函数采用较易手算的
    格子中心判断，不能用于替代官方函数做复现实验。
    """
    layouts = torch.zeros(len(pair_idx), 2, size, size)
    for relation_index, (subject_index, object_index) in enumerate(pair_idx):
        x1, y1, x2, y2 = union_boxes[relation_index, 1:]
        assert x2 > x1 and y2 > y1
        centers_x = x1 + (torch.arange(size) + 0.5) * (x2 - x1) / size
        centers_y = y1 + (torch.arange(size) + 0.5) * (y2 - y1) / size
        for channel, object_row in enumerate((subject_index, object_index)):
            bx1, by1, bx2, by2 = boxes[object_row, 1:]
            inside_x = (centers_x >= bx1) & (centers_x < bx2)
            inside_y = (centers_y >= by1) & (centers_y < by2)
            layouts[relation_index, channel] = (
                inside_y[:, None] & inside_x[None, :]
            ).float()
    return layouts


def 演示双线性插值():
    # 四个格点：左上 1，右上 2，左下 3，右下 4。
    feature_map = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x, y = 0.25, 0.5
    value = (
        (1 - x) * (1 - y) * feature_map[0, 0]
        + x * (1 - y) * feature_map[0, 1]
        + (1 - x) * y * feature_map[1, 0]
        + x * y * feature_map[1, 1]
    )
    print("插值位置 x=0.25、y=0.5，四个邻点加权结果：", value.item())
    assert torch.isclose(value, torch.tensor(2.25))


def 演示冻结与训练():
    torch.manual_seed(7)
    extractor = nn.Linear(4, 4)
    relation_head = nn.Linear(4, 1)
    inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    extractor.eval()  # 改变运行模式，本身不关闭自动求导。
    with torch.no_grad():
        constant_features = extractor(inputs)
    prediction = relation_head(constant_features)
    loss = (prediction - 1.0).square().mean()
    loss.backward()
    print("人工特征提取器的梯度：", extractor.weight.grad)
    print("后续关系层是否得到梯度：", relation_head.weight.grad is not None)
    assert extractor.weight.grad is None
    assert relation_head.weight.grad is not None


def main():
    torch.set_printoptions(precision=3, sci_mode=False)
    print("本练习使用 CPU 和人工数据。所有特征都不是从真实图片提取的。")

    # 两帧的数据合在一张表里；编号 3 是第二帧的人。
    boxes = torch.tensor([
        [0, 10, 10, 50, 70],
        [0, 40, 30, 60, 50],
        [0, 0, 50, 60, 90],
        [1, 12, 10, 52, 70],
        [1, 42, 28, 62, 48],
    ], dtype=torch.float32)
    features = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [0.0, 1.0, 0.0, 1.0],
        [2.0, 0.0, 1.0, 0.0],
        [2.0, 1.0, 3.0, 4.0],
        [1.0, 1.0, 0.0, 2.0],
    ])
    # 教学类别编号：0 背景，1 人，2 杯子，3 椅子。
    labels = torch.tensor([1, 2, 3, 1, 2], dtype=torch.long)
    pair_idx = torch.tensor([[0, 1], [0, 2], [3, 4]], dtype=torch.long)
    im_idx = torch.tensor([0, 0, 1], dtype=torch.long)
    assert boxes.shape == (5, 5) and features.shape == (5, 4)
    print("物体表形状：", list(boxes.shape))
    print("物体特征形状：", list(features.shape))
    print("物体对：", pair_idx.tolist(), "；所属帧：", im_idx.tolist())

    subject_features = features[pair_idx[:, 0]]
    object_features = features[pair_idx[:, 1]]
    print("按主体编号取特征：\n", subject_features)
    print("按客体编号取特征：\n", object_features)
    assert torch.equal(subject_features[0], subject_features[1])
    assert torch.equal(subject_features[2], features[3])

    union_boxes = 创建联合框(boxes, pair_idx, im_idx)
    print("联合框：\n", union_boxes)
    assert torch.equal(union_boxes[0], torch.tensor([0, 10, 10, 60, 70]))
    spatial_layouts = 创建教学空间布局(boxes, pair_idx, union_boxes)
    print("两通道空间布局形状：", list(spatial_layouts.shape))
    print("第一对的主体布局：\n", spatial_layouts[0, 0])
    print("第一对的客体布局：\n", spatial_layouts[0, 1])
    centered_layouts = spatial_layouts - 0.5
    print("减 0.5 后，第一对客体布局：\n", centered_layouts[0, 1])

    # 练习中的维度 4→2，代替官方关系分支的 2048→512。
    subject_projection = nn.Linear(4, 2)
    object_projection = nn.Linear(4, 2)
    with torch.no_grad():
        subject_projection.weight.copy_(torch.tensor([
            [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]
        ]))
        subject_projection.bias.copy_(torch.tensor([0.5, -0.5]))
        object_projection.weight.copy_(torch.tensor([
            [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]
        ]))
        object_projection.bias.zero_()
    subject_representations = subject_projection(subject_features)
    object_representations = object_projection(object_features)
    print("主体投影结果：\n", subject_representations)
    print("客体投影结果：\n", object_representations)
    assert torch.allclose(subject_representations[0], torch.tensor([4.5, 5.5]))
    manual = subject_features @ subject_projection.weight.T + subject_projection.bias
    assert torch.allclose(manual, subject_representations)

    # 每对一个人工区域图，形状 [3,1,2,2]；这里不实施 ROIAlign。
    artificial_union_maps = torch.tensor([
        [[[1.0, 2.0], [3.0, 4.0]]],
        [[[2.0, 1.0], [0.0, 3.0]]],
        [[[0.0, 1.0], [2.0, 3.0]]],
    ])
    # 为演示对齐形状，本例用平均池化与固定通道权重；官方使用可训练卷积。
    pooled_layouts = F.adaptive_avg_pool2d(centered_layouts, (2, 2))
    artificial_geometry_maps = (
        0.75 * pooled_layouts[:, 0:1] + 0.25 * pooled_layouts[:, 1:2]
    )
    fused_maps = artificial_union_maps + artificial_geometry_maps
    flattened_maps = fused_maps.flatten(start_dim=1)
    assert fused_maps.shape == (3, 1, 2, 2)
    assert flattened_maps.shape == (3, 4)
    print("相加后的区域图形状：", list(fused_maps.shape))
    print("展平后的形状：", list(flattened_maps.shape))
    print("第一对：区域图\n", artificial_union_maps[0, 0])
    print("第一对：几何图\n", artificial_geometry_maps[0, 0])
    print("第一对：相加后展平", flattened_maps[0])
    union_projection = nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        union_projection.weight.copy_(torch.tensor([
            [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]
        ]))
    union_representations = union_projection(flattened_maps)
    assert torch.allclose(flattened_maps[0], torch.tensor([1.25, 2.0, 3.25, 4.0]))
    assert torch.allclose(union_representations[0], torch.tensor([3.25, 7.25]))

    # 两张嵌入表以同一人工表初始化，但参数是分开的。
    embedding_values = torch.tensor([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]
    ])
    subject_embedding = nn.Embedding.from_pretrained(embedding_values.clone(), freeze=False)
    object_embedding = nn.Embedding.from_pretrained(embedding_values.clone(), freeze=False)
    subject_semantics = subject_embedding(labels[pair_idx[:, 0]])
    object_semantics = object_embedding(labels[pair_idx[:, 1]])
    print("按主体类别查表：\n", subject_semantics)
    print("按客体类别查表：\n", object_semantics)
    assert torch.equal(object_semantics[0], object_semantics[2])

    relation_features = torch.cat((
        subject_representations, object_representations, union_representations,
        subject_semantics, object_semantics
    ), dim=1)
    print("拼接后的关系特征形状：", list(relation_features.shape))
    print("第一对的关系特征：", relation_features[0])
    assert relation_features.shape == (3, 10)  # 2+2+2+2+2。
    print("一行对应一个物体对，不对应一条已确定的关系标签。")

    # 两种操作：拼接保留所有位置，逐元素相加把对应位置融合。
    a, b = torch.tensor([[1.0, 2.0]]), torch.tensor([[10.0, 20.0]])
    print("拼接 [1,2] 和 [10,20]：", torch.cat((a, b), dim=1).tolist())
    print("相加 [1,2] 和 [10,20]：", (a + b).tolist())
    演示双线性插值()
    演示冻结与训练()
    print("检查全部通过。下一章将学习这些关系向量怎样交换同帧与跨帧信息。")


if __name__ == "__main__":
    main()
