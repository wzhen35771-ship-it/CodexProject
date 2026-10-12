"""空间与时间注意力练习：人工特征、小维度、CPU，帮助跟踪 STTran 数据流。

本程序没有物体检测器、图像特征提取器、关系标签或完整训练过程。
这里的 5 行特征分别代表三个视频帧中的 2、1、2 个物体对。
"""

import math

import torch
from torch import nn


class SpatialLayer(nn.Module):
    """按照官方空间层的顺序：两次残差相加，两次 LayerNorm。"""

    def __init__(self, feature_dim=8, num_heads=2, hidden_dim=12):
        super().__init__()
        self.attention = nn.MultiheadAttention(feature_dim, num_heads, dropout=0.0)
        self.linear_1 = nn.Linear(feature_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, feature_dim)
        self.layer_norm_1 = nn.LayerNorm(feature_dim)
        self.layer_norm_2 = nn.LayerNorm(feature_dim)

    def forward(self, inputs, padding_mask):
        attention_output, attention_weights = self.attention(
            inputs, inputs, inputs,
            key_padding_mask=padding_mask,
            need_weights=True,
        )
        hidden_states = self.layer_norm_1(inputs + attention_output)
        feedforward_output = self.linear_2(torch.relu(self.linear_1(hidden_states)))
        outputs = self.layer_norm_2(hidden_states + feedforward_output)
        return outputs, attention_weights


class TemporalLayer(nn.Module):
    """对应官方时间层的顺序：第二次残差相加后没有第二个 LayerNorm。"""

    def __init__(self, feature_dim=8, num_heads=2, hidden_dim=12):
        super().__init__()
        self.attention = nn.MultiheadAttention(feature_dim, num_heads, dropout=0.0)
        self.linear_1 = nn.Linear(feature_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, inputs, padding_mask, position_vectors):
        attention_output, attention_weights = self.attention(
            query=inputs + position_vectors,
            key=inputs + position_vectors,
            value=inputs,
            key_padding_mask=padding_mask,
            need_weights=True,
        )
        hidden_states = self.layer_norm(inputs + attention_output)
        feedforward_output = self.linear_2(torch.relu(self.linear_1(hidden_states)))
        outputs = hidden_states + feedforward_output
        return outputs, attention_weights


def pack_frame_relations(relation_features, frame_indices):
    """从 [R,D] 打包成 [L,T,D]，每一列是一帧。"""
    num_frames = int(frame_indices.max().item()) + 1
    relations_per_frame = torch.bincount(frame_indices, minlength=num_frames)
    max_relations = int(relations_per_frame.max().item())
    spatial_inputs = relation_features.new_zeros(
        (max_relations, num_frames, relation_features.shape[1])
    )
    padding_mask = torch.ones((num_frames, max_relations), dtype=torch.bool)
    original_row_indices = torch.full((max_relations, num_frames), -1, dtype=torch.long)

    for frame_index in range(num_frames):
        row_indices = torch.where(frame_indices == frame_index)[0]
        num_relations = row_indices.numel()
        spatial_inputs[:num_relations, frame_index] = relation_features[row_indices]
        padding_mask[frame_index, :num_relations] = False
        original_row_indices[:num_relations, frame_index] = row_indices
    return spatial_inputs, padding_mask, original_row_indices


def restore_frame_relations(spatial_outputs, original_row_indices, total_relations):
    """按行号放回 [R,D]，无需依赖展平以后恰好是正确的顺序。"""
    restored_relations = spatial_outputs.new_zeros((total_relations, spatial_outputs.shape[-1]))
    for frame_index in range(spatial_outputs.shape[1]):
        valid_mask = original_row_indices[:, frame_index] >= 0
        restored_relations[original_row_indices[valid_mask, frame_index]] = (
            spatial_outputs[valid_mask, frame_index]
        )
    return restored_relations


def pack_adjacent_windows(relation_features, frame_indices, position_table):
    """窗口大小 2、步长 1；窗口内首先放较早帧，再放较晚帧。"""
    num_frames = int(frame_indices.max().item()) + 1
    assert num_frames >= 2, "本练习要求至少两帧。"
    max_relations = int(torch.bincount(frame_indices).max().item())
    capacity = 2 * max_relations
    num_windows = num_frames - 1
    temporal_inputs = relation_features.new_zeros(
        (capacity, num_windows, relation_features.shape[-1])
    )
    position_vectors = torch.zeros_like(temporal_inputs)
    padding_mask = torch.ones((num_windows, capacity), dtype=torch.bool)
    original_row_indices = torch.full((capacity, num_windows), -1, dtype=torch.long)
    absolute_frame_indices = torch.full((capacity, num_windows), -1, dtype=torch.long)

    for window_index in range(num_windows):
        earlier_row_indices = torch.where(frame_indices == window_index)[0]
        later_row_indices = torch.where(frame_indices == window_index + 1)[0]
        row_indices = torch.cat((earlier_row_indices, later_row_indices))
        num_relations = row_indices.numel()
        num_earlier_relations = earlier_row_indices.numel()
        temporal_inputs[:num_relations, window_index] = relation_features[row_indices]
        position_vectors[:num_earlier_relations, window_index] = position_table[0]
        position_vectors[num_earlier_relations:num_relations, window_index] = position_table[1]
        padding_mask[window_index, :num_relations] = False
        original_row_indices[:num_relations, window_index] = row_indices
        absolute_frame_indices[:num_relations, window_index] = frame_indices[row_indices]
    return (
        temporal_inputs, position_vectors, padding_mask,
        original_row_indices, absolute_frame_indices,
    )


def restore_latter_frame_relations(
    temporal_outputs, original_row_indices, absolute_frame_indices, total_relations,
):
    """latter：第 0 帧由第一个窗口保留，其余帧取前一个窗口的结果。"""
    restored_relations = temporal_outputs.new_zeros((total_relations, temporal_outputs.shape[-1]))
    write_counts = torch.zeros(total_relations, dtype=torch.long)
    for window_index in range(temporal_outputs.shape[1]):
        keep_mask = absolute_frame_indices[:, window_index] == window_index + 1
        if window_index == 0:
            keep_mask = keep_mask | (absolute_frame_indices[:, window_index] == 0)
        row_indices = original_row_indices[keep_mask, window_index]
        restored_relations[row_indices] = temporal_outputs[keep_mask, window_index]
        write_counts[row_indices] += 1
    assert torch.equal(write_counts, torch.ones_like(write_counts))
    return restored_relations


def verify_manual_attention():
    queries = torch.tensor([[1.0, 0.0]])
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    values = torch.tensor([[10.0, 0.0], [0.0, 20.0]])
    scores = queries @ keys.T / math.sqrt(2)
    attention_weights = torch.softmax(scores, dim=-1)
    weighted_result = attention_weights @ values
    print("手算分数：", scores)
    print("手算权重：", attention_weights)
    print("手算加权结果：", weighted_result)
    assert torch.allclose(attention_weights.sum(dim=-1), torch.ones(1))
    assert torch.allclose(weighted_result, torch.tensor([[6.6976, 6.6048]]), atol=1e-4)


def verify_manual_layer_norm():
    inputs = torch.tensor([[1.0, 3.0]])
    feature_mean = inputs.mean(dim=-1, keepdim=True)
    variance = inputs.var(dim=-1, correction=0, keepdim=True)
    manual_result = (inputs - feature_mean) / torch.sqrt(variance + 1e-5)
    layer_norm_module = nn.LayerNorm(2)
    print("归一化均值：", feature_mean, "总体方差：", variance)
    print("归一化结果：", layer_norm_module(inputs))
    assert torch.allclose(manual_result, layer_norm_module(inputs))


def main():
    torch.manual_seed(7)
    torch.set_printoptions(precision=4, sci_mode=False)
    verify_manual_attention()
    verify_manual_layer_norm()

    # 行号 0、1 属于第 0 帧；行号 2 属于第 1 帧；行号 3、4 属于第 2 帧。
    # 数字是人工特征，用来追踪维度；每一列没有指定为“颜色”或“拿着”。
    relation_features = (torch.arange(40, dtype=torch.float32).reshape(5, 8) / 10).requires_grad_()
    frame_indices = torch.tensor([0, 0, 1, 2, 2])
    spatial_inputs, spatial_mask, spatial_row_indices = pack_frame_relations(
        relation_features, frame_indices,
    )
    print("\n原始关系特征形状：", tuple(relation_features.shape))
    print("空间输入形状 [L,T,D]：", tuple(spatial_inputs.shape))
    print("空间行号，每一列是一帧：\n", spatial_row_indices)
    print("空间掩码，每一行是一帧，True 表示补齐：\n", spatial_mask)
    assert torch.equal(spatial_row_indices, torch.tensor([[0, 2, 3], [1, -1, 4]]))
    assert torch.equal(
        restore_frame_relations(spatial_inputs, spatial_row_indices, 5),
        relation_features,
    )

    spatial_model = SpatialLayer()
    spatial_outputs, spatial_weights = spatial_model(spatial_inputs, spatial_mask)
    frame_results = restore_frame_relations(spatial_outputs, spatial_row_indices, 5)
    print("空间权重形状 [T,L,L]：", tuple(spatial_weights.shape))
    print("第 1 帧空间注意力权重：\n", spatial_weights[1])
    assert tuple(frame_results.shape) == (5, 8)
    # 第 1 帧只有一个有效 key，第二列权重必须是 0。
    assert torch.equal(spatial_weights[1, :, 1], torch.zeros(2))

    position_embedding = nn.Embedding(2, 8)
    nn.init.uniform_(position_embedding.weight)
    (
        temporal_inputs, position_vectors, temporal_mask,
        temporal_row_indices, temporal_frame_indices,
    ) = pack_adjacent_windows(
        frame_results, frame_indices, position_embedding.weight,
    )
    print("\n时间输入形状 [2L,T-1,D]：", tuple(temporal_inputs.shape))
    print("时间行号，每一列是一个相邻帧窗口：\n", temporal_row_indices)
    print("时间帧号：\n", temporal_frame_indices)
    print("时间掩码，每一行是一个窗口：\n", temporal_mask)
    print("较早位置向量：", position_embedding.weight[0])
    print("较晚位置向量：", position_embedding.weight[1])
    assert torch.equal(temporal_row_indices, torch.tensor([[0, 2], [1, 3], [2, 4], [-1, -1]]))
    assert torch.equal(temporal_frame_indices, torch.tensor([[0, 1], [0, 2], [1, 2], [-1, -1]]))
    assert torch.equal(temporal_mask, torch.tensor([[False, False, False, True]] * 2))
    assert torch.equal(position_vectors[2, 0], position_embedding.weight[1])  # 关系行 2 在第一个窗口较晚。
    assert torch.equal(position_vectors[0, 1], position_embedding.weight[0])  # 同一关系行 2 在第二个窗口较早。
    # 用时间输入本身核对打包和还原，不受模型随机参数影响。
    assert torch.equal(
        restore_latter_frame_relations(
            temporal_inputs, temporal_row_indices, temporal_frame_indices, 5,
        ),
        frame_results,
    )

    temporal_model = TemporalLayer()
    temporal_outputs, temporal_weights = temporal_model(
        temporal_inputs, temporal_mask, position_vectors,
    )
    final_results = restore_latter_frame_relations(
        temporal_outputs, temporal_row_indices, temporal_frame_indices, 5,
    )
    print("时间权重形状 [T-1,2L,2L]：", tuple(temporal_weights.shape))
    print("最终结果形状 [R,D]：", tuple(final_results.shape))
    print("最终结果：\n", final_results)
    assert torch.isfinite(final_results).all()
    assert torch.equal(temporal_weights[:, :, -1], torch.zeros(2, 4))

    # 只检查计算图能连回人工输入、注意力参数、可学习位置向量。
    # 平方平均是本练习的检查目标，不是真实 STTran 的关系分类损失。
    check_loss = final_results.square().mean()
    check_loss.backward()
    assert relation_features.grad is not None and torch.isfinite(relation_features.grad).all()
    assert spatial_model.attention.in_proj_weight.grad is not None
    assert temporal_model.attention.in_proj_weight.grad is not None
    assert position_embedding.weight.grad is not None
    print("\n检查目标：", check_loss.item())
    print("输入梯度形状：", tuple(relation_features.grad.shape))
    print("位置表梯度形状：", tuple(position_embedding.weight.grad.shape))
    print("所有形状、有效行号、补齐掩码、还原顺序和反向传播检查通过。")


if __name__ == "__main__":
    main()
