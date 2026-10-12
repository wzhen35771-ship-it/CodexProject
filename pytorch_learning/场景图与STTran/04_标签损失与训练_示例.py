"""人工关系特征的训练练习：三组输出、标签、损失和梯度。

这不是完整 STTran：没有图片、检测器、空间或时间 Transformer。
三组关系的输出数量与官方 Action Genome 基线相同：3、6、17。
默认演示 sigmoid + MultiLabelMarginLoss；--loss bce 演示 sigmoid + BCELoss。
运行示例：python 04_标签损失与训练_示例.py --loss margin
"""

import argparse
from pathlib import Path

import torch
from torch import nn


def build_margin_labels(positive_class_lists: list[list[int]], num_classes: int) -> torch.Tensor:
    """正类编号放在前面，剩余位置填 -1；不是 0/1 标签。"""
    labels = torch.full((len(positive_class_lists), num_classes), -1, dtype=torch.long)
    for row_index, positive_classes in enumerate(positive_class_lists):
        if len(set(positive_classes)) != len(positive_classes):
            raise ValueError("正类编号不能重复")
        if any(class_index < 0 or class_index >= num_classes for class_index in positive_classes):
            raise ValueError("正类编号超出类别范围")
        labels[row_index, :len(positive_classes)] = torch.tensor(positive_classes, dtype=torch.long)
    return labels


def build_binary_labels(positive_class_lists: list[list[int]], num_classes: int) -> torch.Tensor:
    """每列表示该类别是否属于正类：属于填 1，其余填 0。"""
    labels = torch.zeros((len(positive_class_lists), num_classes), dtype=torch.float32)
    for row_index, positive_classes in enumerate(positive_class_lists):
        labels[row_index, positive_classes] = 1.0
    return labels


def get_gradient(tensor: torch.Tensor) -> torch.Tensor:
    """反向传播后检查梯度；未参与计算或尚未反传时，grad 可能为 None。"""
    gradients = tensor.grad
    if gradients is None:
        raise RuntimeError("该张量尚未得到梯度，请检查 backward() 和计算图")
    return gradients


def sum_losses(losses: dict[str, torch.Tensor]) -> torch.Tensor:
    """把各项标量损失相加，明确返回可反向传播的张量。"""
    return torch.stack(tuple(losses.values())).sum()


def verify_manual_calculations():
    print("一、核对手算例子")
    scores = torch.tensor([[0.8, 0.2, 0.1]], requires_grad=True)
    labels = torch.tensor([[0, -1, -1]], dtype=torch.long)
    margin_loss = nn.MultiLabelMarginLoss()(scores, labels)
    torch.testing.assert_close(margin_loss, torch.tensor((0.4 + 0.3) / 3))
    margin_loss.backward()
    score_gradients = get_gradient(scores)
    torch.testing.assert_close(score_gradients, torch.tensor([[-2/3, 1/3, 1/3]]))
    print(f"单正类间隔损失：{margin_loss.item():.6f}，应为 0.233333")
    print("对输入分数的梯度：", score_gradients.tolist())

    multi_positive_loss = nn.MultiLabelMarginLoss()(
        scores.detach(), torch.tensor([[0, 2, -1]], dtype=torch.long)
    )
    torch.testing.assert_close(multi_positive_loss, torch.tensor((0.4 + 1.1) / 3))
    print(f"多正类间隔损失：{multi_positive_loss.item():.6f}，应为 0.500000")

    logits = torch.tensor([[2.0, 1.0, 0.0]], requires_grad=True)
    cross_entropy = nn.CrossEntropyLoss()(logits, torch.tensor([0]))
    probabilities = logits.detach().softmax(dim=1)
    cross_entropy.backward()
    expected_gradients = probabilities - torch.tensor([[1.0, 0.0, 0.0]])
    logit_gradients = get_gradient(logits)
    torch.testing.assert_close(logit_gradients, expected_gradients)
    print(f"交叉熵：{cross_entropy.item():.6f}，应约为 0.407606")
    print("概率：", probabilities.tolist())
    print("交叉熵对原始分数的梯度：", logit_gradients.tolist())

    binary_logits = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
    binary_labels = torch.tensor([[1.0, 0.0, 1.0]])
    binary_loss = nn.BCEWithLogitsLoss()(binary_logits, binary_labels)
    probability_loss = nn.BCELoss()(binary_logits.sigmoid(), binary_labels)
    torch.testing.assert_close(binary_loss, probability_loss)
    binary_loss.backward()
    binary_logit_gradients = get_gradient(binary_logits)
    torch.testing.assert_close(binary_logit_gradients, torch.tensor([[-1/6, 1/6, -1/6]]))
    print(f"二元交叉熵：{binary_loss.item():.6f}，应约为 0.693147")
    print("按三个元素取平均后的梯度：", binary_logit_gradients.tolist())


class SmallRelationClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_layer = nn.Sequential(nn.Linear(4, 8), nn.Tanh())
        self.attention_head = nn.Linear(8, 3)
        self.spatial_head = nn.Linear(8, 6)
        self.contacting_head = nn.Linear(8, 17)

    def forward(self, relation_features: torch.Tensor) -> dict[str, torch.Tensor]:
        shared_features = self.shared_layer(relation_features)
        attention_logits = self.attention_head(shared_features)
        spatial_logits = self.spatial_head(shared_features)
        contacting_logits = self.contacting_head(shared_features)
        return {
            "attention_distribution": attention_logits,
            "spatial_distribution": spatial_logits.sigmoid(),
            "contacting_distribution": contacting_logits.sigmoid(),
        }


def train_small_model(loss_choice):
    print("\n二、训练三组关系分类器；这里用人工特征，所有计算都在 CPU 上")
    torch.manual_seed(7)
    # 每行对应一个已经选定的物体对；数字是教学构造的特征。
    relation_features = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
    ])
    attention_labels = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    spatial_positive_classes = [[0, 2], [1], [2, 4], [0], [3, 5], [1, 2]]
    contacting_positive_classes = [[0, 3], [1], [2, 4], [0, 5], [6], [1, 7]]
    if loss_choice == "margin":
        spatial_labels = build_margin_labels(spatial_positive_classes, 6)
        contacting_labels = build_margin_labels(contacting_positive_classes, 17)
        multilabel_loss_fn = nn.MultiLabelMarginLoss()
    else:
        spatial_labels = build_binary_labels(spatial_positive_classes, 6)
        contacting_labels = build_binary_labels(contacting_positive_classes, 17)
        multilabel_loss_fn = nn.BCELoss()

    model = SmallRelationClassifier()
    cross_entropy_loss_fn = nn.CrossEntropyLoss()
    # 教学数据小，用较大的学习率方便观察；这不是官方训练配置。
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    def compute_losses(outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            "注意关系损失": cross_entropy_loss_fn(outputs["attention_distribution"], attention_labels),
            "空间关系损失": multilabel_loss_fn(outputs["spatial_distribution"], spatial_labels),
            "接触关系损失": multilabel_loss_fn(outputs["contacting_distribution"], contacting_labels),
        }

    initial_outputs: dict[str, torch.Tensor] = model(relation_features)
    for name, tensor in initial_outputs.items():
        print(f"{name} 的形状：{list(tensor.shape)}")
    print("注意标签形状与类型：", list(attention_labels.shape), attention_labels.dtype)
    print("第一对物体的空间标签：", spatial_labels[0].tolist())
    print("第一对物体的接触标签：", contacting_labels[0].tolist())
    initial_total_loss = sum_losses(compute_losses(initial_outputs)).item()
    initial_weights = model.spatial_head.weight.detach().clone()

    model.train()
    for step_index in range(1, 101):
        optimizer.zero_grad()
        outputs: dict[str, torch.Tensor] = model(relation_features)
        losses = compute_losses(outputs)
        total_loss = sum_losses(losses)
        total_loss.backward()
        gradient_norm_before_clipping = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        if step_index == 1:
            shared_linear_layer = model.shared_layer[0]
            assert isinstance(shared_linear_layer, nn.Linear)
            shared_weight_gradients = get_gradient(shared_linear_layer.weight)
            spatial_weight_gradients = get_gradient(model.spatial_head.weight)
            print("共享层权重梯度的形状：", list(shared_weight_gradients.shape))
            print("共享层权重梯度范数：", shared_weight_gradients.norm().item())
            print("空间头权重梯度范数：", spatial_weight_gradients.norm().item())
            print("全体梯度裁剪前的范数：", gradient_norm_before_clipping.item())
        optimizer.step()
        if step_index in (1, 20, 50, 100):
            loss_text = "，".join(f"{name}={value.item():.4f}" for name, value in losses.items())
            print(f"第 {step_index:3d} 次更新前：总损失={total_loss.item():.4f}，{loss_text}")

    model.eval()
    with torch.no_grad():
        final_outputs: dict[str, torch.Tensor] = model(relation_features)
        final_total_loss = sum_losses(compute_losses(final_outputs)).item()
        print("注意关系预测类别：", final_outputs["attention_distribution"].argmax(dim=1).tolist())
    assert final_total_loss < initial_total_loss, "本例的固定训练结果应该让损失下降"
    assert not torch.equal(initial_weights, model.spatial_head.weight.detach())
    print(f"初始损失 {initial_total_loss:.6f} → 最终损失 {final_total_loss:.6f}")
    print("空间头参数已发生更新：", not torch.equal(initial_weights, model.spatial_head.weight.detach()))

    # 输出仅保存到学习文件夹内，便于重跑和检查。
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / f"04_标签损失与训练_权重_{loss_choice}.pt"
    torch.save({"state_dict": model.state_dict()}, save_path)
    loaded_model = SmallRelationClassifier()
    checkpoint = torch.load(save_path, map_location="cpu", weights_only=True)
    loaded_model.load_state_dict(checkpoint["state_dict"])
    loaded_model.eval()
    with torch.no_grad():
        loaded_outputs: dict[str, torch.Tensor] = loaded_model(relation_features)
    for name in final_outputs:
        torch.testing.assert_close(loaded_outputs[name], final_outputs[name])
    print("权重已保存，加载后输出一致：", save_path)


def verify_loss_mask():
    print("\n三、一个独立二元损失被屏蔽后，其直接梯度为零")
    logits = torch.zeros((1, 3), requires_grad=True)
    labels = torch.tensor([[1.0, 0.0, 1.0]])
    mask = torch.tensor([[0.0, 1.0, 0.0]])  # 本例中 1 表示屏蔽。
    keep_mask = 1.0 - mask
    element_losses = nn.BCEWithLogitsLoss(reduction="none")(logits, labels)
    valid_loss = (element_losses * keep_mask).sum() / keep_mask.sum().clamp_min(1.0)
    valid_loss.backward()
    logit_gradients = get_gradient(logits)
    torch.testing.assert_close(logit_gradients, torch.tensor([[-0.25, 0.0, -0.25]]))
    print("逐项损失：", element_losses.tolist())
    print("掩码：", mask.tolist())
    print("原始分数梯度：", logit_gradients.tolist())
    print("这是独立二元损失的教学示例；CE 和间隔损失内的类别相互耦合。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="三组关系标签与训练的中文练习")
    parser.add_argument("--loss", choices=["margin", "bce"], default="margin")
    args = parser.parse_args()
    torch.set_num_threads(1)
    verify_manual_calculations()
    train_small_model(args.loss)
    verify_loss_mask()
