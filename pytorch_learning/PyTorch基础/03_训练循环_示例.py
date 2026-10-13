"""CPU 人工关系分类器：训练、评估、少量数据拟合、保存和加载。

输入是已知位置差，不是像素。训练准确率不是真实场景图评价结果。
"""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class SmallRelationClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(2, 8)
        self.output_layer = nn.Linear(8, 3)

    def forward(self, relation_features):
        hidden = torch.relu(self.input_layer(relation_features))
        return self.output_layer(hidden)


def make_artificial_data():
    # dx/dy = 主语中心 - 宾语中心；图像 y 轴向下增大。
    # 本例只构造三个互斥类别：左侧、右侧、上方。
    features = torch.tensor(
        [
            [-0.8, 0.0], [-0.6, 0.1], [-0.7, -0.1], [-0.5, 0.0],
            [0.8, 0.0], [0.6, 0.1], [0.7, -0.1], [0.5, 0.0],
            [0.0, -0.8], [0.1, -0.6], [-0.1, -0.7], [0.0, -0.5],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0] * 4 + [1] * 4 + [2] * 4, dtype=torch.long)
    return features, labels


def evaluate(model, features, labels, criterion):
    # eval 切换层的模式；no_grad 负责关闭梯度记录。
    model.eval()
    with torch.no_grad():
        logits = model(features)
        loss = criterion(logits, labels).item()
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == labels).float().mean().item()
    return loss, accuracy, logits


def main():
    parser = argparse.ArgumentParser(description="在 12 条人工关系数据上练习训练循环")
    parser.add_argument("--epochs", type=int, default=300, help="最大训练轮数，默认 300")
    parser.add_argument("--batch-size", type=int, default=4, help="批次大小，默认 4")
    arguments = parser.parse_args()
    if arguments.epochs < 1 or arguments.batch_size < 1:
        parser.error("epochs 和 batch-size 必须为正整数")

    torch.manual_seed(7)
    torch.set_num_threads(1)
    features, labels = make_artificial_data()
    dataset = TensorDataset(features, labels)
    generator = torch.Generator().manual_seed(7)
    loader = DataLoader(
        dataset, batch_size=arguments.batch_size, shuffle=True, generator=generator
    )
    model = SmallRelationClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    print("PyTorch 版本：", torch.__version__)
    print("设备：CPU；样本数：", len(dataset), "每轮 batch 数：", len(loader))
    print("特征形状：", tuple(features.shape), "标签形状：", tuple(labels.shape))
    print("参数总数：", sum(parameter.numel() for parameter in model.parameters()))
    initial_loss, initial_accuracy, _ = evaluate(model, features, labels, criterion)
    print(f"训练前：loss={initial_loss:.6f}，训练准确率={initial_accuracy:.1%}")

    updates = 0
    for epoch in range(1, arguments.epochs + 1):
        model.train()
        for batch_features, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            if not torch.isfinite(loss).item():
                raise RuntimeError("损失不是有限值，请检查数据与计算路径")
            loss.backward()

            # 第一个 batch 用于观察梯度；后续每个 batch 仍做同样的更新。
            if updates == 0:
                print(
                    "首个 batch：输入", tuple(batch_features.shape),
                    "logits", tuple(logits.shape), "标签", tuple(batch_labels.shape),
                    "损失形状", tuple(loss.shape),
                )
                for name, parameter in model.named_parameters():
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                        raise RuntimeError(f"参数 {name} 的梯度缺失或不是有限值")
                    print(f"{name}：梯度范数={parameter.grad.norm().item():.6f}")
            optimizer.step()
            updates += 1

        train_loss, train_accuracy, _ = evaluate(model, features, labels, criterion)
        if epoch == 1 or epoch % 25 == 0:
            print(f"epoch={epoch:3d}：loss={train_loss:.6f}，训练准确率={train_accuracy:.1%}")
        if epoch >= 50 and train_accuracy == 1.0 and train_loss < 0.01:
            print(f"第 {epoch} 轮达到教学拟合条件，结束训练。")
            break

    train_loss, train_accuracy, original_logits = evaluate(model, features, labels, criterion)
    print(f"最终：loss={train_loss:.6f}，训练准确率={train_accuracy:.1%}，更新次数={updates}")
    if train_accuracy < 1.0 or train_loss >= 0.05:
        raise RuntimeError("尚未通过少量数据拟合检查；核对设置或增加训练轮数后重试")

    # 仅保存通过拟合检查的教学权重。路径以脚本目录为准。
    output_directory = Path(__file__).resolve().parent / "outputs"
    output_directory.mkdir(exist_ok=True)
    checkpoint_path = output_directory / "03_训练循环_权重.pt"
    torch.save(model.state_dict(), checkpoint_path)

    # 只加载刚由本程序保存的权重，重新创建相同结构后核对输出。
    loaded_model = SmallRelationClassifier()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    loaded_model.load_state_dict(state)
    _, loaded_accuracy, loaded_logits = evaluate(loaded_model, features, labels, criterion)
    if not torch.allclose(original_logits, loaded_logits, rtol=1e-6, atol=1e-7):
        raise RuntimeError("加载后的 logits 与保存前不一致")
    print("加载后训练准确率：", f"{loaded_accuracy:.1%}")
    print("预测编号：", loaded_logits.argmax(dim=1).tolist())
    print("真实编号：", labels.tolist())
    print("权重路径：", checkpoint_path)
    print("保存加载检查通过。这是人工训练集拟合，真实基线与真实效果尚未涉及。")


if __name__ == "__main__":
    main()
