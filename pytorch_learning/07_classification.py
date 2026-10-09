"""第七课：把物体对的两个特征，分类为“没有拿着”或“拿着”。"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


torch.manual_seed(2026)  # 固定随机种子，方便重复观察。
torch.set_num_threads(1)  # 小练习使用一个处理器线程即可。

# 每行是一对物体：第一列是接触程度，第二列是归一化后的距离。
# 这些数值和标签是人为编写的教学数据，没有读取图片。
train_features = torch.tensor([
    [0.1, 0.9], [0.2, 0.8], [0.0, 1.0], [0.3, 0.7],
    [0.9, 0.1], [0.8, 0.2], [1.0, 0.0], [0.7, 0.3],
], dtype=torch.float32)
train_labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)

# 验证数据不参与参数更新，用来观察新样本上的预测。
val_features = torch.tensor([
    [0.15, 0.85], [0.25, 0.75], [0.85, 0.15], [0.75, 0.25],
], dtype=torch.float32)
val_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

dataset = TensorDataset(train_features, train_labels)
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
model = nn.Linear(2, 2)  # 两个输入特征，输出两个类别的原始得分。
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)


def evaluate(features, labels):
    """只计算指标，不求导，也不更新参数。"""
    model.eval()
    with torch.no_grad():
        scores = model(features)
        loss = criterion(scores, labels)
        predictions = scores.argmax(dim=1)
        accuracy = (predictions == labels).float().mean()
    return loss.item(), accuracy.item()


print("类别约定：0=没有拿着，1=拿着")
print("训练输入形状：", list(train_features.shape))
print("训练标签形状：", list(train_labels.shape))
print("每批样本数：4；每轮批次数：", len(loader))
with torch.no_grad():
    print("一个批次的输出形状：", list(model(train_features[:4]).shape))

initial_loss, initial_accuracy = evaluate(val_features, val_labels)
print(f"训练前验证损失：{initial_loss:.4f}；验证正确率：{initial_accuracy:.0%}")

for epoch in range(1, 31):
    model.train()
    for features, labels in loader:
        optimizer.zero_grad()
        scores = model(features)
        loss = criterion(scores, labels)  # 直接传原始得分，不提前做概率转换。
        loss.backward()
        optimizer.step()

    if epoch in (1, 10, 20, 30):
        train_loss, train_accuracy = evaluate(train_features, train_labels)
        val_loss, val_accuracy = evaluate(val_features, val_labels)
        print(
            f"第{epoch:02d}轮：训练损失={train_loss:.4f}，"
            f"训练正确率={train_accuracy:.0%}；"
            f"验证损失={val_loss:.4f}，验证正确率={val_accuracy:.0%}"
        )

model.eval()
with torch.no_grad():
    final_scores = model(val_features)
    final_predictions = final_scores.argmax(dim=1)
print("验证预测类别：", final_predictions.tolist())
print("验证目标类别：", val_labels.tolist())
print("课堂小结：模型输出得分；损失负责训练；最大得分的位置决定预测类别。")
