"""第五课补充：把模型、损失、梯度和一次参数更新连起来。"""

import torch
from torch import nn

# 固定初始化，方便重复观察。
torch.manual_seed(7)

# 准备模型：每条样本的特征数依次为 2 → 4 → 3。
model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 3))

# 一条样本、两个特征；正确类别编号是 0。
features = torch.tensor([[-0.6, 0.0]])
labels = torch.tensor([0], dtype=torch.long)

# 损失计算器负责评价预测；优化器负责调整模型的权重和偏置。
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 第 1 步：清除旧梯度。
optimizer.zero_grad()

# 第 2 步：模型根据输入计算三个类别的得分。
logits = model(features)

# 第 3 步：根据得分和正确标签计算损失。
loss = criterion(logits, labels)
print("更新前的得分：", logits.detach())
print("更新前的损失：", loss.item())

# 第 4 步：计算损失对模型参数的梯度；此时参数尚未更新。
loss.backward()
# model[2] 是 Sequential 中第三个模块，即最后的 Linear(4, 3)。
print("最后一层偏置的梯度：", model[2].bias.grad)

# 第 5 步：优化器根据梯度更新模型参数。
optimizer.step()

# 更新后重新进行前向计算，才能得到新的得分和损失。
with torch.no_grad():
    updated_logits = model(features)
    updated_loss = criterion(updated_logits, labels)

print("更新后的得分：", updated_logits)
print("更新后的损失：", updated_loss.item())
