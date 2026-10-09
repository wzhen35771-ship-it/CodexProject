"""第五课：让线性模型从四组数据中学会预测。"""

import torch
from torch import nn


torch.manual_seed(0)  # 固定随机种子，方便重复运行并比较结果

# 每行是一条样本，每条样本只有一个输入数字
x = torch.tensor([[-1.0], [0.0], [1.0], [2.0]])
target = torch.tensor([[-2.0], [1.0], [4.0], [7.0]])  # 目标规律：3×输入+1

model = nn.Linear(1, 1)  # 一个输入特征，输出一个预测值；模型会学习权重和偏置
loss_function = nn.MSELoss()  # 所有预测误差的平方，再取平均
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

print("输入形状：", x.shape)
print("目标形状：", target.shape)
print("初始权重：", round(model.weight.item(), 4))
print("初始偏置：", round(model.bias.item(), 4))

for step in range(100):
    optimizer.zero_grad()  # 每次求导前清除上一次的梯度
    prediction = model(x)  # 四条输入一起交给模型
    loss = loss_function(prediction, target)
    loss.backward()  # 计算损失对模型参数的梯度

    # 这里打印的是本次参数更新前算出的损失
    if step in (0, 9, 49, 99):
        print(f"第 {step + 1} 次更新前的损失：{loss.item():.6f}")

    optimizer.step()  # 优化器更新模型的权重和偏置

# 更新后重新预测，才能看到最终参数对应的结果
with torch.no_grad():
    final_prediction = model(x)
    final_loss = loss_function(final_prediction, target)

print("最终权重：", round(model.weight.item(), 4))
print("最终偏置：", round(model.bias.item(), 4))
print("最终预测：", [round(value, 4) for value in final_prediction[:, 0].tolist()])
print("目标数值：", target[:, 0].tolist())
print(f"最终损失：{final_loss.item():.6f}")
