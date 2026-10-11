"""第四课：让优化器根据梯度更新参数。"""
import torch

x = torch.tensor(2.0)
target = torch.tensor(6.0)
w = torch.tensor(1.0, requires_grad=True)

# 优化器管理列表中的参数；lr 是学习率，创建它并不会立即更新参数。
optimizer = torch.optim.SGD([w], lr=0.1)

for step in range(5):
    optimizer.zero_grad()  # 一、清除上一次保存的梯度。
    prediction = w * x  # 使用当前参数重新预测。
    loss = (prediction - target) ** 2  # 计算本轮损失。
    loss.backward()  # 二、计算梯度，保存到 w.grad。

    old_w = w.item()
    gradient = w.grad.item()
    optimizer.step()  # 三、优化器根据梯度更新 w。

    print(f"第 {step + 1} 次更新")
    print(f"  本次更新前的预测：{prediction.item():.6f}")
    print(f"  本次更新前的损失：{loss.item():.6f}")
    print(f"  本次梯度：{gradient:.6f}")
    print(f"  参数变化：{old_w:.6f} → {w.item():.6f}")

with torch.no_grad():
    print("\n训练结束后的预测：", (w * x).item())

print("练习：把 w 的初始值改成 4.0，先判断第一次更新方向，再运行核对。")
