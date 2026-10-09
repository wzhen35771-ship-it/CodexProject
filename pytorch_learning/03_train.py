"""第三课：用梯度更新参数，完成一个最小训练过程。"""
import torch

# 数据是固定的；只有参数 w 需要学习，所以只有 w 开启求导。
x = torch.tensor(2.0)
target = torch.tensor(6.0)  # 目标值：希望模型预测得到 6。
w = torch.tensor(1.0, requires_grad=True)  # 参数的初始猜测。
learning_rate = 0.1  # 学习率：控制每次调整的步长。

for step in range(5):
    prediction = w * x  # 预测：输入乘以参数。
    loss = (prediction - target) ** 2  # 损失：预测与目标的差的平方。
    loss.backward()  # 计算损失对参数 w 的梯度，并保存到 w.grad。

    # 记录更新前的数据，方便比较。
    old_w = w.item()
    old_prediction = prediction.item()
    old_loss = loss.item()
    gradient = w.grad.item()

    # 更新参数时不记录新的求导过程；w 的求导设置仍然保留。
    with torch.no_grad():
        w -= learning_rate * w.grad
        new_prediction = w * x  # 参数更新后必须重新计算预测。
        new_loss = (new_prediction - target) ** 2

    print(f"第 {step + 1} 次更新")
    print(f"  参数：{old_w:.6f} → {w.item():.6f}")
    print(f"  预测：{old_prediction:.6f} → {new_prediction.item():.6f}")
    print(f"  损失：{old_loss:.6f} → {new_loss.item():.6f}")
    print(f"  本次梯度：{gradient:.6f}")

    w.grad.zero_()  # 清除本次梯度，避免下一次求导时累加。

print("\n训练结束：参数逐渐接近 3，预测逐渐接近目标值 6。")
print("练习一：只运行第一次更新，手算并解释输出。")
print("练习二：把参数初始值改成 4.0，先预测第一次更新方向，再运行。")
print("练习三：在自己的文件中独立写出五次更新的训练过程。")
