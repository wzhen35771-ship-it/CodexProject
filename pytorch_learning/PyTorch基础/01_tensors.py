"""第一课：张量。使用学习环境中的 Python 运行，所有示例均在处理器上执行。"""
import torch

print("PyTorch 版本：", torch.__version__)

# 每行代表一个物体，每个物体有三个示例特征。
x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print("\n一、张量：", x)
print("形状：", x.shape)  # 两个物体，每个物体有三个特征。
print("数据类型：", x.dtype)  # torch.float32 表示 32 位浮点数。
print("运行设备：", x.device)  # cpu 表示使用处理器。

# 索引从零开始；冒号表示这个维度上的全部元素。
print("\n二、第一个物体的特征：", x[0])
print("每个物体的第二个特征：", x[:, 1])

# 对每个元素进行运算，或沿指定维度求平均值。
print("\n三、每个元素加十：", x + 10)
print("每个元素乘二：", x * 2)
print("每个物体的特征平均值：", x.mean(dim=1))

# 矩阵乘法：形状为 [2, 3] 的矩阵乘以 [3, 2] 的矩阵，得到 [2, 2]。
# 字母 w 表示权重矩阵；scores 保存计算得到的示例得分。
w = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
scores = x @ w
print("\n四、计算得到的示例得分：", scores)
print("得分张量的形状：", scores.shape)

print("\n练习一：把 x 改成三行四列，解释它的形状。")
print("练习二：先预测 x.sum(dim=0) 和 x.sum(dim=1) 的结果，再运行核对。")
print("练习三：解释为什么当前示例中 x * x 可以运行，而 x @ x 会报错。")
