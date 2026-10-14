"""独立练习：先写答案，再查看示例。"""

import torch

# 1. 创建三行两列的张量，数字依次为 1 到 6。
# 在下面编写代码并记录实际输出。
x = torch.tensor([[1, 2], [3, 4], [5, 6]])
# 2. 打印 shape、ndim、numel、dtype、device，并用中文说明含义。
# 在下面编写代码并记录实际输出。
print("shape:",x.shape)  # shape: torch.Size([3, 2])  张量的形状
print("ndim:",x.ndim)    # ndim: 2  张量的维度数
print("numel:",x.numel())  # numel: 6  张量 中元素的总数
print("dtype:",x.dtype)  # dtype: torch.int64  张量中元素的数据类型
print("device:",x.device)  # device: cpu  张量所在的设备（CPU 或 GPU）
# 3. 取第二行、第一列和第三行第二列；先预测再运行。
# 在下面编写代码并记录实际输出。
print("第二行:", x[1])  # 第二行: tensor([3, 4])
print("第一列:", x[:, 0])  # 第一列: tensor([1, 3, 5])
print("第三行第二列:", x[2, 1])  # 第三行第二列: tensor(6)