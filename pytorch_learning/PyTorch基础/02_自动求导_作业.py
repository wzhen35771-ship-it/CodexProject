# 1. 创建值为 3.0 的张量 x，开启求导。
# 2. 计算 `y = 2 * x ** 2 + 5 * x`。
# 3. 先手算 y 和导数，再运行代码核对。
# 4. 打印求导前后的 x，确认它没有被更新。
import torch
x=torch.tensor(3.0,requires_grad=True)
y=2*x**2+5*x
y.backward()
print("y:",y.item())
print("x.grad:",x.grad.item())
print("x:",x.item())

