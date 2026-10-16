import torch

a = torch.tensor(3.0)
b = torch.tensor([1.0, 2.0, 3.0])
x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
for name, value in [('a', a), ('b', b), ('x', x)]:
    print(name, value)
    print('形状、维度、元素数：', value.shape, value.ndim, value.numel())
    print('类型、设备：', value.dtype, value.device)
print('第二行：', x[1])
print('最后一列：', x[:, -1])
print('第一行第二列：', x[0, 1].item())
assert x.shape == (2, 3) and x.numel() == 6
