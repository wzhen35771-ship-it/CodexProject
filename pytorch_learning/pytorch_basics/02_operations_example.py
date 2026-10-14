import torch

x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
print('逐元素乘法：', x * x)
print('每列总和：', x.sum(dim=0))
print('每行总和：', x.sum(dim=1))
print('保留行轴：', x.sum(dim=0, keepdim=True).shape)
offset = torch.tensor([10., 20., 30.])
print('广播相加：', x + offset)
w = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
print('矩阵乘法：', x @ w)
assert torch.equal(x.sum(dim=0), torch.tensor([5., 7., 9.]))
