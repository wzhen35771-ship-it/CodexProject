import torch

x = torch.tensor(3.0, requires_grad=True)
for step in range(2):
    if x.grad is not None:
        x.grad.zero_()
    before = x.detach().clone()
    y = 2 * x**2 + 5 * x
    y.backward()
    print('步数、参数、函数值、梯度：', step, x.item(), y.item(), x.grad.item())
    assert torch.equal(x.detach(), before)
    with torch.no_grad():
        x -= 0.1 * x.grad
    print('更新后的参数：', x.item())
