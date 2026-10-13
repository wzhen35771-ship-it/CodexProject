"""手算求导、梯度累加与两次 SGD 更新。全部使用 CPU 标量。"""

import math

import torch


def main():
    value = torch.tensor(3.0, requires_grad=True)
    result = 2 * value**2 + 5 * value
    print("求导前：x =", value.item(), "y =", result.item(), "grad =", value.grad)
    result.backward()
    print("第一次求导后：x =", value.item(), "grad =", value.grad.item())
    assert result.item() == 33.0
    assert value.grad.item() == 17.0
    assert value.item() == 3.0

    # 重做前向会得到新的计算图，但已有梯度仍会累加。
    new_result = 2 * value**2 + 5 * value
    new_result.backward()
    print("未清零，第二次求导后的 grad =", value.grad.item())
    assert value.grad.item() == 34.0

    value.grad.zero_()
    third_result = 2 * value**2 + 5 * value
    third_result.backward()
    print("清零后重新求导的 grad =", value.grad.item())
    assert value.grad.item() == 17.0

    # 将标量声明为可训练参数，使用最简单的 SGD。
    weight = torch.nn.Parameter(torch.tensor(3.0))
    optimizer = torch.optim.SGD([weight], lr=0.1)
    expected_values = [(3.0, 4.0, 4.0, 2.6), (2.6, 2.56, 3.2, 2.28)]

    for iteration, expected in enumerate(expected_values, start=1):
        optimizer.zero_grad(set_to_none=True)
        before = weight.item()
        loss = (weight - 1.0) ** 2
        loss.backward()
        after_backward = weight.item()
        gradient = weight.grad.item()
        optimizer.step()
        after_step = weight.item()
        print(
            f"第 {iteration} 次：更新前={before:.4f}，损失={loss.item():.4f}，"
            f"梯度={gradient:.4f}，求导后={after_backward:.4f}，更新后={after_step:.4f}"
        )
        actual_values = (before, loss.item(), gradient, after_step)
        assert after_backward == before
        for actual, target in zip(actual_values, expected):
            assert math.isclose(actual, target, rel_tol=1e-6, abs_tol=1e-6)

    # 评估计算无需记录梯度；原参数的 requires_grad 不受影响。
    with torch.no_grad():
        prediction = weight * 2.0
    print("no_grad 内输出是否需要梯度：", prediction.requires_grad)
    print("原参数是否仍需要梯度：", weight.requires_grad)
    assert not prediction.requires_grad
    assert weight.requires_grad
    print("求导、累加与更新检查通过。请独立重写两次更新过程。")


if __name__ == "__main__":
    main()
