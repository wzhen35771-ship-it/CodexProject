"""第二课：自动求导。先预测输出，再运行核对。"""
import torch

# 一、一个变量的求导。
# 使用浮点数；requires_grad=True 表示需要跟踪运算，以便计算梯度。
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x
print("一、一个变量的求导")
print("输入 x：", x.item())
print("计算结果 y：", y.item())
print("反向传播之前的梯度：", x.grad)  # 此时尚未计算，输出 None。

y.backward()  # 计算 y 对 x 的导数，结果存入 x.grad。
print("反向传播之后的梯度：", x.grad.item())
print("求导不会改变 x 的值：", x.item())

# 二、多个元素的求导。先求和，让结果成为一个标量。
v = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
total = (v ** 2).sum()
total.backward()
print("\n二、多个元素的求导（可选）")
print("各元素平方后求和：", total.item())
print("每个元素对应的梯度：", v.grad)

# 三、梯度默认会累加；重新计算表达式，建立新的计算图。
print("\n三、梯度的累加（可选）")
new_y = x ** 2 + 3 * x
new_y.backward()
print("第二次求导后的累积梯度：", x.grad.item())
x.grad.zero_()  # 将已保存的梯度清零，x 本身不变。
print("清零之后的梯度：", x.grad.item())

print("\n练习：新建一个需要求导的张量 x，值为 3.0。")
print("计算 y = 2 * x ** 2 + 5 * x，先预测 y 和梯度，再运行核对。")
print("说明：本课只计算梯度，没有更新参数，也没有训练模型。")
