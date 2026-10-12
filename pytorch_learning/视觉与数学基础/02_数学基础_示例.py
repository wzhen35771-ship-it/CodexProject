"""视觉与数学基础：小规模 CPU 算例，可按部分运行。

先手算、再运行，最后改一个数字观察结果。无需网络或数据集。
"""

import argparse

import torch
from torch.nn import functional as F


def print_heading(text):
    print(f"\n{'=' * 12} {text} {'=' * 12}")


def demonstrate_tensor_shapes():
    print_heading("一、标量、向量、矩阵与图片形状")
    scalar = torch.tensor(3.0)
    vector = torch.tensor([1., 2., 3.])
    matrix = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    image_batch = torch.zeros(5, 3, 16, 32)
    for name, tensor in [("标量", scalar), ("向量", vector), ("矩阵", matrix), ("图片批次", image_batch)]:
        print(f"{name}：形状 {list(tensor.shape)}，{tensor.ndim} 个维度，{tensor.numel()} 个元素")

    values = torch.tensor([1., 2., 3., 4.])
    print("四个元素原形状：", list(values.shape))
    print("增加行维度：", list(values.unsqueeze(0).shape))
    print("增加列维度：", list(values.unsqueeze(1).shape))
    print("练习：如果图片数量改为 8、高度改为 32，形状和总元素数是什么？")


def demonstrate_vector_operations():
    print_heading("二、向量加减、点积与模长")
    a = torch.tensor([1., 2., 3.])
    b = torch.tensor([4., 5., 6.])
    print("向量 a：", a)
    print("向量 b：", b)
    print("对应位置相加：", a + b)
    print("对应位置相减：", a - b)
    print("对应位置相乘：", a * b)
    print("点积（乘后求和）：", (a @ b).item())
    print("用求和验证点积：", (a * b).sum().item())
    v = torch.tensor([3., 4.])
    print("[3,4] 的模长：", torch.linalg.vector_norm(v).item())
    print("练习：换成 a=[2,3]、b=[4,1]，先手算逐元素乘和点积。")


def demonstrate_matrix_operations():
    print_heading("三、矩阵乘法与线性层")
    inputs = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    weights = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    b = torch.tensor([0.5, -0.5])
    print("输入 X：\n", inputs)
    print("数学写法的权重 W：\n", weights)
    print("形状：", list(inputs.shape), "@", list(weights.shape))
    print("矩阵乘积：\n", inputs @ weights)
    print("转置后的输入 X：\n", inputs.T)
    print("每个样本加相同偏置后的结果：\n", inputs @ weights + b)

    linear_layer = torch.nn.Linear(3, 2)
    with torch.no_grad():
        linear_layer.weight.copy_(weights.T)
        linear_layer.bias.copy_(b)
    print("PyTorch 线性层实际权重形状：", list(linear_layer.weight.shape))
    print("线性层输出：\n", linear_layer(inputs).detach())
    print("与手算矩阵公式相同：", torch.allclose(linear_layer(inputs), inputs @ weights + b))
    print("练习：把数学写法 W 的第一列改成 [1,1,1]，第一列输出怎样变化？")


def demonstrate_broadcasting():
    print_heading("四、广播既方便，也可能掩盖形状错误")
    predictions = torch.tensor([[1.], [2.], [3.], [4.]])
    original_targets = torch.tensor([1., 2., 3., 4.])
    incorrect_difference = predictions - original_targets
    print("预测形状：", list(predictions.shape))
    print("未调整的目标形状：", list(original_targets.shape))
    print("直接相减后的形状：", list(incorrect_difference.shape))
    print("直接相减：\n", incorrect_difference)
    print("错误的平均平方损失：", (incorrect_difference ** 2).mean().item())
    correct_targets = original_targets.unsqueeze(1)
    print("修正后的目标形状：", list(correct_targets.shape))
    print("正确的平均平方损失：", F.mse_loss(predictions, correct_targets).item())
    print("练习：用 [6,1] 和 [6] 的例子检查结果形状，不要只看元素数。")


def demonstrate_statistics():
    print_heading("五、均值、总体方差与标准差")
    values = torch.tensor([1., 2., 3.])
    mean = values.mean()
    variance = ((values - mean) ** 2).mean()
    std = variance.sqrt()
    standardized_values = (values - mean) / std
    print("数值：", values)
    print("均值：", mean.item())
    print("与均值的差：", values - mean)
    print(f"总体方差（分母是元素数）：{variance.item():.4f}")
    print(f"标准差：{std.item():.4f}")
    print("标准化结果：", standardized_values)
    print("标准化后均值：", standardized_values.mean().item())
    print("说明：这里用显式公式计算总体方差，避免不同方差校正约定的混淆。")


def demonstrate_autograd():
    print_heading("六、链式法则、梯度与一步更新")
    x = torch.tensor(2.0)
    targets = torch.tensor(6.0)

    # 从相同初始值分别做一次更新，不要把两个实验的梯度混在一起。
    for learning_rate in [0.1, 0.5]:
        w = torch.tensor(1.0, requires_grad=True)
        predictions = w * x
        loss = (predictions - targets) ** 2
        manual_gradient = 2 * (predictions.detach() - targets) * x
        print(f"\n学习率 {learning_rate}，初始参数 {w.item()}")
        print("初始预测、初始损失：", predictions.item(), loss.item())
        print("反向传播之前的梯度：", w.grad)
        loss.backward()
        print("手算公式 2×(预测-目标)×输入：", manual_gradient.item())
        print("自动求导保存的梯度：", w.grad.item())

        with torch.no_grad():
            w -= learning_rate * w.grad
            updated_predictions = w * x
            updated_loss = (updated_predictions - targets) ** 2
        print(f"更新后的参数：{w.item():.4f}，预测：{updated_predictions.item():.4f}，损失：{updated_loss.item():.4f}")
        w.grad.zero_()
        print(f"清零后的梯度：{w.grad.item():.4f}；参数仍是：{w.item():.4f}")

    print("\n练习：把初始 w 改成 4、学习率设为 0.1，先手算再运行。")


def demonstrate_loss_gradients():
    print_heading("七、多个参数的梯度与平均平方损失")
    parameters = torch.tensor([2., 3.], requires_grad=True)
    total_loss = (parameters ** 2).sum()
    total_loss.backward()
    print("参数：[2,3]；平方和损失：", total_loss.item())
    print("每个参数自己的梯度：", parameters.grad)

    predictions = torch.tensor([2., 4.], requires_grad=True)
    targets = torch.tensor([1., 6.])
    mean_loss = F.mse_loss(predictions, targets)
    mean_loss.backward()
    print("\n预测：", predictions.detach(), "；目标：", targets)
    print("平均平方损失：", mean_loss.item())
    print("损失对预测值的梯度：", predictions.grad)
    print("公式：2×(预测-目标)÷元素数")
    print("练习：对新张量使用 reduction='sum'，观察损失及梯度放大多少。")


def demonstrate_classification_losses():
    print_heading("八、原始分数、概率与分类损失")
    logits = torch.tensor([[2., 1., 0.]], requires_grad=True)
    class_targets = torch.tensor([0], dtype=torch.long)
    probabilities = torch.softmax(logits.detach(), dim=1)
    cross_entropy = F.cross_entropy(logits, class_targets)
    cross_entropy.backward()

    print("单标签三分类")
    print("原始分数：", logits.detach())
    print("类别概率：", probabilities)
    print("概率的和：", probabilities.sum(dim=1))
    print("预测类别编号：", logits.detach().argmax(dim=1))
    print("真实类别编号、类型：", class_targets, class_targets.dtype)
    print(f"交叉熵（损失直接接收原始分数）：{cross_entropy.item():.4f}")
    print(f"手算 -ln(正确类别概率)：{-probabilities[0, 0].log().item():.4f}")
    print("原始分数的梯度：", logits.grad)

    binary_logits = torch.tensor([0.0], requires_grad=True)
    binary_targets = torch.tensor([1.0])
    binary_loss = F.binary_cross_entropy_with_logits(binary_logits, binary_targets)
    binary_loss.backward()
    print("\n二分类")
    print("原始分数：", binary_logits.detach())
    print("sigmoid 转换后的概率：", torch.sigmoid(binary_logits.detach()))
    print("真实标签、类型：", binary_targets, binary_targets.dtype)
    print(f"二元交叉熵（内部处理 sigmoid）：{binary_loss.item():.4f}")
    print("原始分数的梯度：", binary_logits.grad)
    print("练习：把三分类真实类别改成 2，把二分类真实标签改成 0，观察损失和梯度。")


def main():
    exercises = {
        "形状": demonstrate_tensor_shapes,
        "向量": demonstrate_vector_operations,
        "矩阵": demonstrate_matrix_operations,
        "广播": demonstrate_broadcasting,
        "统计": demonstrate_statistics,
        "求导": demonstrate_autograd,
        "损失": demonstrate_loss_gradients,
        "分类": demonstrate_classification_losses,
    }
    parser = argparse.ArgumentParser(description="由浅入深的视觉数学算例，中文输出，CPU 即可。")
    parser.add_argument("--部分", dest="section", choices=["全部", *exercises], default="全部", help="选择一部分运行，默认运行全部。")
    args = parser.parse_args()
    torch.set_printoptions(precision=4, sci_mode=False)
    print("运行设备：CPU；不下载数据；先手算，再看输出。")
    if args.section == "全部":
        for exercise in exercises.values():
            exercise()
    else:
        exercises[args.section]()
    print("\n完成。请选择一段，改变一个数字，再解释输出为什么变化。")


if __name__ == "__main__":
    main()
