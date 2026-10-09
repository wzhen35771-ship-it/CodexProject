"""视觉与数学基础：小规模 CPU 算例，可按部分运行。

先手算、再运行，最后改一个数字观察结果。无需网络或数据集。
"""

import argparse

import torch
from torch.nn import functional as F


def 标题(文字):
    print(f"\n{'=' * 12} {文字} {'=' * 12}")


def 形状练习():
    标题("一、标量、向量、矩阵与图片形状")
    标量 = torch.tensor(3.0)
    向量 = torch.tensor([1., 2., 3.])
    矩阵 = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    图片批次 = torch.zeros(5, 3, 16, 32)
    for 名称, 数据 in [("标量", 标量), ("向量", 向量), ("矩阵", 矩阵), ("图片批次", 图片批次)]:
        print(f"{名称}：形状 {list(数据.shape)}，{数据.ndim} 个维度，{数据.numel()} 个元素")

    数值 = torch.tensor([1., 2., 3., 4.])
    print("四个元素原形状：", list(数值.shape))
    print("增加行维度：", list(数值.unsqueeze(0).shape))
    print("增加列维度：", list(数值.unsqueeze(1).shape))
    print("练习：如果图片数量改为 8、高度改为 32，形状和总元素数是什么？")


def 向量练习():
    标题("二、向量加减、点积与模长")
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


def 矩阵练习():
    标题("三、矩阵乘法与线性层")
    X = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    W = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    b = torch.tensor([0.5, -0.5])
    print("输入 X：\n", X)
    print("数学写法的权重 W：\n", W)
    print("形状：", list(X.shape), "@", list(W.shape))
    print("矩阵乘积：\n", X @ W)
    print("转置后的输入 X：\n", X.T)
    print("每个样本加相同偏置后的结果：\n", X @ W + b)

    线性层 = torch.nn.Linear(3, 2)
    with torch.no_grad():
        线性层.weight.copy_(W.T)
        线性层.bias.copy_(b)
    print("PyTorch 线性层实际权重形状：", list(线性层.weight.shape))
    print("线性层输出：\n", 线性层(X).detach())
    print("与手算矩阵公式相同：", torch.allclose(线性层(X), X @ W + b))
    print("练习：把数学写法 W 的第一列改成 [1,1,1]，第一列输出怎样变化？")


def 广播练习():
    标题("四、广播既方便，也可能掩盖形状错误")
    预测 = torch.tensor([[1.], [2.], [3.], [4.]])
    原目标 = torch.tensor([1., 2., 3., 4.])
    错误差值 = 预测 - 原目标
    print("预测形状：", list(预测.shape))
    print("未调整的目标形状：", list(原目标.shape))
    print("直接相减后的形状：", list(错误差值.shape))
    print("直接相减：\n", 错误差值)
    print("错误的平均平方损失：", (错误差值 ** 2).mean().item())
    正确目标 = 原目标.unsqueeze(1)
    print("修正后的目标形状：", list(正确目标.shape))
    print("正确的平均平方损失：", F.mse_loss(预测, 正确目标).item())
    print("练习：用 [6,1] 和 [6] 的例子检查结果形状，不要只看元素数。")


def 统计练习():
    标题("五、均值、总体方差与标准差")
    数值 = torch.tensor([1., 2., 3.])
    均值 = 数值.mean()
    方差 = ((数值 - 均值) ** 2).mean()
    标准差 = 方差.sqrt()
    标准化结果 = (数值 - 均值) / 标准差
    print("数值：", 数值)
    print("均值：", 均值.item())
    print("与均值的差：", 数值 - 均值)
    print(f"总体方差（分母是元素数）：{方差.item():.4f}")
    print(f"标准差：{标准差.item():.4f}")
    print("标准化结果：", 标准化结果)
    print("标准化后均值：", 标准化结果.mean().item())
    print("说明：这里用显式公式计算总体方差，避免不同方差校正约定的混淆。")


def 求导练习():
    标题("六、链式法则、梯度与一步更新")
    x = torch.tensor(2.0)
    目标 = torch.tensor(6.0)

    # 从相同初始值分别做一次更新，不要把两个实验的梯度混在一起。
    for 学习率 in [0.1, 0.5]:
        w = torch.tensor(1.0, requires_grad=True)
        预测 = w * x
        损失 = (预测 - 目标) ** 2
        手算梯度 = 2 * (预测.detach() - 目标) * x
        print(f"\n学习率 {学习率}，初始参数 {w.item()}")
        print("初始预测、初始损失：", 预测.item(), 损失.item())
        print("反向传播之前的梯度：", w.grad)
        损失.backward()
        print("手算公式 2×(预测-目标)×输入：", 手算梯度.item())
        print("自动求导保存的梯度：", w.grad.item())

        with torch.no_grad():
            w -= 学习率 * w.grad
            新预测 = w * x
            新损失 = (新预测 - 目标) ** 2
        print(f"更新后的参数：{w.item():.4f}，预测：{新预测.item():.4f}，损失：{新损失.item():.4f}")
        w.grad.zero_()
        print(f"清零后的梯度：{w.grad.item():.4f}；参数仍是：{w.item():.4f}")

    print("\n练习：把初始 w 改成 4、学习率设为 0.1，先手算再运行。")


def 损失练习():
    标题("七、多个参数的梯度与平均平方损失")
    参数 = torch.tensor([2., 3.], requires_grad=True)
    总损失 = (参数 ** 2).sum()
    总损失.backward()
    print("参数：[2,3]；平方和损失：", 总损失.item())
    print("每个参数自己的梯度：", 参数.grad)

    预测 = torch.tensor([2., 4.], requires_grad=True)
    目标 = torch.tensor([1., 6.])
    平均损失 = F.mse_loss(预测, 目标)
    平均损失.backward()
    print("\n预测：", 预测.detach(), "；目标：", 目标)
    print("平均平方损失：", 平均损失.item())
    print("损失对预测值的梯度：", 预测.grad)
    print("公式：2×(预测-目标)÷元素数")
    print("练习：对新张量使用 reduction='sum'，观察损失及梯度放大多少。")


def 分类练习():
    标题("八、原始分数、概率与分类损失")
    原始分数 = torch.tensor([[2., 1., 0.]], requires_grad=True)
    类别目标 = torch.tensor([0], dtype=torch.long)
    概率 = torch.softmax(原始分数.detach(), dim=1)
    交叉熵 = F.cross_entropy(原始分数, 类别目标)
    交叉熵.backward()

    print("单标签三分类")
    print("原始分数：", 原始分数.detach())
    print("类别概率：", 概率)
    print("概率的和：", 概率.sum(dim=1))
    print("预测类别编号：", 原始分数.detach().argmax(dim=1))
    print("真实类别编号、类型：", 类别目标, 类别目标.dtype)
    print(f"交叉熵（损失直接接收原始分数）：{交叉熵.item():.4f}")
    print(f"手算 -ln(正确类别概率)：{-概率[0, 0].log().item():.4f}")
    print("原始分数的梯度：", 原始分数.grad)

    二元分数 = torch.tensor([0.0], requires_grad=True)
    二元目标 = torch.tensor([1.0])
    二元损失 = F.binary_cross_entropy_with_logits(二元分数, 二元目标)
    二元损失.backward()
    print("\n二分类")
    print("原始分数：", 二元分数.detach())
    print("sigmoid 转换后的概率：", torch.sigmoid(二元分数.detach()))
    print("真实标签、类型：", 二元目标, 二元目标.dtype)
    print(f"二元交叉熵（内部处理 sigmoid）：{二元损失.item():.4f}")
    print("原始分数的梯度：", 二元分数.grad)
    print("练习：把三分类真实类别改成 2，把二分类真实标签改成 0，观察损失和梯度。")


def main():
    部分表 = {
        "形状": 形状练习,
        "向量": 向量练习,
        "矩阵": 矩阵练习,
        "广播": 广播练习,
        "统计": 统计练习,
        "求导": 求导练习,
        "损失": 损失练习,
        "分类": 分类练习,
    }
    解析器 = argparse.ArgumentParser(description="由浅入深的视觉数学算例，中文输出，CPU 即可。")
    解析器.add_argument("--部分", choices=["全部", *部分表], default="全部", help="选择一部分运行，默认运行全部。")
    设置 = 解析器.parse_args()
    torch.set_printoptions(precision=4, sci_mode=False)
    print("运行设备：CPU；不下载数据；先手算，再看输出。")
    if 设置.部分 == "全部":
        for 练习函数 in 部分表.values():
            练习函数()
    else:
        部分表[设置.部分]()
    print("\n完成。请选择一段，改变一个数字，再解释输出为什么变化。")


if __name__ == "__main__":
    main()
