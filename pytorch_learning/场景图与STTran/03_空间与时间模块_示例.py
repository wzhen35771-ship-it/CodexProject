"""空间与时间注意力练习：人工特征、小维度、CPU，帮助跟踪 STTran 数据流。

本程序没有物体检测器、图像特征提取器、关系标签或完整训练过程。
这里的 5 行特征分别代表三个视频帧中的 2、1、2 个物体对。
"""

import math

import torch
from torch import nn


class 空间层(nn.Module):
    """按照官方空间层的顺序：两次残差相加，两次 LayerNorm。"""

    def __init__(self, 特征维数=8, 头数=2, 中间维数=12):
        super().__init__()
        self.注意力 = nn.MultiheadAttention(特征维数, 头数, dropout=0.0)
        self.线性1 = nn.Linear(特征维数, 中间维数)
        self.线性2 = nn.Linear(中间维数, 特征维数)
        self.归一化1 = nn.LayerNorm(特征维数)
        self.归一化2 = nn.LayerNorm(特征维数)

    def forward(self, 输入, 补齐掩码):
        新信息, 权重 = self.注意力(
            输入, 输入, 输入,
            key_padding_mask=补齐掩码,
            need_weights=True,
        )
        中间结果 = self.归一化1(输入 + 新信息)
        前馈结果 = self.线性2(torch.relu(self.线性1(中间结果)))
        输出 = self.归一化2(中间结果 + 前馈结果)
        return 输出, 权重


class 时间层(nn.Module):
    """对应官方时间层的顺序：第二次残差相加后没有第二个 LayerNorm。"""

    def __init__(self, 特征维数=8, 头数=2, 中间维数=12):
        super().__init__()
        self.注意力 = nn.MultiheadAttention(特征维数, 头数, dropout=0.0)
        self.线性1 = nn.Linear(特征维数, 中间维数)
        self.线性2 = nn.Linear(中间维数, 特征维数)
        self.归一化 = nn.LayerNorm(特征维数)

    def forward(self, 输入, 补齐掩码, 位置向量):
        新信息, 权重 = self.注意力(
            query=输入 + 位置向量,
            key=输入 + 位置向量,
            value=输入,
            key_padding_mask=补齐掩码,
            need_weights=True,
        )
        中间结果 = self.归一化(输入 + 新信息)
        前馈结果 = self.线性2(torch.relu(self.线性1(中间结果)))
        输出 = 中间结果 + 前馈结果
        return 输出, 权重


def 打包同帧关系(特征, 帧编号):
    """从 [R,D] 打包成 [L,T,D]，每一列是一帧。"""
    帧数 = int(帧编号.max().item()) + 1
    每帧数量 = torch.bincount(帧编号, minlength=帧数)
    最大数量 = int(每帧数量.max().item())
    空间输入 = 特征.new_zeros((最大数量, 帧数, 特征.shape[1]))
    补齐掩码 = torch.ones((帧数, 最大数量), dtype=torch.bool)
    原始行号 = torch.full((最大数量, 帧数), -1, dtype=torch.long)

    for 帧 in range(帧数):
        行号 = torch.where(帧编号 == 帧)[0]
        数量 = 行号.numel()
        空间输入[:数量, 帧] = 特征[行号]
        补齐掩码[帧, :数量] = False
        原始行号[:数量, 帧] = 行号
    return 空间输入, 补齐掩码, 原始行号


def 还原同帧关系(空间输出, 原始行号, 总关系数):
    """按行号放回 [R,D]，无需依赖展平以后恰好是正确的顺序。"""
    还原结果 = 空间输出.new_zeros((总关系数, 空间输出.shape[-1]))
    for 帧 in range(空间输出.shape[1]):
        有效 = 原始行号[:, 帧] >= 0
        还原结果[原始行号[有效, 帧]] = 空间输出[有效, 帧]
    return 还原结果


def 打包相邻窗口(特征, 帧编号, 位置表):
    """窗口大小 2、步长 1；窗口内首先放较早帧，再放较晚帧。"""
    帧数 = int(帧编号.max().item()) + 1
    assert 帧数 >= 2, "本练习要求至少两帧。"
    最大数量 = int(torch.bincount(帧编号).max().item())
    容量 = 2 * 最大数量
    窗口数 = 帧数 - 1
    时间输入 = 特征.new_zeros((容量, 窗口数, 特征.shape[-1]))
    位置向量 = torch.zeros_like(时间输入)
    补齐掩码 = torch.ones((窗口数, 容量), dtype=torch.bool)
    原始行号 = torch.full((容量, 窗口数), -1, dtype=torch.long)
    绝对帧编号 = torch.full((容量, 窗口数), -1, dtype=torch.long)

    for 窗口 in range(窗口数):
        较早行号 = torch.where(帧编号 == 窗口)[0]
        较晚行号 = torch.where(帧编号 == 窗口 + 1)[0]
        行号 = torch.cat((较早行号, 较晚行号))
        数量 = 行号.numel()
        较早数量 = 较早行号.numel()
        时间输入[:数量, 窗口] = 特征[行号]
        位置向量[:较早数量, 窗口] = 位置表[0]
        位置向量[较早数量:数量, 窗口] = 位置表[1]
        补齐掩码[窗口, :数量] = False
        原始行号[:数量, 窗口] = 行号
        绝对帧编号[:数量, 窗口] = 帧编号[行号]
    return 时间输入, 位置向量, 补齐掩码, 原始行号, 绝对帧编号


def 还原较晚帧模式(时间输出, 原始行号, 绝对帧编号, 总关系数):
    """latter：第 0 帧由第一个窗口保留，其余帧取前一个窗口的结果。"""
    还原结果 = 时间输出.new_zeros((总关系数, 时间输出.shape[-1]))
    被写次数 = torch.zeros(总关系数, dtype=torch.long)
    for 窗口 in range(时间输出.shape[1]):
        保留位置 = 绝对帧编号[:, 窗口] == 窗口 + 1
        if 窗口 == 0:
            保留位置 = 保留位置 | (绝对帧编号[:, 窗口] == 0)
        行号 = 原始行号[保留位置, 窗口]
        还原结果[行号] = 时间输出[保留位置, 窗口]
        被写次数[行号] += 1
    assert torch.equal(被写次数, torch.ones_like(被写次数))
    return 还原结果


def 注意力手算核对():
    查询 = torch.tensor([[1.0, 0.0]])
    键 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    值 = torch.tensor([[10.0, 0.0], [0.0, 20.0]])
    分数 = 查询 @ 键.T / math.sqrt(2)
    权重 = torch.softmax(分数, dim=-1)
    加权结果 = 权重 @ 值
    print("手算分数：", 分数)
    print("手算权重：", 权重)
    print("手算加权结果：", 加权结果)
    assert torch.allclose(权重.sum(dim=-1), torch.ones(1))
    assert torch.allclose(加权结果, torch.tensor([[6.6976, 6.6048]]), atol=1e-4)


def 归一化手算核对():
    输入 = torch.tensor([[1.0, 3.0]])
    均值 = 输入.mean(dim=-1, keepdim=True)
    方差 = 输入.var(dim=-1, correction=0, keepdim=True)
    手算 = (输入 - 均值) / torch.sqrt(方差 + 1e-5)
    模块 = nn.LayerNorm(2)
    print("归一化均值：", 均值, "总体方差：", 方差)
    print("归一化结果：", 模块(输入))
    assert torch.allclose(手算, 模块(输入))


def main():
    torch.manual_seed(7)
    torch.set_printoptions(precision=4, sci_mode=False)
    注意力手算核对()
    归一化手算核对()

    # 行号 0、1 属于第 0 帧；行号 2 属于第 1 帧；行号 3、4 属于第 2 帧。
    # 数字是人工特征，用来追踪维度；每一列没有指定为“颜色”或“拿着”。
    特征 = (torch.arange(40, dtype=torch.float32).reshape(5, 8) / 10).requires_grad_()
    帧编号 = torch.tensor([0, 0, 1, 2, 2])
    空间输入, 空间掩码, 空间行号 = 打包同帧关系(特征, 帧编号)
    print("\n原始关系特征形状：", tuple(特征.shape))
    print("空间输入形状 [L,T,D]：", tuple(空间输入.shape))
    print("空间行号，每一列是一帧：\n", 空间行号)
    print("空间掩码，每一行是一帧，True 表示补齐：\n", 空间掩码)
    assert torch.equal(空间行号, torch.tensor([[0, 2, 3], [1, -1, 4]]))
    assert torch.equal(还原同帧关系(空间输入, 空间行号, 5), 特征)

    空间模型 = 空间层()
    空间输出, 空间权重 = 空间模型(空间输入, 空间掩码)
    同帧结果 = 还原同帧关系(空间输出, 空间行号, 5)
    print("空间权重形状 [T,L,L]：", tuple(空间权重.shape))
    print("第 1 帧空间注意力权重：\n", 空间权重[1])
    assert tuple(同帧结果.shape) == (5, 8)
    # 第 1 帧只有一个有效 key，第二列权重必须是 0。
    assert torch.equal(空间权重[1, :, 1], torch.zeros(2))

    位置嵌入 = nn.Embedding(2, 8)
    nn.init.uniform_(位置嵌入.weight)
    时间输入, 位置向量, 时间掩码, 时间行号, 时间帧号 = 打包相邻窗口(
        同帧结果, 帧编号, 位置嵌入.weight,
    )
    print("\n时间输入形状 [2L,T-1,D]：", tuple(时间输入.shape))
    print("时间行号，每一列是一个相邻帧窗口：\n", 时间行号)
    print("时间帧号：\n", 时间帧号)
    print("时间掩码，每一行是一个窗口：\n", 时间掩码)
    print("较早位置向量：", 位置嵌入.weight[0])
    print("较晚位置向量：", 位置嵌入.weight[1])
    assert torch.equal(时间行号, torch.tensor([[0, 2], [1, 3], [2, 4], [-1, -1]]))
    assert torch.equal(时间帧号, torch.tensor([[0, 1], [0, 2], [1, 2], [-1, -1]]))
    assert torch.equal(时间掩码, torch.tensor([[False, False, False, True]] * 2))
    assert torch.equal(位置向量[2, 0], 位置嵌入.weight[1])  # 关系行 2 在第一个窗口较晚。
    assert torch.equal(位置向量[0, 1], 位置嵌入.weight[0])  # 同一关系行 2 在第二个窗口较早。
    # 用时间输入本身核对打包和还原，不受模型随机参数影响。
    assert torch.equal(还原较晚帧模式(时间输入, 时间行号, 时间帧号, 5), 同帧结果)

    时间模型 = 时间层()
    时间输出, 时间权重 = 时间模型(时间输入, 时间掩码, 位置向量)
    最终结果 = 还原较晚帧模式(时间输出, 时间行号, 时间帧号, 5)
    print("时间权重形状 [T-1,2L,2L]：", tuple(时间权重.shape))
    print("最终结果形状 [R,D]：", tuple(最终结果.shape))
    print("最终结果：\n", 最终结果)
    assert torch.isfinite(最终结果).all()
    assert torch.equal(时间权重[:, :, -1], torch.zeros(2, 4))

    # 只检查计算图能连回人工输入、注意力参数、可学习位置向量。
    # 平方平均是本练习的检查目标，不是真实 STTran 的关系分类损失。
    检查目标 = 最终结果.square().mean()
    检查目标.backward()
    assert 特征.grad is not None and torch.isfinite(特征.grad).all()
    assert 空间模型.注意力.in_proj_weight.grad is not None
    assert 时间模型.注意力.in_proj_weight.grad is not None
    assert 位置嵌入.weight.grad is not None
    print("\n检查目标：", 检查目标.item())
    print("输入梯度形状：", tuple(特征.grad.shape))
    print("位置表梯度形状：", tuple(位置嵌入.weight.grad.shape))
    print("所有形状、有效行号、补齐掩码、还原顺序和反向传播检查通过。")


if __name__ == "__main__":
    main()
