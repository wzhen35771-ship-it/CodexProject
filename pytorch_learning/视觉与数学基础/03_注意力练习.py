"""第三步的注意力练习：用小矩阵理解匹配、权重和信息融合。

全部在处理器上运行；不下载数据、图片或模型。
这些向量是教学数字，不代表真实物体的语义特征。
"""

import math

import torch


def 计算注意力(查询, 键, 值):
    """计算最基础的单头注意力，不含额外掩码和随机丢弃。"""
    # 查询形状：[查询数量, 特征维度]
    # 键形状：  [候选数量, 特征维度]
    # 值形状：  [候选数量, 值维度]
    原始得分 = 查询 @ 键.T
    特征维度 = 查询.shape[-1]
    缩放得分 = 原始得分 / math.sqrt(特征维度)
    权重 = torch.softmax(缩放得分, dim=-1)  # 每一行分别归一化。
    输出 = 权重 @ 值
    return 原始得分, 缩放得分, 权重, 输出


def 展示计算(标题, 查询, 键, 值):
    print("\n" + 标题)
    print("查询：", 查询)
    print("键：", 键)
    print("值：", 值)
    print("查询、键、值的形状：", list(查询.shape), list(键.shape), list(值.shape))

    原始得分, 缩放得分, 权重, 输出 = 计算注意力(查询, 键, 值)
    print("第一步，查询与键做点积：", 原始得分)
    print("第二步，除以特征维度的平方根：", 缩放得分)
    print("第三步，转换成注意力权重：", 权重)
    print("各行权重之和：", 权重.sum(dim=-1))
    print("第四步，用权重融合值：", 输出)
    print("输出形状：", list(输出.shape))
    return 权重, 输出


def 主程序():
    torch.set_printoptions(precision=4, sci_mode=False)

    # float32 是浮点类型；device="cpu" 明确在处理器上计算。
    查询 = torch.tensor([[1., 0.]], dtype=torch.float32, device="cpu")
    键 = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float32, device="cpu")
    值 = torch.tensor([[10., 0.], [0., 20.]], dtype=torch.float32, device="cpu")

    原始权重, 原始输出 = 展示计算("一、一个查询，两份信息", 查询, 键, 值)
    print("手算核对：权重约 [0.6698, 0.3302]；输出约 [6.6976, 6.6048]。")

    第二个查询 = torch.tensor([[0., 1.]], dtype=torch.float32)
    展示计算("二、改变查询，匹配方向也会变化", 第二个查询, 键, 值)

    相同匹配查询 = torch.tensor([[1., 1.]], dtype=torch.float32)
    展示计算("三、两个匹配得分相同，权重各为一半", 相同匹配查询, 键, 值)

    更改后的值 = torch.tensor([[10., 0.], [0., 100.]], dtype=torch.float32)
    新权重, 新输出 = 展示计算("四、只改变值，观察权重与输出", 查询, 键, 更改后的值)
    print("权重是否保持相同：", "是" if torch.allclose(原始权重, 新权重) else "否")
    print("原来输出：", 原始输出)
    print("改变值后的输出：", 新输出)
    print("原因：匹配使用查询和键；融合内容来自值。")

    两个查询 = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float32)
    展示计算("五、两个查询同时计算，每行得到自己的结果", 两个查询, 键, 值)

    print("\n六、自注意力的输入来源相同，学习投影可以不同")
    输入特征 = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float32)
    # 以下投影是教学固定数字；真实模型中它们是通过训练学习的参数。
    查询投影 = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float32)
    键投影 = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.float32)
    值投影 = torch.tensor([[10., 0.], [0., 20.]], dtype=torch.float32)
    投影后查询 = 输入特征 @ 查询投影
    投影后键 = 输入特征 @ 键投影
    投影后值 = 输入特征 @ 值投影
    展示计算("同一组输入经过三种投影", 投影后查询, 投影后键, 投影后值)
    print("来源相同，并不保证查询、键、值的数值相同。")

    print("\n理解检查")
    print("1. 如果查询改成 [0, 0]，两个权重和输出分别是多少？")
    print("2. 为什么原始得分为 0 的候选仍有非零权重？")
    print("3. 值有三个分量时，输出会有几个分量？")
    print("4. 注意力输出是融合特征，还需要分类层才能得到关系预测。")
    print("5. 本文件没有训练模型；也没有实现 STTran 或 ImparTail。")


if __name__ == "__main__":
    主程序()
