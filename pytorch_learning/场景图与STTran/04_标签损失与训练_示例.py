"""人工关系特征的训练练习：三组输出、标签、损失和梯度。

这不是完整 STTran：没有图片、检测器、空间或时间 Transformer。
三组关系的输出数量与官方 Action Genome 基线相同：3、6、17。
默认演示 sigmoid + MultiLabelMarginLoss；--loss bce 演示 sigmoid + BCELoss。
运行示例：python 04_标签损失与训练_示例.py --loss margin
"""

import argparse
from pathlib import Path

import torch
from torch import nn


def 构造间隔标签(正类列表, 类别数):
    """正类编号放在前面，剩余位置填 -1；不是 0/1 标签。"""
    标签 = torch.full((len(正类列表), 类别数), -1, dtype=torch.long)
    for 行号, 正类 in enumerate(正类列表):
        if len(set(正类)) != len(正类):
            raise ValueError("正类编号不能重复")
        if any(编号 < 0 or 编号 >= 类别数 for 编号 in 正类):
            raise ValueError("正类编号超出类别范围")
        标签[行号, :len(正类)] = torch.tensor(正类, dtype=torch.long)
    return 标签


def 构造二元标签(正类列表, 类别数):
    """每列表示该类别是否属于正类：属于填 1，其余填 0。"""
    标签 = torch.zeros((len(正类列表), 类别数), dtype=torch.float32)
    for 行号, 正类 in enumerate(正类列表):
        标签[行号, 正类] = 1.0
    return 标签


def 验证手算():
    print("一、核对手算例子")
    分数 = torch.tensor([[0.8, 0.2, 0.1]], requires_grad=True)
    标签 = torch.tensor([[0, -1, -1]], dtype=torch.long)
    间隔损失 = nn.MultiLabelMarginLoss()(分数, 标签)
    torch.testing.assert_close(间隔损失, torch.tensor((0.4 + 0.3) / 3))
    间隔损失.backward()
    torch.testing.assert_close(分数.grad, torch.tensor([[-2/3, 1/3, 1/3]]))
    print(f"单正类间隔损失：{间隔损失.item():.6f}，应为 0.233333")
    print("对输入分数的梯度：", 分数.grad.tolist())

    多正类损失 = nn.MultiLabelMarginLoss()(
        分数.detach(), torch.tensor([[0, 2, -1]], dtype=torch.long)
    )
    torch.testing.assert_close(多正类损失, torch.tensor((0.4 + 1.1) / 3))
    print(f"多正类间隔损失：{多正类损失.item():.6f}，应为 0.500000")

    原始分数 = torch.tensor([[2.0, 1.0, 0.0]], requires_grad=True)
    交叉熵 = nn.CrossEntropyLoss()(原始分数, torch.tensor([0]))
    概率 = 原始分数.detach().softmax(dim=1)
    交叉熵.backward()
    预期梯度 = 概率 - torch.tensor([[1.0, 0.0, 0.0]])
    torch.testing.assert_close(原始分数.grad, 预期梯度)
    print(f"交叉熵：{交叉熵.item():.6f}，应约为 0.407606")
    print("概率：", 概率.tolist())
    print("交叉熵对原始分数的梯度：", 原始分数.grad.tolist())

    二元原始分数 = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
    二元标签 = torch.tensor([[1.0, 0.0, 1.0]])
    二元损失 = nn.BCEWithLogitsLoss()(二元原始分数, 二元标签)
    概率版本损失 = nn.BCELoss()(二元原始分数.sigmoid(), 二元标签)
    torch.testing.assert_close(二元损失, 概率版本损失)
    二元损失.backward()
    torch.testing.assert_close(二元原始分数.grad, torch.tensor([[-1/6, 1/6, -1/6]]))
    print(f"二元交叉熵：{二元损失.item():.6f}，应约为 0.693147")
    print("按三个元素取平均后的梯度：", 二元原始分数.grad.tolist())


class 小关系分类器(nn.Module):
    def __init__(self):
        super().__init__()
        self.共享层 = nn.Sequential(nn.Linear(4, 8), nn.Tanh())
        self.注意关系头 = nn.Linear(8, 3)
        self.空间关系头 = nn.Linear(8, 6)
        self.接触关系头 = nn.Linear(8, 17)

    def forward(self, 关系特征):
        共享特征 = self.共享层(关系特征)
        注意原始分数 = self.注意关系头(共享特征)
        空间原始分数 = self.空间关系头(共享特征)
        接触原始分数 = self.接触关系头(共享特征)
        return {
            "attention_distribution": 注意原始分数,
            "spatial_distribution": 空间原始分数.sigmoid(),
            "contacting_distribution": 接触原始分数.sigmoid(),
        }


def 训练小模型(损失选择):
    print("\n二、训练三组关系分类器；这里用人工特征，所有计算都在 CPU 上")
    torch.manual_seed(7)
    # 每行对应一个已经选定的物体对；数字是教学构造的特征。
    关系特征 = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
    ])
    注意标签 = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    空间正类 = [[0, 2], [1], [2, 4], [0], [3, 5], [1, 2]]
    接触正类 = [[0, 3], [1], [2, 4], [0, 5], [6], [1, 7]]
    if 损失选择 == "margin":
        空间标签 = 构造间隔标签(空间正类, 6)
        接触标签 = 构造间隔标签(接触正类, 17)
        多标签损失函数 = nn.MultiLabelMarginLoss()
    else:
        空间标签 = 构造二元标签(空间正类, 6)
        接触标签 = 构造二元标签(接触正类, 17)
        多标签损失函数 = nn.BCELoss()

    模型 = 小关系分类器()
    交叉熵函数 = nn.CrossEntropyLoss()
    # 教学数据小，用较大的学习率方便观察；这不是官方训练配置。
    优化器 = torch.optim.Adam(模型.parameters(), lr=0.03)

    def 计算损失(输出):
        return {
            "注意关系损失": 交叉熵函数(输出["attention_distribution"], 注意标签),
            "空间关系损失": 多标签损失函数(输出["spatial_distribution"], 空间标签),
            "接触关系损失": 多标签损失函数(输出["contacting_distribution"], 接触标签),
        }

    初始输出 = 模型(关系特征)
    for 名字, 张量 in 初始输出.items():
        print(f"{名字} 的形状：{list(张量.shape)}")
    print("注意标签形状与类型：", list(注意标签.shape), 注意标签.dtype)
    print("第一对物体的空间标签：", 空间标签[0].tolist())
    print("第一对物体的接触标签：", 接触标签[0].tolist())
    初始总损失 = sum(计算损失(初始输出).values()).item()
    初始权重 = 模型.空间关系头.weight.detach().clone()

    模型.train()
    for 轮次 in range(1, 101):
        优化器.zero_grad()
        输出 = 模型(关系特征)
        各项损失 = 计算损失(输出)
        总损失 = sum(各项损失.values())
        总损失.backward()
        裁剪前范数 = torch.nn.utils.clip_grad_norm_(模型.parameters(), max_norm=5.0)
        if 轮次 == 1:
            print("共享层权重梯度的形状：", list(模型.共享层[0].weight.grad.shape))
            print("共享层权重梯度范数：", 模型.共享层[0].weight.grad.norm().item())
            print("空间头权重梯度范数：", 模型.空间关系头.weight.grad.norm().item())
            print("全体梯度裁剪前的范数：", 裁剪前范数.item())
        优化器.step()
        if 轮次 in (1, 20, 50, 100):
            文本 = "，".join(f"{名字}={值.item():.4f}" for 名字, 值 in 各项损失.items())
            print(f"第 {轮次:3d} 次更新前：总损失={总损失.item():.4f}，{文本}")

    模型.eval()
    with torch.no_grad():
        最终输出 = 模型(关系特征)
        最终总损失 = sum(计算损失(最终输出).values()).item()
        print("注意关系预测类别：", 最终输出["attention_distribution"].argmax(dim=1).tolist())
    assert 最终总损失 < 初始总损失, "本例的固定训练结果应该让损失下降"
    assert not torch.equal(初始权重, 模型.空间关系头.weight.detach())
    print(f"初始损失 {初始总损失:.6f} → 最终损失 {最终总损失:.6f}")
    print("空间头参数已发生更新：", not torch.equal(初始权重, 模型.空间关系头.weight.detach()))

    # 输出仅保存到学习文件夹内，便于重跑和检查。
    保存目录 = Path(__file__).resolve().parent / "outputs"
    保存目录.mkdir(exist_ok=True)
    保存路径 = 保存目录 / f"04_标签损失与训练_权重_{损失选择}.pt"
    torch.save({"state_dict": 模型.state_dict()}, 保存路径)
    新模型 = 小关系分类器()
    检查点 = torch.load(保存路径, map_location="cpu", weights_only=True)
    新模型.load_state_dict(检查点["state_dict"])
    新模型.eval()
    with torch.no_grad():
        加载输出 = 新模型(关系特征)
    for 名字 in 最终输出:
        torch.testing.assert_close(加载输出[名字], 最终输出[名字])
    print("权重已保存，加载后输出一致：", 保存路径)


def 验证损失掩码():
    print("\n三、一个独立二元损失被屏蔽后，其直接梯度为零")
    原始分数 = torch.zeros((1, 3), requires_grad=True)
    标签 = torch.tensor([[1.0, 0.0, 1.0]])
    掩码 = torch.tensor([[0.0, 1.0, 0.0]])  # 本例中 1 表示屏蔽。
    保留 = 1.0 - 掩码
    逐项损失 = nn.BCEWithLogitsLoss(reduction="none")(原始分数, 标签)
    有效损失 = (逐项损失 * 保留).sum() / 保留.sum().clamp_min(1.0)
    有效损失.backward()
    torch.testing.assert_close(原始分数.grad, torch.tensor([[-0.25, 0.0, -0.25]]))
    print("逐项损失：", 逐项损失.tolist())
    print("掩码：", 掩码.tolist())
    print("原始分数梯度：", 原始分数.grad.tolist())
    print("这是独立二元损失的教学示例；CE 和间隔损失内的类别相互耦合。")


if __name__ == "__main__":
    参数解析器 = argparse.ArgumentParser(description="三组关系标签与训练的中文练习")
    参数解析器.add_argument("--loss", choices=["margin", "bce"], default="margin")
    参数 = 参数解析器.parse_args()
    torch.set_num_threads(1)
    验证手算()
    训练小模型(参数.loss)
    验证损失掩码()
