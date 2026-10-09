"""第八课：只让保留的标签损失参与本次反向传播。"""

import torch
from torch import nn


torch.manual_seed(2026)  # 本例使用固定值；设置种子便于后续扩展。

# 三个得分相互独立，作为可以直接求导的叶张量。
# 得分为零，经 sigmoid 转换后，概率都是 0.5。
logits = torch.zeros(3, requires_grad=True)
targets = torch.tensor([1.0, 0.0, 1.0])  # 三个关系标签是否成立。
mask = torch.tensor([0.0, 1.0, 0.0])  # 1=屏蔽；0=保留。
keep = 1 - mask  # 1=保留；0=不参与本次损失。

# 不提前求平均，保留每个标签对应的一项损失。
criterion = nn.BCEWithLogitsLoss(reduction="none")
individual_losses = criterion(logits, targets)
effective_losses = individual_losses * keep

# 对保留项取平均；若全部被屏蔽，分母至少为一，避免除以零。
kept_count = keep.sum().clamp(min=1)
loss = effective_losses.sum() / kept_count
loss.backward()

print("目标标签：", targets.tolist())
print("屏蔽掩码（1=屏蔽）：", mask.tolist())
print("保留标记（1=保留）：", keep.tolist())
print("全部逐项损失：", [round(value, 4) for value in individual_losses.tolist()])
print("有效逐项损失：", [round(value, 4) for value in effective_losses.tolist()])
print("参与平均的标签数量：", int(keep.sum().item()))
print(f"有效平均损失：{loss.item():.4f}")
print("三个得分的梯度：", logits.grad.tolist())
print("预期梯度：[-0.25, 0.0, -0.25]")
print("第二个标签的梯度为零；第一个和第三个标签仍然提供梯度。")
print("这是掩码思想的小演示，尚未实现论文的掩码选择、课程策略或完整网络。")
