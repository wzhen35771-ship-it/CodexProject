"""第六课：把四组数据分成小批次，逐批读取。"""

import torch
from torch.utils.data import DataLoader, TensorDataset


torch.manual_seed(0)  # 后面练习打乱顺序时，便于重复比较

# 输入和目标的行号一一对应，每行是一条完整样本
x = torch.tensor([[-1.0], [0.0], [1.0], [2.0]])
target = torch.tensor([[-2.0], [1.0], [4.0], [7.0]])

dataset = TensorDataset(x, target)  # 将每条输入与它的目标配成一组
loader = DataLoader(dataset, batch_size=2, shuffle=False)

print("全部输入的形状：", x.shape)
print("全部目标的形状：", target.shape)
print("样本总数：", len(dataset))
print("一轮的批次数：", len(loader))

for batch_number, (batch_x, batch_target) in enumerate(loader, start=1):
    print(f"第 {batch_number} 批：")
    print("  输入数值：", batch_x.tolist())
    print("  目标数值：", batch_target.tolist())
    print("  输入形状：", batch_x.shape)
    print("  目标形状：", batch_target.shape)

print("一轮读取完成：四条样本已经全部读过一次。")
