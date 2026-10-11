"""第四步第一天：认识物体实例、关系三元组和物体对索引。

数据与四维特征均为人工给定；这是数据表示练习。
"""

import torch


物体 = [
    {"编号": 0, "类别": "人", "框": [10, 5, 50, 95]},
    {"编号": 1, "类别": "杯子", "框": [30, 40, 45, 60]},
    {"编号": 2, "类别": "杯子", "框": [70, 60, 85, 80]},
]
关系 = [
    (0, "看着", 1),
    (0, "拿着", 1),
    (0, "未接触", 2),
]

print("一、节点：具体的物体实例")
for 节点 in 物体:
    print(f"编号 {节点['编号']}：{节点['类别']}，框 {节点['框']}")

print("\n二、人工给定的关系标注")
for 主体, 关系名称, 客体 in 关系:
    print(f"{物体[主体]['类别']}{主体} —{关系名称}→ {物体[客体]['类别']}{客体}")

# 同一对物体可以对应多条关系，取不同物体对时去掉重复项。
物体对 = sorted({(主体, 客体) for 主体, _, 客体 in 关系})
pair_idx = torch.tensor(物体对, dtype=torch.long)
print("\n三、物体对索引")
print("节点数量：", len(物体))
print("关系数量：", len(关系))
print("不同物体对数量：", len(物体对))
print("pair_idx：\n", pair_idx)
print("pair_idx 形状：", list(pair_idx.shape))

# 教学图片只有一帧；框的第一列用于说明属于第 0 帧。
boxes = torch.tensor([[0, *节点["框"]] for 节点 in 物体], dtype=torch.float32)
im_idx = torch.zeros(len(物体对), dtype=torch.long)

# 四维数值用于手算索引与拼接；特征维度没有预先指定的语义。
features = torch.tensor([
    [1.0, 0.0, 0.5, 0.2],
    [0.0, 1.0, 0.3, 0.4],
    [0.2, 0.8, 0.6, 0.1],
])
主体特征 = features[pair_idx[:, 0]]
客体特征 = features[pair_idx[:, 1]]
教学物体对特征 = torch.cat([主体特征, 客体特征], dim=1)

print("\n四、根据索引取出两端特征，再拼接")
print("boxes 形状：", list(boxes.shape), "；每行是帧编号与四个框坐标")
print("im_idx：", im_idx.tolist(), "；每个物体对所属帧")
print("features 形状：", list(features.shape))
print("主体特征：\n", 主体特征)
print("客体特征：\n", 客体特征)
print("拼接结果：\n", 教学物体对特征)
print("教学物体对特征形状：", list(教学物体对特征.shape))
print("这里拼接两份四维特征；官方 STTran 还组合联合区域、空间布局和类别信息。")

print("\n练习：给关系列表增加 (0, '看着', 2)，先预测三种数量怎样变化，再运行。")

