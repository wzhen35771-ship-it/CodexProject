# 第 6 课：Dataset、DataLoader 与批次

## 本课概念

Dataset 提供单条样本，DataLoader 将样本组织成 batch。
`TensorDataset(features, labels)` 让相同索引的特征和标签成对返回。
一个 epoch 遍历一次整个数据集。10 条样本、batch_size=4、drop_last=False 时得到 4、4、2 三批。
`shuffle=True` 打乱顺序而保持特征标签对应；评估通常不需要打乱。
本课的关系特征为主语中心减宾语中心：[dx, dy]。标签 0=左侧、1=右侧、2=上方。图像 y 轴向下增长。
这些是人为构造的互斥数据，不代表真实场景图中所有关系。

## 怎么学

1. 先读概念，用纸笔预测示例输出。
2. 运行 [06_data_example.py](06_data_example.py)，逐行对照预测。
3. 关闭示例，完成 [06_data_exercise.py](06_data_exercise.py)。
4. 写下形状、实际输出、错误原因；能解释后进入下一课。

## 独立练习与验收

1. 把 batch_size 改为 4、6、5，分别预测批次数与最后一批大小。
2. 设置 drop_last=True，解释丢弃了哪些数据。
3. 按已知中心坐标手算一条 [dx, dy]，解释主语宾语顺序。

不以运行成功代替理解：至少独立完成全部任务，并解释每个关键操作。
