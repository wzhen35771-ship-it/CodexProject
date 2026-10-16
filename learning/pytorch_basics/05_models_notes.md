# 第 5 课：神经网络与损失

## 本课概念

`nn.Module` 管理模型层和参数；在 `__init__` 创建层，在 `forward` 编写计算。
本课模型：[B, 2] → Linear(2, 8) → ReLU → Linear(8, 3) → [B, 3]。
Linear(2, 8) 权重形状为 [8, 2]、偏置 [8]。ReLU 把负值变为 0，引入非线性。
logits 是未归一化得分。`CrossEntropyLoss` 接收 [B, 3] logits 与 [B] long 类别编号，不需要提前 softmax。
`argmax(dim=1)` 为每条样本选择得分最大的类别。本课只演示互斥的三个类别。

## 怎么学

1. 先读概念，用纸笔预测示例输出。
2. 运行 [05_models_example.py](05_models_example.py)，逐行对照预测。
3. 关闭示例，完成 [05_models_exercise.py](05_models_exercise.py)。
4. 写下形状、实际输出、错误原因；能解释后进入下一课。

## 独立练习与验收

看懂前向与损失后，运行补充示例 [05_parameter_update_example.py](05_parameter_update_example.py)。它用一条人工样本完成一次“清零 → 前向 → 损失 → 求导 → 更新”，并重新前向核对损失。本机 PyTorch 2.6.0+cpu 运行时，损失由约 1.5134 降至 1.3435；这只是一次教学更新。

`backward()` 计算梯度，`optimizer.step()` 修改模型的权重和偏置。原来的 logits 和 loss 不会自动变成更新后的结果，需要再次调用模型并计算损失。

1. 解释 B 的含义，并手算模型总参数数。
2. 将隐藏层改为 4，预测参数数量后验证。
3. 创建全零 [1, 3] logits，标签为 0，计算损失并解释为何约为 1.099。

不以运行成功代替理解：至少独立完成全部任务，并解释每个关键操作。
