# 第 2 课：读懂 batch 循环和训练函数

前置：[数据结构](01_data_structures_notes.md)。目标是把“逐个处理样本”和“每轮处理多个 batch”写成可读的 Python 程序，而不是先背训练代码。

运行 [完整示例](02_control_flow_and_functions_example.py)，核对输出后修改样本数和批大小。本例使用普通数字，演示流程，不计算梯度，也不更新模型参数。用主计划中的基础时段学习本课，先预测再运行。

## 1. 条件与缩进

```python
loss = 0.3
if loss < 0.5:
    message = "small"
else:
    message = "large"
print(message)  # small
```

冒号后的缩进块属于该分支。用四个空格，避免混用 Tab。`if`、`elif`、`else` 表示依次检查条件；`and`、`or`、`not` 组合布尔条件。

训练脚本常用条件判断是否保存模型、是否进入评估。条件成立只意味着代码走了某条分支，不代表模型真的学会了。

## 2. 循环的五个常见形式

```python
labels = [0, 1, 2]

for label in labels:          # 直接读取元素
    print(label)

for epoch in range(2):        # 依次得到 0、1，不包含 2
    print(epoch)

for index, label in enumerate(labels):
    print(index, label)       # 得到位置和值

features = [[1, 2], [3, 4], [5, 6]]
for feature, label in zip(features, labels):
    print(feature, label)    # 按位置配对

positive_labels = [label for label in labels if label > 0]
print(positive_labels)       # [1, 2]：列表推导式
```

`enumerate(labels, start=1)` 适合打印从 `1` 开始的编号，但不能直接把编号当作从 `0` 开始的列表索引。

**`zip` 默认遇到较短序列的末尾就停止。** `list(zip([0, 1, 2], ["left", "right"]))` 只有两对，第三个元素被跳过。特征和标签应先检查长度相同；否则可能静默漏样本。Python 3.10 及以上还支持 `zip(..., strict=True)`，长度不同会报错。

列表推导适合简单映射或筛选。含多步操作、打印和多层条件时，普通 `for` 更便于排查。

## 3. 函数：输入、处理、返回值

```python
def mean(values):
    if len(values) == 0:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)

def summarize_batch(features, labels, scale=1.0):
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same length")
    scores = [sum(feature) * scale for feature in features]
    return {"count": len(labels), "mean_score": mean(scores)}

summary = summarize_batch([[1, 2], [3, 4]], [0, 1], scale=0.5)
print(summary)  # {'count': 2, 'mean_score': 2.5}
```

`features`、`labels` 是参数；`scale=1.0` 是默认参数；`scale=0.5` 是按名字传入的关键字参数。`return` 把结果交给调用者，并结束当前函数。只调用 `print()` 不会把结果返回；没有执行 `return` 的函数返回 `None`。

`raise ValueError(...)` 主动报告输入不合要求。这里空列表没有平均值，应先说明原因，而不是让除以零错误出现在后面。

## 4. 默认列表参数为什么危险

不要这样写：

```python
def bad_record(value, history=[]):
    history.append(value)
    return history
```

默认列表在定义函数时创建，后续未显式传入 `history` 的调用共享它。第二次调用会保留第一次的记录。

改用 `None`，在函数内部新建列表：

```python
def record(value, history=None):
    if history is None:
        history = []
    history.append(value)
    return history

print(record(0.4))  # [0.4]
print(record(0.2))  # [0.2]

history = []
record(0.4, history)
record(0.2, history)
print(history)      # [0.4, 0.2]：显式共享，用于同一次运行的记录
```

显式传入列表后仍会修改该列表，这是有意记录历史。如果希望保留输入，应先复制，再追加。

## 5. 从样本走到 batch

```python
features = [[1, 2], [3, 4], [5, 6]]
labels = [0, 1, 2]
batch_size = 2

for epoch in range(2):
    for start in range(0, len(features), batch_size):
        batch_features = features[start:start + batch_size]
        batch_labels = labels[start:start + batch_size]
        summary = summarize_batch(batch_features, batch_labels)
        print(epoch, start, summary["count"])
```

每轮先处理前两个样本，再处理最后一个；两个 epoch 都如此。末尾 batch 可能小于 `batch_size`。本课循环没有打乱样本；以后 `DataLoader` 可以负责组 batch、打乱顺序等操作。

这里的 `mean_score` 是数字和的平均值，**不是训练 loss**。进入 PyTorch 课后，要能在每个 batch 中指出“前向 → 损失 → 清梯度 → 反向 → 参数更新”的实际代码。

## 6. 练习

1. 用 `enumerate` 打印 `["img_a", "img_b", "img_c"]` 的从 `1` 开始的展示编号。再说明用列表取数据时应从哪个索引开始。
2. 预测 `list(zip([1, 2, 3], [10, 20]))` 的输出。写出一段在配对前报告长度不一致的代码。
3. 写 `count_positive(values)`，返回大于 `0` 的元素数。用 `[2, -1, 0, 3]` 验证输出为 `2`。
4. 把 `record` 改为返回新列表且不修改显式传入的 `history`。验证原列表保持原样。
5. 取 `5` 个样本、`batch_size = 2`、`3` 个 epoch，手算 batch 个数及每个 batch 的样本数，再改完整示例验证。

<details>
<summary>自测答案与验收提示</summary>

1. `for display_index, name in enumerate(image_names, start=1): print(display_index, name)`；列表首个索引是 `0`，展示编号和数据索引要区分。
2. 输出 `[(1, 10), (2, 20)]`。配对前用 `if len(features) != len(labels): raise ValueError("length mismatch")`。
3. 可写 `return len([value for value in values if value > 0])`，也可在普通循环中累计计数。
4. 函数内部先写 `result = [] if history is None else history.copy()`，然后 `result.append(value)` 并 `return result`。
5. 每个 epoch 有 `3` 个 batch，样本数依次 `2、2、1`；三个 epoch 共 `9` 个 batch。能独立写循环、解释输入输出、指出 `zip` 和默认列表陷阱，才算通过。

</details>

下一课：[类与模块](03_classes_and_modules_notes.md)。官方参考：[Python 控制流与函数](https://docs.python.org/3/tutorial/controlflow.html)、[列表推导和遍历](https://docs.python.org/3/tutorial/datastructures.html)。
