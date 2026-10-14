# Python 常用函数与语法速查

配合本目录的 01～05 示例使用。先掌握 `len`、`range`、`enumerate`、`zip` 和 `sum`，其余写法遇到时再查。返回 [学习指南](00_study_guide.md)。

## 1. 查看和计算数据

| 函数 | 作用 | 示例及结果 |
|---|---|---|
| `print()` | 打印内容 | `print("你好")` 显示“你好” |
| `len()` | 获取元素数量 | `len([3, 5, 7])` → `3` |
| `type()` | 查看数据类型 | `type(3)` → `<class 'int'>` |
| `sum()` | 求和 | `sum([1, 2, 3])` → `6` |
| `min()` | 取最小值 | `min(8, 6)` → `6` |
| `max()` | 取最大值 | `max([1, 5, 2])` → `5` |
| `abs()` | 取绝对值 | `abs(-0.8)` → `0.8` |

计算平均值：

```python
values = [3, 5, 7]
average = sum(values) / len(values)
print(average)  # 5.0
```

列表为空时不能这样求平均，因为会除以零。

## 2. 循环和配对

| 函数 | 作用 | 示例 |
|---|---|---|
| `range()` | 生成整数序列，常用于循环 | `range(3)` 依次产生 `0、1、2` |
| `enumerate()` | 同时获取编号和元素 | `enumerate(["a", "b"], start=1)` 产生 `(1, "a")、(2, "b")` |
| `zip()` | 将相同位置的元素配对 | `zip([1, 2], ["a", "b"])` 产生 `(1, "a")、(2, "b")` |

### range：控制循环次数或索引

完整写法是 `range(起点, 终点, 步长)`，包含起点，不包含终点。只传一个参数时表示终点，起点默认为 `0`；步长默认为 `1`。

```python
list(range(3))        # [0, 1, 2]
list(range(1, 4))     # [1, 2, 3]
list(range(0, 6, 2))  # [0, 2, 4]

for epoch in range(1, 3):
    print(epoch)  # 依次打印 1、2
```

### enumerate：遍历时带上编号

```python
batches = ["第一批数据", "第二批数据"]
for batch_index, batch in enumerate(batches, start=1):
    print(batch_index, batch)
# 1 第一批数据
# 2 第二批数据
```

`start=1` 只影响编号，不会跳过第一个元素。不写它时，默认从 `0` 编号。

在第 05 课中，每个批次又由“特征列表、标签列表”组成，所以还可以进一步解包：

```python
for batch_index, (batch_features, batch_labels) in enumerate(batches, start=1):
    # batch_index：批次编号
    # batch_features：这一批的特征
    # batch_labels：这一批的标签
    pass
```

这段写法需要 `batches` 中的元素是 `(batch_features, batch_labels)`，可对照 [05 示例](05_training_script_reading_example.py)。

### zip：把输入与答案一一配对

```python
predictions = [0, 1, 2]
labels = [0, 0, 2]
print(list(zip(predictions, labels)))
# [(0, 0), (1, 0), (2, 2)]

for prediction, label in zip(predictions, labels):
    print(prediction == label)
# True
# False
# True
```

默认情况下，`zip()` 按较短的序列停止：

```python
list(zip([1, 2, 3], ["a", "b"]))
# [(1, "a"), (2, "b")]
```

`enumerate()` 和 `zip()` 返回迭代器，可以放进 `for` 循环，也可以用 `list(...)` 展开查看；一个迭代器遍历完后，需要重新创建才能再遍历。

结合 `sum()` 统计正确预测数量：

```python
correct = sum(
    prediction == label
    for prediction, label in zip(predictions, labels)
)
print(correct)  # 2
```

求和时，`True` 按 `1` 计算，`False` 按 `0` 计算。计算准确率前，应确认预测和标签等长且非空，再用 `correct / len(labels)`。

## 3. 转换数据类型

| 写法 | 作用 | 示例及结果 |
|---|---|---|
| `int()` | 转为整数 | `int("4")` → `4` |
| `float()` | 转为浮点数 | `float("0.5")` → `0.5` |
| `str()` | 转为字符串 | `str(4)` → `"4"` |
| `bool()` | 转为布尔值 | `bool([])` → `False` |
| `list()` | 转为列表 | `list((1, 2))` → `[1, 2]` |
| `tuple()` | 转为元组 | `tuple([1, 2])` → `(1, 2)` |

命令行传入的参数通常最初是字符串，可以通过转换得到用于计算的数字：

```python
value = "4"
number = int(value)
print(number + 1)  # 5
```

转换要求内容合法，例如 `int("abc")` 会抛出 `ValueError`。

## 4. 操作列表的方法

方法通常通过“对象`.`方法名”调用，例如 `history.append(...)`。

| 方法 | 作用 | 示例及结果 |
|---|---|---|
| `.append(x)` | 添加一个元素 | `[1, 2]` 添加 `3` 后变成 `[1, 2, 3]` |
| `.extend(xs)` | 逐个添加多个元素 | `[1, 2]` 扩展 `[3, 4]` 后变成 `[1, 2, 3, 4]` |
| `.pop()` | 删除并返回最后一个元素 | `[1, 2, 3]` 调用后返回 `3`，列表剩 `[1, 2]` |

`append` 和 `extend` 的区别：

```python
a = [1, 2]
a.append([3, 4])
print(a)  # [1, 2, [3, 4]]

b = [1, 2]
b.extend([3, 4])
print(b)  # [1, 2, 3, 4]
```

这几个方法会直接修改原列表。`append()` 和 `extend()` 返回 `None`，因此不要把返回值赋给原列表：

```python
history = []
history.append(0.5)  # 正确：history 变成 [0.5]

# 错误示范：执行后 history 会变成 None
# history = history.append(0.2)
```

## 5. 数据集的特殊方法

第 03、05 课中的数据集类定义了特殊方法，让对象支持 `len()` 和下标访问。

| 使用方式 | 自动调用的方法 | 含义 |
|---|---|---|
| `len(dataset)` | `dataset.__len__()` | 获取样本数量 |
| `dataset[0]` | `dataset.__getitem__(0)` | 获取第一个样本 |
| `model(value)` | `model.__call__(value)` | 把对象像函数一样调用 |

第 05 课中，创建数据集后：

```python
dataset = RelationDataset(make_samples())
print(len(dataset))  # 6
print(dataset[0])    # ((-0.8, 0.0), 0)

features, label = dataset[0]
print(features)  # (-0.8, 0.0)
print(label)     # 0
```

下标从 `0` 开始，`dataset[0]` 是第一个样本；返回值中的特征和标签可以分别接收。这里的 `RelationDataset` 和 `make_samples` 需要先定义，可对照 [05 示例](05_training_script_reading_example.py)。

## 6. 关键字和运算符：不是函数

| 写法 | 作用 |
|---|---|
| `def` | 定义函数 |
| `return result` | 从函数返回结果，结束本次函数调用 |
| `raise ValueError("提示")` | 主动抛出异常 |
| `assert condition` | 检查条件，不成立时抛出 `AssertionError` |
| `try / except` | 尝试执行代码，并捕获指定异常 |
| `==` | 比较两个值是否相等 |
| `//` | 除法后向下取整，例如 `9 // 4` 是 `2` |

`raise` 和 `print` 的区别：

```python
print("参数有问题")  # 只打印，继续执行
# raise ValueError("参数有问题")  # 抛出异常，中断当前执行
```

异常可以被外层匹配的 `except` 捕获；没有捕获时，通常会使程序停止。`assert` 适合调试和检查内部假设，使用 `python -O` 运行时会跳过断言，不能依赖它进行必要的输入检查。

### 批次数量的计算

```python
batch_count = (sample_count + batch_size - 1) // batch_size
```

当 `sample_count` 是非负整数、`batch_size` 是正整数时，这个公式计算样本一共需要多少批，最后不足一批也算一批。

| 样本数 | 每批最多几个 | 分批方式 | 批次数 |
|---|---|---|---|
| 3 | 4 | 3 | 1 |
| 6 | 4 | 4 + 2 | 2 |
| 8 | 4 | 4 + 4 | 2 |
| 9 | 4 | 4 + 4 + 1 | 3 |

例如 `6 // 4` 得到 `1`，会漏掉剩余两个样本；`(6 + 4 - 1) // 4` 得到 `2`，实现向上取整。

## 7. 第 05 课自己定义的函数

这些名称不是 Python 内置功能，需要先通过 `def` 定义，再调用。

| 函数 | 用途 |
|---|---|
| `make_samples()` | 创建样本 |
| `make_batches()` | 把样本分批 |
| `predict_by_rule()` | 根据固定规则预测标签 |
| `accuracy()` | 计算准确率 |
| `positive_integer()` | 检查并转换正整数参数 |
| `main()` | 组织整个程序的执行流程 |

阅读第 05 课时，可以沿着“准备样本 → 创建数据集 → 分批 → 预测 → 统计准确率”的顺序找这些函数。该课使用固定规则，没有计算梯度或更新模型参数。
