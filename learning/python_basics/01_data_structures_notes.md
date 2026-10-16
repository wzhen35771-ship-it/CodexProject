# 第 1 课：用 Python 组织特征、标签和配置

本课先认识训练脚本里的普通 Python 数据。学完应能独立解释赋值、索引、切片，以及为什么修改一个列表会影响另一个变量。

先运行 [完整示例](01_data_structures_example.py)，再逐段修改。示例只使用 Python 标准库，不需要安装 PyTorch。用主计划中的基础时段学习本课；已掌握部分可通过验收后跳过。

## 1. 赋值和常见类型

```python
learning_rate = 0.01       # float：小数
batch_size = 4            # int：整数
is_training = True        # bool：布尔值
model_name = "baseline"   # str：文本
checkpoint = None         # None：此处没有值

print(type(batch_size).__name__)  # int
print(batch_size == 4)            # True
```

`=` 把名字绑定到对象，`==` 比较值；先赋值再使用。`None` 和 `0` 含义不同：前者可以表示“尚未设置”，后者是一个具体数值。

普通数字可以做 `+`、`-`、`*`、`/` 运算；`/` 得到浮点数，`//` 是向下取整除法。不要把文本 `"4"` 当作整数 `4`，有需要时明确写 `int("4")`。

## 2. 列表、元组、字典分别装什么

```python
features = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
labels = [0, 1, 2]
sample = (features[0], labels[0])
config = {"batch_size": 2, "learning_rate": 0.01}

print(sample)                # ([0.1, 0.2], 0)
print(config["batch_size"])  # 2
```

| 容器 | 访问方式 | 可以修改吗 | 训练脚本中的常见作用 |
|---|---|---|---|
| `list` 列表 | `labels[0]` | 可以替换、追加元素 | 样本集合、指标记录 |
| `tuple` 元组 | `sample[1]` | 不能替换元组元素 | 一组配对返回值，如特征和标签 |
| `dict` 字典 | `config["batch_size"]` | 可以增删键值 | 配置、带名字的样本字段 |

字典的键和列表的位置不同。`config[0]` 不代表第一个配置项；读取不存在的键会报 `KeyError`，允许缺省时可用 `config.get("seed", 0)`。`len(labels)` 是元素数量。

元组不可变指不能写 `sample[0] = ...`；如果元组里面装着列表，那个列表自身仍可改变。本例 `sample[0]` 引用的就是 `features[0]`。

这些嵌套列表只是用于练习数据组织，尚不是张量。列表没有 PyTorch 的 `.shape`、设备或自动求导功能。

## 3. 索引和切片

```python
labels = [0, 1, 2, 3]
print(labels[0])    # 0：第一个
print(labels[-1])   # 3：最后一个
print(labels[1:3])  # [1, 2]：包含位置 1，不包含位置 3
print(labels[:2])   # [0, 1]
print(labels[::2])  # [0, 2]：每隔两个位置取一个
```

索引从 `0` 开始。单个索引越界会报 `IndexError`；切片 `labels[1:99]` 会取到实际末尾。先确认样本数，再访问 `features[index]` 和对应的 `labels[index]`。

`features[:2]` 得到外层的新列表，但里面的两行依然引用原来的列表。这与下面的浅拷贝问题相同。

## 4. 引用和浅拷贝：会直接影响数据处理

```python
original = [0, 1]
alias = original
alias.append(2)
print(original)      # [0, 1, 2]

copied = original.copy()
copied[0] = 9
print(original)      # [0, 1, 2]
print(copied)        # [9, 1, 2]
```

`alias = original` 没有复制列表。两个名字指向同一对象，修改对象会从两个名字观察到。`.copy()` 复制了外层列表，因此上例替换整数元素不会改变 `original`。

`==` 比较值，`is` 判断是不是同一个对象；这里 `alias is original` 是 `True`，`copied is original` 是 `False`。判断“尚未设置”时通常写 `checkpoint is None`。完整示例末尾的 `assert` 会检查预期结果，全部成立才打印 `checks: passed`。

遇到嵌套列表，外层复制还不够：

```python
from copy import deepcopy

rows = [[1, 2], [3, 4]]
shallow = rows.copy()
independent = deepcopy(rows)
shallow[0][0] = 99
independent[1][0] = 88

print(rows)         # [[99, 2], [3, 4]]
print(shallow)      # [[99, 2], [3, 4]]
print(independent)  # [[1, 2], [88, 4]]
```

浅拷贝的两份外层列表共享内层行；`deepcopy` 在这个普通嵌套列表里也复制了内层对象。以后处理张量时，应重新学习 `clone()`、`detach()` 的含义，不能直接把这里的规则当成张量复制方案。

## 5. 练习：先做，再展开答案

1. 为三张图片建立 `image_names` 列表、三个整数标签的 `labels` 列表，以及包含 `batch_size`、`learning_rate` 的字典。打印各容器长度及第二个样本的名称和标签。
2. 对 `[10, 20, 30, 40]`，不运行代码先写出 `[1:3]`、`[-1]`、`[::2]` 的结果。
3. 运行 `a = [1, 2]; b = a; b[0] = 8`。说明 `a` 为什么变化，再改成不会改变 `a` 的版本。
4. 用 `rows = [[1, 2], [3, 4]]` 证明切片和 `.copy()` 都是浅拷贝，再用 `deepcopy` 做一份可独立修改的版本。
5. 写出一个函数需要返回“特征和标签”时可以使用的元组形式，并解释元组中列表是否仍能修改。

<details>
<summary>自测答案与验收提示</summary>

1. 例如 `image_names = ["a.jpg", "b.jpg", "c.jpg"]`、`labels = [0, 1, 2]`、`config = {"batch_size": 2, "learning_rate": 0.01}`；样本数都是 `3`，配置项数是 `2`，第二个样本是 `image_names[1]` 和 `labels[1]`。
2. 分别是 `[20, 30]`、`40`、`[10, 30]`。
3. `a` 是 `[8, 2]`，因为 `a` 和 `b` 引用同一列表；改为 `b = a.copy()` 后修改 `b[0]`，`a` 保持 `[1, 2]`。
4. 分别创建 `sliced = rows[:]` 和 `copied = rows.copy()`，修改内层元素会改变 `rows`。对普通嵌套列表用 `independent = deepcopy(rows)` 后修改内层元素，不会改变 `rows`。每次验证要重新初始化 `rows`。
5. 返回 `(features, labels)`。不能替换元组的位置，但仍能修改它引用的列表。能够手算上述输出、解释共享对象、自行换一组数据，才算通过。

</details>

## 6. 接到 PyTorch 学习

现在只需辨认三件事：样本字段装在哪个容器、如何取到一条记录、是否意外共享了可变对象。接着学习 [循环与函数](02_control_flow_and_functions_notes.md)。后续阅读 [PyTorch 训练示例](../pytorch_basics/07_training_example.py) 时，先沿第 7 课导入找到第 6 课 `make_data()` 的返回值，再检查训练参数，不急着解释张量计算。

官方参考：[Python 入门：数字、字符串和列表](https://docs.python.org/3/tutorial/introduction.html)、[Python 数据结构](https://docs.python.org/3/tutorial/datastructures.html)。
