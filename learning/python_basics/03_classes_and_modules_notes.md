# 第 3 课：读懂 Dataset 和模型类的 Python 外壳

前置：[循环与函数](02_control_flow_and_functions_notes.md)。本课的任务是能说明 `self` 指谁、对象保存了什么、`dataset[index]` 和 `model(value)` 如何调用方法。

运行 [完整示例](03_classes_and_modules_example.py)。它只使用 Python 标准库，没有自动求导或参数更新。用主计划中的基础时段学习本课，完成后独立复述。

## 1. 类、实例、属性和方法

```python
class ScalarLinear:
    def __init__(self, weight, bias=0.0):
        self.weight = weight
        self.bias = bias

    def forward(self, value):
        return self.weight * value + self.bias

model = ScalarLinear(weight=2.0, bias=1.0)
print(model.weight)        # 2.0：实例属性
print(model.forward(3.0))  # 7.0：调用实例方法
```

`class` 定义一种对象的结构。`model` 是这个类的一个实例；`__init__` 在新实例初始化时运行。`self` 是当前实例的约定参数名，`self.weight` 把值存到这个实例上。调用 `model.forward(3.0)` 时，Python 自动把 `model` 作为 `self` 传入。

创建 `other = ScalarLinear(4.0)` 会得到另一个实例。修改 `model.weight` 不会自动修改 `other.weight`。本例把属性写在 `__init__` 中；不要把每次运行独立的历史列表随手写成共享的类属性。

## 2. 三个特殊方法：让对象使用熟悉的语法

```python
class FeatureDataset:
    def __init__(self, features, labels):
        if len(features) != len(labels):
            raise ValueError("features and labels must have the same length")
        self.features = tuple(tuple(row) for row in features)
        self.labels = tuple(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {"features": self.features[index], "label": self.labels[index]}

dataset = FeatureDataset([[1.0, 2.0], [3.0, 4.0]], [0, 1])
print(len(dataset))  # 2：调用 __len__
print(dataset[0])    # {'features': (1.0, 2.0), 'label': 0}：调用 __getitem__
```

这里把普通数字组成的行转换为元组，减少意外改动原始列表的机会。`__getitem__` 的参数是位置索引；超出范围时，内部元组会报 `IndexError`。

在前面的 `ScalarLinear` 内增加：

```python
    def __call__(self, value):
        return self.forward(value)
```

现在 `model(3.0)` 得到 `7.0`，调用的是 `__call__`。这些特殊方法是 Python 协议，不是 PyTorch 独有语法。

## 3. 继承与 super()

```python
class ScaledFeatureDataset(FeatureDataset):
    def __init__(self, features, labels, scale):
        super().__init__(features, labels)
        self.scale = scale

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        scaled = tuple(value * self.scale for value in sample["features"])
        return {"features": scaled, "label": sample["label"]}

dataset = ScaledFeatureDataset([[1.0, 2.0]], [0], scale=2.0)
print(len(dataset))  # 1：沿用父类的 __len__
print(dataset[0])    # {'features': (2.0, 4.0), 'label': 0}
```

括号中的 `FeatureDataset` 是父类；子类沿用父类的方法，也能覆盖同名方法。这个单继承例子里的 `super()` 让子类调用父类初始化和读取逻辑，不用重复写一份。

要先调用父类初始化，再使用父类创建的属性。不要把 `super()` 理解为“会自动训练模型”。

## 4. import 和 main 入口

```python
import math
from math import sqrt

print(math.sqrt(9))  # 3.0：模块名.函数名
print(sqrt(16))     # 4.0：直接使用导入的名字

def main():
    print("start")

if __name__ == "__main__":
    main()
```

模块通常是一个 `.py` 文件；`import` 导入模块，`from ... import ...` 导入模块中的指定名字。Python 的 `math` 是标准库模块；以后 `import torch` 导入的是另行安装的包。

直接运行脚本时，`__name__` 是 `"__main__"`；正常导入它时，`__name__` 是模块名，入口中的 `main()` 不会执行。不过其他顶层语句仍会执行，因此把演示打印和训练启动放到 `main()` 内。

课程文件用数字编号方便排序，可以直接运行；若要写 `import dataset` 这样的常规导入，模块文件名应是合法的 Python 标识符，例如 `dataset.py`，不应以数字开头。本课只导入标准库，不需要另建文件。

## 5. 与 PyTorch 对应时必须分清的边界

| 本课的 Python 结构 | 后续 PyTorch 代码 | 需要补学的内容 |
|---|---|---|
| `__len__`、`__getitem__` | 常见的 map-style `Dataset` | 返回张量或其他样本字段，结合 `DataLoader` 组成 batch |
| `__init__`、实例属性 | `nn.Module` 子类和网络层 | `super().__init__()`、层与参数的注册 |
| `forward` 方法 | `nn.Module.forward` | 张量运算与输入输出形状 |
| `model(value)` | `model(features)` | `nn.Module` 的调用流程和 hooks 等框架功能 |

**普通 Python 类保存一个 `float` 属性不会获得梯度，也不会注册可训练参数。** 本课的 `ScalarLinear` 只算数值结果，不能直接交给 PyTorch 优化器作为可训练模型。

PyTorch 初学时按标准方式继承 `nn.Module`、调用 `super().__init__()`、在 `forward` 中写计算，然后调用 `model(features)`。无需照搬本课手写的 `__call__`；框架已经提供了它。阅读 [现有模型示例](../pytorch_basics/05_models_example.py)，找出 `SmallClassifier` 的父类、初始化、两个层和 `forward`，然后进入正式 PyTorch 课程。

## 6. 练习

1. 创建两个 `ScalarLinear` 实例，分别设置不同的 `weight`。修改其中一个，再打印两个属性，证明实例彼此独立。
2. 创建含三条样本的 `FeatureDataset`，使用 `len(dataset)` 和 `dataset[1]`，写出调用的特殊方法及实际输出。
3. 创建 `ScaledFeatureDataset`，令 `scale=0.5`，验证 `[2.0, 4.0]` 变成 `(1.0, 2.0)`，标签保持原值。
4. 为 `ScalarLinear` 增加 `__call__`，验证 `model(3.0)` 与 `model.forward(3.0)` 相等，并说明 PyTorch 课为什么无需照搬这个实现。
5. 在完整示例里暂时把一条打印放到 `main()` 外，解释直接运行和导入时有什么差别。最后恢复，把“类定义、导入、演示运行”分开标注。

<details>
<summary>自测答案与验收提示</summary>

1. 例如 `first = ScalarLinear(2.0)`、`second = ScalarLinear(4.0)`，执行 `first.weight = 8.0` 后，两者分别为 `8.0` 和 `4.0`。
2. `len(dataset)` 调用 `__len__`；`dataset[1]` 调用 `__getitem__(1)`。用 `[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]` 和 `[0, 1, 2]` 初始化时，输出 `3` 和 `{'features': (3.0, 4.0), 'label': 1}`。
3. `ScaledFeatureDataset([[2.0, 4.0]], [1], scale=0.5)[0]` 输出 `{'features': (1.0, 2.0), 'label': 1}`。通过 `super().__getitem__` 取样，再缩放数值。
4. 方法体是 `return self.forward(value)`；权重 `2.0`、偏置 `1.0` 时，两种调用都是 `7.0`。`nn.Module` 已有包含框架逻辑的调用实现，普通转发函数不能替代它。
5. 入口外的顶层打印在正常导入时也会执行；入口内打印只在直接作为脚本运行时执行。类定义导入后可供调用，但定义方法并不等于执行方法。能解释 `self`、三个特殊方法、父类初始化和普通类与 `nn.Module` 的区别，才算通过。

</details>

官方参考：[Python 类与继承](https://docs.python.org/3/tutorial/classes.html)、[Python 模块与脚本入口](https://docs.python.org/3/tutorial/modules.html)。
