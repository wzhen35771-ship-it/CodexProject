# PyTorch 第一课：张量

目标：能创建张量、读懂形状、取出一行或一列，并解释简单运算。

学习环境使用处理器（CPU）。Python 解释器在 `.venv/Scripts/python.exe`，无须激活环境。

在 PowerShell 中运行：

```powershell
& 'E:\CodexProject\pytorch_learning\.venv\Scripts\python.exe' 'E:\CodexProject\pytorch_learning\01_tensors.py'
```

## 把张量理解为按维度排列的数字

```python
import torch

x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
print(x)
print(x.shape)  # torch.Size([2, 3])：2 行、3 列
```

这是二维张量。可以假设每行对应一个物体，每个物体有三个示例特征；这里的数字只是练习数据。

常见形状：单个数字是 `[]`；三个数字组成的向量是 `[3]`；两行三列是 `[2, 3]`。

## 取出数据

索引从 0 开始：

```python
print(x[0])     # 第一行：[1, 2, 3]
print(x[1, 2])  # 第二行第三列：6
print(x[:, 1])  # 所有行的第二列：[2, 5]
```

冒号 `:` 表示这个维度上的全部元素。

## 运算和维度

```python
print(x + 10)        # 每个元素加 10
print(x * 2)         # 每个元素乘 2
print(x.sum(dim=0))  # 沿行的方向求和，每一列得到一个和：[5, 7, 9]
print(x.sum(dim=1))  # 沿列的方向求和，每一行得到一个和：[6, 15]
```

参数 `dim` 表示维度，指定沿哪个维度求和。这个二维例子中，`dim=0` 后剩下列，`dim=1` 后剩下行。

代码中的函数名和属性名要保留原写法，它们的中文含义如下：

| 代码写法 | 中文含义 |
| --- | --- |
| `torch.tensor` | 创建张量 |
| `shape` | 形状 |
| `dtype` | 数据类型 |
| `device` | 运行设备 |
| `sum` | 求和 |
| `mean` | 求平均值 |
| `dim` | 维度 |

## 今天的练习

1. 创建一个形状为 `[3, 4]` 的张量。
2. 取出第二行、最后一列以及第一行第二列。
3. 先在纸上预测 `sum(dim=0)` 和 `sum(dim=1)` 的值和形状，再运行核对。
4. 可选：读脚本中的矩阵乘法例子。`*` 是逐元素乘法，`@` 是矩阵乘法。

下一课：自动求导；之后依次学习线性模型、损失与优化器、数据加载、分类模型、关系损失掩码。

官方参考：[PyTorch 张量教程（英文原文）](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)。
