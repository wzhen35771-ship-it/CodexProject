# 第 5 课：把 Python 语法串成训练脚本的数据流

先用纯 Python 确认语法与数据如何流动，再进入真实 PyTorch 训练。配套示例只有固定规则预测和循环记录，没有 Tensor、梯度、损失、优化器或参数更新。

## 1. 从配置到数据

```python
config = {"epochs": 2, "batch_size": 4}
samples = [{"features": (-0.8, 0.0), "label": 0}]
```

`config` 是字典；`samples` 是由字典组成的列表；每个 `features` 是二元 tuple。`dx/dy` 表示主语中心减去宾语中心，图像 y 轴向下。标签 0、1、2 分别表示左侧、右侧、上方，仅用于人工三分类例子。

阅读训练源码时，先写清每层容器装什么。不要把 Python 列表与 Tensor 混为一谈：本课只有列表长度和元素，真正的 Tensor 形状在下一阶段学习。

## 2. 理解 Dataset 的取样约定

普通类可以实现两个方法：

```python
def __len__(self):
    return len(self.samples)

def __getitem__(self, index):
    sample = self.samples[index]
    return sample["features"], sample["label"]
```

`len(dataset)` 调用 `__len__`；`dataset[0]` 调用 `__getitem__(0)`。返回两个值时，实际返回 tuple，调用方通过 `features, label = dataset[0]` 解包。

这是 map-style Dataset 所需的主要接口约定。示例类没有继承 `torch.utils.data.Dataset`，也没有使用 `DataLoader`。后续 TensorDataset / DataLoader 会在此基础上产生真正的 Tensor 批次，并引入随机顺序等行为。

## 3. 分批与解包

示例手动按顺序分批，保留不足一个完整批次的尾批。6 个样本、`batch_size=4`，每轮得到 4 条和 2 条两个批次。

```python
for batch_index, (batch_features, batch_labels) in enumerate(batches, start=1):
    predictions = [predict_by_rule(features) for features in batch_features]
```

`enumerate` 提供编号与元素；元素又是一个二元 tuple，所以这里有两层解包。列表推导式为每条 features 调用一次函数。`extend()` 把预测逐项加入总列表；如果换成 `append(predictions)`，得到的是嵌套列表，准确率函数会面对错误的数据结构。

## 4. epoch、batch 与参数更新次数

不丢尾批时，每轮批次数是向上取整：`(sample_count + batch_size - 1) // batch_size`。默认示例运行 2 轮，每轮 2 批，总共处理 4 批。

| 概念 | 本课纯 Python 示例 | 后续真实训练 |
|---|---|---|
| 一轮 epoch | 全部人工样本遍历一次 | 训练数据遍历一次，可能提前停止 |
| 一个 batch | 一组 features 和 labels 列表 | 一组 Tensor，交给模型计算 |
| 预测 | 固定规则 `predict_by_rule` | `model(batch_features)` 得到 logits |
| 参数更新 | 0 次 | 执行一次 `optimizer.step()` 通常更新一次 |

在标准“每批更新一次”的训练里，实际运行轮数 × 每轮批次数等于更新次数。使用梯度累积、跳过异常批次或提前停止后，应按实际执行的 `optimizer.step()` 计数，不能只看命令行设置的最大 epochs。

## 5. 准确率只验证当前小例子

```python
correct = sum(prediction == label for prediction, label in zip(predictions, labels))
accuracy = correct / len(labels)
```

比较得到布尔值，求和时 True 按 1 计；`zip` 把对应的预测与标签配对。`zip` 遇到长度不等会截断，因此函数先检查长度相等且非空。本课固定规则在专门构造的六条数据上能全部命中，这不能证明模型训练有效，也不是真实场景图的 R@K / mR@K。

## 6. 运行、追踪与衔接

在项目根目录运行：

```powershell
python pytorch_learning/python_basics/05_training_script_reading_example.py
python pytorch_learning/python_basics/05_training_script_reading_example.py --epochs 3 --batch-size 5
python pytorch_learning/python_basics/05_training_script_reading_example.py --batch-size 10
```

运行前预测每轮 batch 数和每批样本数，再与输出比较。`if __name__ == "__main__":` 保证脚本直接执行时调用 `main()`；其他文件 import 它时，不会自动启动循环。

接下来阅读 [PyTorch 训练循环讲义](../pytorch_basics/03_training_loop_notes.md) 与 [真实训练示例](../pytorch_basics/03_training_loop_example.py)：

1. 找到 `argparse`、`make_artificial_data()`、`TensorDataset` 和 `DataLoader`，逐步画出数据流。
2. 找到 `SmallRelationClassifier(nn.Module)`，区分构造函数创建层与 `forward()` 计算输出。
3. 标出 `optimizer.zero_grad()`、`loss.backward()`、`optimizer.step()` 的位置，并用 PyTorch 课程解释作用。
4. 解释为何该真实示例提前停止时，实际更新次数不能简单用最大 epochs 计算。

本课先完成“代码在调用谁、数据是什么、循环执行几次”；反向传播的数学和 Tensor 操作在 PyTorch 课程继续验收。

## 综合作业

打开 [作业文件](05_training_script_reading_exercise.py)，实现 `count_batches`、`make_batches`、`compute_accuracy`。先手算 `7 条样本、batch_size=3、epochs=2` 的批次数与批次长度，再完成代码。

```powershell
python pytorch_learning/python_basics/05_training_script_reading_exercise.py --check
```

作业初始状态含 `NotImplementedError`，`--check` 会明确提示未完成并以退出码 2 结束。完成后，自检应验证尾批、空数据、无效 batch_size、预测长度不等和准确率边界；最后用自己写的一组样本解释循环。自检通过不等于已掌握 PyTorch 训练。

验收：解释普通 Dataset 类的两种调用；画出配置到准确率的数据流；独立完成三个函数；明确本课参数更新次数为 0，并指出真实训练更新参数的位置。

<details>
<summary>参考答案：独立完成后再展开</summary>

7 条样本、每批 3 条，批次长度为 `[3, 3, 1]`；每轮 3 批，2 轮共 6 批。若真实训练每批执行一次 `optimizer.step()`，则更新 6 次；本课规则预测仍更新 0 次。

```python
def count_batches(sample_count, batch_size):
    if sample_count < 0 or batch_size < 1:
        raise ValueError("样本数必须非负，batch_size 必须为正数")
    return (sample_count + batch_size - 1) // batch_size


def make_batches(samples, batch_size):
    if batch_size < 1:
        raise ValueError("batch_size 必须为正数")
    return [samples[start:start + batch_size] for start in range(0, len(samples), batch_size)]


def compute_accuracy(predictions, labels):
    if len(predictions) != len(labels) or not labels:
        raise ValueError("预测与标签必须等长且非空")
    return sum(prediction == label for prediction, label in zip(predictions, labels)) / len(labels)
```

</details>
