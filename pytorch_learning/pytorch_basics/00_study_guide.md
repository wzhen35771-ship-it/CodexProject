# 当前阶段：把张量、求导和训练连起来

如果列表、字典、函数、类、路径与循环尚不熟悉，先学习 [Python 先修指南](../python_basics/00_study_guide.md)。它将常见语法连接到样本、配置、Dataset 和训练脚本。

每天投入 5 小时，先完成这三课。目标是能独立说明和编写“输入 → 预测 → 损失 → 梯度 → 参数更新”，再进入静态场景图模型。是否进入下一阶段由验收结果决定。

已有两份作业说明你接触过张量和求导，但仅凭这两份代码，还不能判断是否能独立解释和迁移。先做下面的检查；已能完成的内容只复习错题。

## 开始前的环境检查

在你实际使用的终端中运行：

```powershell
python -c "import sys; print(sys.executable)"
python -c "import torch; print(torch.__version__); print(torch.tensor([1.0, 2.0]))"
```

第二条成功后才运行配套示例。如果提示 `No module named torch`，先解决该解释器的 PyTorch 环境问题，再开展代码练习。不同终端、编辑器和环境可能使用不同解释器，要以第一条打印的路径为准。本次已使用 VS Code 配置的 Python 解释器，在 PyTorch 2.6.0 的 CPU 环境实际运行三个示例，张量与求导检查、人工训练集拟合和保存加载检查均通过。这验证资料可运行，个人掌握程度仍由独立练习检查。

本机已验证的解释器为：

```text
C:/Users/admin/Documents/Codex/2026-10-12/ban-2/outputs/programming-starter/.venv/Scripts/python.exe
```

在 PowerShell 中运行张量示例：

```powershell
& 'C:/Users/admin/Documents/Codex/2026-10-12/ban-2/outputs/programming-starter/.venv/Scripts/python.exe' -X utf8 'C:/Users/admin/Desktop/CodexProject/pytorch_learning/pytorch_basics/01_tensors_example.py'
```

运行其他课程时替换最后的脚本文件名。这个路径是本机现有环境；其他电脑需替换为其实际解释器。

所有例子使用人工数据和 CPU，不需要图片、数据下载或显卡。训练示例在成功训练并通过检查后，把教学权重保存到本目录的 `outputs/`。

## 先做 30 分钟检查

1. 不看旧作业，写一个 `[3, 4]` 张量，取第二行、最后一列，计算两个方向的求和。
2. 对 `y = 2x² + 5x`，手算 `x=3` 时的函数值与导数，再用代码核对。
3. 说明 `backward()` 和 `optimizer.step()` 分别改变什么。
4. 说明图片 `[H, W, C]` 变为 `[C, H, W]` 应使用什么操作，为什么。

若第 1、2 项可以独立完成，第一、二课优先解决广播、维度交换、梯度累加和更新过程。第 3、4 项也能解释清楚时，直接进入第三课；不要因做过类似题就跳过训练验收。

## 7 天安排

每天按“概念与手算 1 小时、阅读 1 小时、实践 2 小时、复盘 1 小时”执行。下面是当前阶段的安排。

| 天 | 本次任务 | 必须留下的结果 |
| --- | --- | --- |
| 第 1 天 | 第一课：索引、求和、广播、形状变换 | 手算结果、逐步形状表、独立完成的张量作业 |
| 第 2 天 | 第二课：求导、梯度累加、清零、更新 | 两次迭代的参数与梯度表，独立完成的求导作业 |
| 第 3 天 | 第三课前半：数据、标签、`nn.Module`、前向 | `[N, 2] → [N, 8] → [N, 3]` 数据流与参数形状 |
| 第 4 天 | 第三课：batch、epoch、交叉熵、训练循环 | 一次迭代中每个步骤的作用；每轮更新次数 |
| 第 5 天 | 修改分类器、检查梯度和少量数据过拟合 | 训练损失和准确率记录；说明人工标签如何生成 |
| 第 6 天 | 评估、保存、加载、定位错误 | 加载前后 logits 一致的检查结果；一个错误的修复记录 |
| 第 7 天 | 脱离示例重写小分类器并讲解 | 独立程序、运行证据、未解决问题清单 |

三课对应文件：

- [01_tensors_notes.md](01_tensors_notes.md) 与 [01_tensors_example.py](01_tensors_example.py)
- [02_autograd_notes.md](02_autograd_notes.md) 与 [02_autograd_example.py](02_autograd_example.py)
- [03_training_loop_notes.md](03_training_loop_notes.md) 与 [03_training_loop_example.py](03_training_loop_example.py)

原有 [01_tensors_exercise.py](01_tensors_exercise.py) 和 [02_autograd_exercise.py](02_autograd_exercise.py) 保留。先独立重做，再核对已有答案。第三课独立练习可按命名规范自行新建 `03_training_loop_exercise.py`。

## 进入下一阶段的条件

你应能够不看示例完成以下任务：

- 给一张人工 RGB 图片完成 HWC → CHW → NCHW 变换，解释每个轴。
- 手算一个标量的导数，并解释为何求导后参数值没变。
- 写出模型、损失、清零、反向传播、更新和评估。
- 在少量人工数据上达到全部分类正确，并明确这是训练集拟合检查。
- 保存和加载模型，核对同一输入的输出。
- 解释该分类器使用已知物体位置特征，还缺少哪些从真实图片生成场景图的模块。

尚不能独立完成时，记录具体阻塞点，继续补当前阶段。能独立完成后，转入总方案中的图像、物体框、注意力和静态场景图评价。当前不把 STTran 的时间模块作为必修主线。

## 每天和每周怎么记录

每项学习写五句话：

```text
问题：今天要搞清楚什么？
操作：手算、运行或改了哪一段代码？
证据：实际输入、形状、输出、损失或报错是什么？
卡点：还无法解释什么？尝试过什么？
下一步：明天完成什么可检查的任务？
```

“已理解张量”“学了训练”过于宽泛。可以写：“给 `[2, 3, 2, 3]` 张量做按通道归一化，解释 `[1, 3, 1, 1]` 广播；运行值与手算一致。”没有运行的内容写“待验证”。

官方依据：[PyTorch 基础教程](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)。遇到 API 疑问先查看官方文档，再用小输入确认。
