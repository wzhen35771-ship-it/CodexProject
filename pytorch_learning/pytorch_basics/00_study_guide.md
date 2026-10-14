# PyTorch 学习规划：从第一课重新开始

更新日期：2026-10-14。当前起点明确为第一课，不根据目录里已有代码推断你已经掌握。

目标：独立理解和编写“数据 → 模型预测 → 损失 → 梯度 → 参数更新 → 评估 → 保存加载”。先打好基础，再衔接静态场景图研究。

## 学习节奏

建议 16 天完成 8 课，每课两天；进度由独立练习决定。沿用每天最多 5 小时的安排：概念和手算 1 小时、逐行读代码 1 小时、独立练习 2 小时、复盘 1 小时。累时可拆开，不必为了赶日期跳课。

第一天读讲义、预测并运行示例；第二天关闭示例独立练习、修复错误。列表、函数、循环、类的语法卡住时，查阅 [Python 先修指南](../python_basics/00_study_guide.md)，再回到当前课。

| 课 | 讲义 | 示例代码 | 建议安排 |
| --- | --- | --- | --- |
| 1 | [认识张量](01_tensors_notes.md) | [01_tensors_example.py](01_tensors_example.py) | 第 1～2 天 |
| 2 | [运算、求和与广播](02_operations_notes.md) | [02_operations_example.py](02_operations_example.py) | 第 3～4 天 |
| 3 | [形状变换与图像表示](03_shapes_notes.md) | [03_shapes_example.py](03_shapes_example.py) | 第 5～6 天 |
| 4 | [自动求导与参数更新](04_autograd_notes.md) | [04_autograd_example.py](04_autograd_example.py) | 第 7～8 天 |
| 5 | [神经网络与损失](05_models_notes.md) | [05_models_example.py](05_models_example.py) | 第 9～10 天 |
| 6 | [Dataset、DataLoader 与批次](06_data_notes.md) | [06_data_example.py](06_data_example.py) | 第 11～12 天 |
| 7 | [完整训练循环](07_training_notes.md) | [07_training_example.py](07_training_example.py) | 第 13～14 天 |
| 8 | [评估、保存加载与综合验收](08_evaluation_notes.md) | [08_evaluation_example.py](08_evaluation_example.py) | 第 15～16 天 |

## 运行环境与命令

全部示例使用 CPU 和人工数据，无需显卡或下载数据。已在本机发现安装 PyTorch 2.6.0+cpu 的解释器：

```powershell
$studyPython = 'C:/Users/admin/Documents/Codex/2026-10-12/ban-2/outputs/programming-starter/.venv/Scripts/python.exe'
& $studyPython -c "import sys, torch; print(sys.executable); print(torch.__version__)"
& $studyPython -X utf8 'C:/Users/admin/Desktop/CodexProject/pytorch_learning/pytorch_basics/01_tensors_example.py'
```

替换最后的文件名即可运行后续示例。第 7、8 课会导入前面的配套文件，请将整套课程放在同一目录。第 8 课保存权重到本课程 outputs/。练习脚本是待完成模板，直接运行没有输出是正常现象。

如果使用自己的环境，先执行 `python -c "import sys, torch; print(sys.executable); print(torch.__version__)"`；导入失败时先确认编辑器与终端解释器一致，再按 [官方安装入口](https://pytorch.org/get-started/locally/) 选择适合系统的命令。无需为了本课升级已可运行的环境。

## 今天只做第一课

1. 打开第一课讲义，区分 shape、ndim、numel。
2. 预测标量、一维、二维张量的打印结果。
3. 运行第一课示例，核对第二行与最后一列。
4. 独立完成第一课练习，并回答：三行两列是几维、几个元素？

不要今天同时学习求导和训练循环。第一课验收通过后，再学第二课。

## 学习记录

每课记五项：今天的问题、自己写的代码、实际输入输出、卡住的位置、下一次任务。未完成课保持“待学习”；示例已运行不等于个人练习已通过。

| 课 | 初始学习状态 |
| --- | --- |
| 第一课 | 从这里开始，待独立验收 |
| 第二至八课 | 待学习 |

## 结课验收与后续研究

不看示例写出完整分类程序；解释输入、标签、logits 和 loss 的形状；证明 backward 只求导、step 才更新；区分训练和评估；核对保存加载后的 logits。完成前继续补基础。

完成后按 [静态场景图方案](../02_scene_graph_study_plan.md) 学图像特征、物体框、物体对、关系预测和评价。人工关系分类输入已经是位置差，尚未实现图像检测或真实场景图生成。

参考路线：[PyTorch 官方基础教程](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)。本课程为适应从零学习而细分顺序与人工练习，不是官方教程的逐字翻译。
