# PyTorch 八课学习指南

更新日期：2026-10-16。当前已讲解到训练循环与评估，独立掌握程度见 [学习进度](../../docs/progress.md)。已学内容先验收，再决定补哪一课。

目标：独立理解和编写“数据 → 模型预测 → 损失 → 梯度 → 参数更新 → 评估 → 保存加载”。先打好基础，再衔接静态场景图研究。

## 学习节奏

完全从零时可参考每课两天的节奏；表中的天数仅为完整课程估算。当前日程统一按 [十周计划](../../docs/study_plan.md)，不与它额外叠加。

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
& $studyPython -X utf8 'C:/Users/admin/Desktop/CodexProject/learning/pytorch_basics/01_tensors_example.py'
```

替换最后的文件名即可运行后续示例。第 7、8 课会导入前面的配套文件，请将整套课程放在同一目录。第 8 课保存权重到本课程 outputs/。练习脚本是待完成模板，直接运行没有输出是正常现象。

如果使用自己的环境，先执行 `python -c "import sys, torch; print(sys.executable); print(torch.__version__)"`；导入失败时先确认编辑器与终端解释器一致，再按 [官方安装入口](https://pytorch.org/get-started/locally/) 选择适合系统的命令。无需为了本课升级已可运行的环境。

## 当前任务与记录

先完成 [第七课练习](07_training_exercise.py)，再做 [第八课练习](08_evaluation_exercise.py)。若张量、求导或模型概念卡住，回到对应讲义；不要求所有内容从第一课重读。

每课记录实际命令、关键输出、自己的解释和未解决问题。已有练习可能包含个人答案，也可能尚空；示例运行通过不等于个人验收通过。环境见 [环境说明](../../docs/environment.md)，示例历史记录见 [运行记录](00_verification.md)。

## 结课验收与后续研究

不看示例写出完整分类程序；解释输入、标签、logits 和 loss 的形状；证明 backward 只求导、step 才更新；区分训练和评估；核对保存加载后的 logits。完成前继续补基础。

完成后按 [静态场景图方案](../../docs/study_plan.md) 学图像特征、物体框、物体对、关系预测和评价。人工关系分类输入已经是位置差，尚未实现图像检测或真实场景图生成。

参考路线：[PyTorch 官方基础教程](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)。本课程为适应从零学习而细分顺序与人工练习，不是官方教程的逐字翻译。
