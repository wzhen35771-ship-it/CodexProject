# 面向 PyTorch 的 Python 先修学习

本目录补足读写 PyTorch 学习代码所需的 Python 能力。文件和目录采用英文命名，讲解使用中文。示例只用标准库；真实 Tensor、自动求导和训练放在相邻的 [PyTorch 课程](../pytorch_basics/00_study_guide.md)。

每天投入 5 小时，按 5 天安排。你已掌握的内容可以用当天验收证明后跳过；Python 能力未过关时，先补这一部分，再继续总体计划。整体进度见 [学习方案](../02_scene_graph_study_plan.md)，全部入口见 [学习总览](../00_overview.md)。

| 天数 | 学习主题与资料 | 当天可检查结果 |
|---|---|---|
| 第 1 天 | [数据结构讲义](01_data_structures_notes.md)、[示例](01_data_structures_example.py) | 能访问与修改样本列表、字典、tuple，区分索引与切片 |
| 第 2 天 | [控制流与函数讲义](02_control_flow_and_functions_notes.md)、[示例](02_control_flow_and_functions_example.py) | 能写函数、循环、条件分支，并解释返回值与解包 |
| 第 3 天 | [类与模块讲义](03_classes_and_modules_notes.md)、[示例](03_classes_and_modules_example.py) | 能解释 self、构造、方法调用、继承和模块入口 |
| 第 4 天 | [路径、配置与排错讲义](04_files_config_and_debugging_notes.md)、[示例](04_files_config_and_debugging_example.py) | 能定位路径，读写 JSON，传 CLI 参数，确认解释器并读 traceback |
| 第 5 天 | [训练脚本阅读讲义](05_training_script_reading_notes.md)、[示例](05_training_script_reading_example.py)、[综合作业](05_training_script_reading_exercise.py) | 能追踪批次与控制流，独立完成自检，并指出真实训练的参数更新位置 |

每天 5 小时分配：概念 1 小时、阅读并手算示例 1 小时、修改示例与独立练习 2 小时、记录验证结果和卡点 1 小时。先预测输出，再运行；对能解释的部分做小改动；最后不看答案完成验收。

阅读 01～05 时，可随时查看 [常用函数与语法速查](common_functions_cheatsheet.md)：包含 `len`、`range`、`enumerate`、`zip`、列表方法，以及 `raise`、`assert` 和批次数量公式。

## 运行方式

打开 VS Code 的项目根目录 `CodexProject`，在 PowerShell 运行：

```powershell
python -c "import sys; print(sys.executable)"
python pytorch_learning/python_basics/01_data_structures_example.py
python pytorch_learning/python_basics/02_control_flow_and_functions_example.py
python pytorch_learning/python_basics/03_classes_and_modules_example.py
python pytorch_learning/python_basics/04_files_config_and_debugging_example.py
python pytorch_learning/python_basics/05_training_script_reading_example.py
```

如果终端已经位于本目录，直接运行 `python 01_data_structures_example.py`。每次排查依赖先核对实际解释器；第 4 课介绍 `python -m pip` 和虚拟环境。这里无需安装 PyTorch，不会自动安装包。

## 怎样判定完成

每课记录四项：实际命令、关键输出、你对结果的解释、尚未解决的问题。只阅读文字或复制成功输出，不能作为独立完成的证据。参考答案放在讲义的折叠区，先完成练习再对照。

第 5 天综合作业初始包含 TODO；未完成时 `--check` 会提示作业尚未完成。把它改完并通过自检后，再对照 [Tensor 讲义](../pytorch_basics/01_tensors_notes.md)、[自动求导讲义](../pytorch_basics/02_autograd_notes.md)、[训练循环讲义](../pytorch_basics/03_training_loop_notes.md)，检查 Python 语法是否已能独立解释。

完整 PyTorch 示例能否运行，还取决于当前解释器是否安装 PyTorch。先修课的规则预测准确率不能替代真实模型训练或场景图指标。

参考阅读：[Python 官方教程](https://docs.python.org/zh-cn/3/tutorial/)、[argparse](https://docs.python.org/zh-cn/3/library/argparse.html)、[pathlib](https://docs.python.org/zh-cn/3/library/pathlib.html)、[venv](https://docs.python.org/zh-cn/3/library/venv.html)。
