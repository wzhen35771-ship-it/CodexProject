# 当前学习入口

课题：基于多路径自适应融合的静态场景图生成。

文件名、目录名和代码标识符使用英文，讲解与注释使用中文。每天投入 5 小时，按照 Python 先修、PyTorch 基础、静态场景图模型和研究实验推进。

| 入口 | 用途 |
| --- | --- |
| [Python 先修指南](python_basics/00_study_guide.md) | 读训练代码需要的 Python 语法，5 天安排、示例与综合练习 |
| [PyTorch 基础指南](pytorch_basics/00_study_guide.md) | 张量、自动求导、训练循环的三课讲义与独立作业 |
| [静态场景图学习方案](02_scene_graph_study_plan.md) | 先修检查、8 周主线、当前任务与研究实验验收 |
| [论文阅读清单](04_paper_reading_list.md) | 综述、注意力、检测、关系预测、图传递与长尾的分阶段阅读顺序 |
| [当前学习周报](03_weekly_report.md) | 已学习内容、待验收问题与下一周计划 |
| [命名规范](01_naming_conventions.md) | 英文命名、文件组织与代码名称 |

## 从哪里开始

先检查能否独立操作列表和字典、定义函数、解释类的属性与方法、定位文件路径，以及读懂 `for` 循环和函数返回值。尚不熟悉时，从 `python_basics/` 开始；能独立完成该专题综合练习时，进入 `pytorch_basics/`。

Python 示例使用标准库，可以先运行；PyTorch 示例需使用已经安装 PyTorch 的解释器。基础练习与研究仓库分别管理环境。

周报 PDF 位于工作区的 `output/pdf/weekly_report.pdf`，可编辑正文为 `03_weekly_report.md`。后续按验收结果接入真实图像、静态模型和标准评价。
