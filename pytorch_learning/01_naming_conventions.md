# 英文命名与资料组织规范

## 文件和目录

名称统一使用小写英文与下划线 `snake_case`，课程保留两位数字序号，说明文字和注释使用中文。

| 用途 | 名称或示例 |
| --- | --- |
| 学习入口 | `00_overview.md` |
| 命名规范 | `01_naming_conventions.md` |
| 研究学习方案 | `02_scene_graph_study_plan.md` |
| 当前周报 | `03_weekly_report.md` |
| Python 先修目录 | `python_basics/` |
| PyTorch 基础目录 | `pytorch_basics/` |
| 专题入口 | `00_study_guide.md` |
| 讲义 | `01_tensors_notes.md` |
| 示例 | `01_tensors_example.py` |
| 个人练习 | `01_tensors_exercise.py` |
| 教学权重 | `03_training_loop_weights.pt`，运行成功后存入专题的 `outputs/` |
| 周报导出 | 工作区 `output/pdf/weekly_report.pdf` |

同一课的讲义、示例与练习采用相同序号和主题，末尾分别使用 `_notes`、`_example`、`_exercise`。个人练习的内容由学习者完成，重命名时保持其答案。

每个专题设置一个入口，讲义直接链接配套示例与后续课程。重命名时同步检查 Markdown 链接、运行命令和代码中的保存路径。

## Python 代码名称

函数、变量、参数与属性使用英文 `snake_case`，类使用 `PascalCase`，常量使用大写英文。注释、讲解、输出提示可以使用中文。

保留框架的 `forward`、`__init__`、`__getitem__` 等约定名称，以及引用的官方项目源码文件名。类、属性和函数名称不要与数据含义混淆。

## 环境与生成文件

`requirements.txt` 只维护基础课程依赖；Python 先修示例使用标准库。研究仓库另设环境。`.venv/`、`__pycache__/`、`outputs/` 按现有规则忽略。

## Git 提交标题

新提交标题使用简洁英文，格式为 `study: action and subject`，例如 `study: add Python prerequisites for PyTorch`。标题以实际变更为准，保留现有历史。
