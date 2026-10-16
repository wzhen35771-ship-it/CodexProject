# SGG 学习项目

这里集中保存 Python/PyTorch 基础课程、场景图论文和学习记录。文件采用英文命名，说明与注释使用中文。

## 现在从哪里开始

1. 查看 [当前进度与下一步](docs/progress.md)。
2. 按 [十周学习计划](docs/study_plan.md) 安排每天 5 小时；每周学习 6 天、休息 1 天。
3. 本轮先完成 BGNN 消融阅读，再做 [第七课训练练习](learning/pytorch_basics/07_training_exercise.py) 和 [第八课评估练习](learning/pytorch_basics/08_evaluation_exercise.py)。不会自动把文件存在视为已经掌握。

## 目录与入口

| 目录 | 内容 |
|---|---|
| [learning/](learning/README.md) | Python 和 PyTorch 讲义、示例、个人练习 |
| [docs/](docs/study_plan.md) | 唯一主学习计划、进度、论文索引、命名规范及历史记录 |
| [papers/](papers/README.md) | 十篇原始论文 PDF；text/ 保存综述提取全文 |
| [output/pdf/](output/pdf/sgg_survey_notes.pdf) | 已导出的综述笔记和历史周报 |
| [scripts/](scripts/README.md) | 代理启动与笔记生成脚本 |
| [zotero-env/](zotero-env/README.md) | 已配置的 PDF 提取环境，保持原路径 |

## 使用约定

- 课程编号保持不变，同一课的 notes/example/exercise 对应同一主题；个人作业不替换为答案。
- 已完成的讲解与独立验收分开记录，见 [进度](docs/progress.md)。
- 日程只维护 [study_plan.md](docs/study_plan.md)；论文阅读细节放在 [paper_reading_list.md](docs/paper_reading_list.md)。
- 标准库示例可用系统 Python；PyTorch 使用已安装 torch 的解释器，见 [环境说明](docs/environment.md)。
- 论文保留原始内容，文件名不承担阅读排序；顺序由索引维护。
- 缓存、虚拟环境、生成权重及本地配置不加入版本管理；PDF 成果与原论文可跟踪。

本次目录迁移与清理见 [整理记录](docs/project_cleanup.md)。
