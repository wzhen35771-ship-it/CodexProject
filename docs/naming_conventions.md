# 命名与目录规范

- 根 README.md 是唯一项目入口；各目录 README.md 用于导航。
- 目录与脚本使用小写英文 snake_case。中文用于正文、注释与提示。
- 课程保留两位编号：01_tensors_notes.md、01_tensors_example.py、01_tensors_exercise.py。同目录配套文件不随意移动。
- 数字开头的课程脚本通过 importlib.import_module 导入；后续新增通用模块应采用合法 Python 模块名，不以数字开头。
- papers/ 使用稳定论文简称，不用阅读顺序编号；阅读顺序集中在 docs/paper_reading_list.md。
- docs/study_plan.md 是唯一当前日程；docs/progress.md 记录当前进度；历史方案放 docs/archive/，周报放 docs/reports/ 并注明日期。
- output/pdf/ 放导出文档；tmp/ 放可重新生成的预览；课程 outputs/ 放临时权重。
- scripts/ 放工具脚本；zotero-env/.venv/ 保留原路径，避免破坏虚拟环境入口。
- 重命名必须同步修复 Markdown 链接、命令和脚本路径；个人 exercise 答案须保留。
- 函数/变量使用 snake_case，类使用 PascalCase，常量使用大写；保留框架约定名称。
- Git 提交标题可用 study: action and subject；保留已有历史。缓存、环境与敏感本地配置不提交。
