# 项目整理记录

日期：2026-10-16。本次整理所有工作区课程、论文、成果与工具文件；.git、现有虚拟环境和工作区外应用配置未修改。

## 新目录

```text
CodexProject/
├── README.md                 项目唯一入口
├── docs/                     当前计划、进度、论文索引、环境与验证
│   ├── archive/              旧研究方案，明确标为历史
│   └── reports/              按日期归档周报
├── learning/                 原 pytorch_learning 基础课程
│   ├── python_basics/
│   └── pytorch_basics/
├── papers/                   十篇原始 PDF，稳定简称命名
│   └── text/                 综述提取全文
├── output/pdf/               中文阅读笔记与历史周报 PDF
├── scripts/                  代理启动、PDF 源脚本
└── zotero-env/               原有 PDF 提取虚拟环境
```

## 文件命名与迁移

- pytorch_learning/ 中的课程迁到 learning/；所有课号与配套脚本名保留，跨课导入仍有效。
- 根 SGG学习计划.md → docs/study_plan.md，作为唯一当前日程。
- 旧八周研究方案 → docs/archive/research_plan_2026_10_14.md，保留原研究假设与旧数值，标明未在本次核实。
- 周报正文 → docs/reports/weekly_report_2026_10_13.md，PDF → output/pdf/weekly_report_2026_10_13.pdf。
- 原论文移到 papers/，移除阅读顺序编号，例如 03_reltr.pdf → reltr.pdf。
- 综述原文 → papers/sgg_survey_2022_v2.pdf，笔记 → output/pdf/sgg_survey_notes.pdf，提取全文 → papers/text/sgg_survey_2022_v2.txt。
- 代理入口 → scripts/start_codex_proxy.cmd 与 .ps1；默认端口从旧 7897 统一为 7892，并允许 -Port 参数。
- PDF 生成脚本由 tmp/pdfs/ 移到 scripts/build_sgg_notes.py，同步修复根目录和输出路径。

## 内容修正

统一十周计划与当前课程进度；保留旧方案供追溯。修复指向已删除三课版的链接、错误类名、数据函数名和并不存在的提前停止说明。将综述版本纠正为当前使用的 2022 年 v2，而非后续同名期刊版本。论文清单补充 Deformable DETR 阅读入口，明确尚未下载本地 PDF。

环境描述改为本次实际核对的 PyTorch 2.6.0+cpu。练习存在或示例运行成功不再直接等同于掌握课程。

## 清理与保留

删除已安装 Zotero 的旧安装包、六张可再生 PDF 预览图和旧目录 Python 缓存，清除空目录。实际删除大小见下方统计。十篇论文、两份 PDF 成果、全部个人练习答案保留并核对哈希；现有 PDF 提取环境保留原路径。

原文本回退包位于 .maintenance/original_texts_2026_10_16.zip，文件迁移映射位于 .maintenance/file_moves.json。该目录被 Git 忽略，包含本机路径；它是小型回退记录，不是第二份学习材料。

验证详情见 [验证记录](verification.md)。没有替用户提交 Git，也没有向其他聊天发送消息。其他聊天中的旧文件链接需要改用新入口。

清理统计：删除 111627031 字节，约 106.46 MiB（约 111.63 MB）。
