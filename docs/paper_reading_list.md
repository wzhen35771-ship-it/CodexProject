# 论文阅读清单

顺序由本页维护，时间安排统一见 [十周计划](study_plan.md)。PDF 文件名不随阅读顺序变化。已完成综述第一遍，BGNN 本轮先做概念收尾；基础验收后进入 Attention。

| 阶段 | 本地论文 / 原文 | 阅读重点与验收 |
|---|---|---|
| 已读回查 | [综述 PDF](../papers/sgg_survey_2022_v2.pdf) · [2022 v2 原文](https://arxiv.org/abs/2201.00443v2) | 场景图、任务、数据、R/mR；写出有方向三元组，说明已知输入与待预测内容 |
| 当前收尾 | [BGNN](../papers/bgnn.pdf) · [原文](https://arxiv.org/abs/2104.00308v2) | 二部图、置信度控制、双层重采样；解释一次传播和一张消融表 |
| 注意力基础 | [Attention Is All You Need](../papers/attention_is_all_you_need.pdf) · [原文](https://arxiv.org/abs/1706.03762) | 手算 Q/K/V 注意力，多头、自/交叉注意力、残差和位置编码；翻译实验按需查阅 |
| 检测基础 | [DETR](../papers/detr.pdf) · [原文](https://arxiv.org/abs/2005.12872) | 查询、集合预测、匹配、损失；用 3 个预测和 2 个标注解释匹配 |
| 三元组预测 | [RelTR](../papers/reltr.pdf) · [原文](https://arxiv.org/abs/2201.11460) | 主宾查询、谓词输出、三元组监督；追踪一条三元组并与 DETR 比较 |
| 核心补读 | [Deformable DETR 原文](https://arxiv.org/abs/2010.04159)（尚未保存本地 PDF） | 参考点、采样偏移、权重与多尺度；为 EGTR 源码打基础 |
| 关系特征提取 | [EGTR](../papers/egtr.pdf) · [原文](https://arxiv.org/abs/2404.02072) | 自注意力 Q/K 特征、门控聚合、连通性、自适应平滑；找到前向、损失、推理入口 |

## 按研究问题补读

| 方向 | 论文 | 重点 |
|---|---|---|
| 对象与关系联合表示 | [Relationformer](../papers/relationformer.pdf) · [原文](https://arxiv.org/abs/2203.10202) | object token 与 relation token 如何共同支持关系预测 |
| 图传播与长尾 | [BGNN](../papers/bgnn.pdf) | 深入公式和实现，区分传播机制与重采样的贡献 |
| 类别先验 | [Logit Adjustment](../papers/logit_adjustment.pdf) | 区分训练中/训练后调整，手算得分变化 |
| 表示与混淆 | [PE-Net](../papers/pe_net.pdf) | 原型学习与正则化；区分表示改进与得分校正 |
| 监督分配 | [Hydra-SGG](../papers/hydra_sgg.pdf) · [原文](https://arxiv.org/abs/2409.10262) | 一对一/一对多匹配、假负例、辅助分支；先掌握集合匹配 |

## 三轮阅读

1. 问题与结构：摘要、引言、结构图、结论，写输入输出与核心改动。
2. 方法与形状：一次一个模块，解释公式变量并用小例子核对。
3. 代码与实验：定位前向、损失、推理，再用消融核对模块作用。

每篇保留一页总览：问题、流程图、核心公式、关键形状、实验依据、未解决问题。推导和调试另记。示例运行、单图推理、小样本训练和完整复现分开记录。

## 作者代码入口

- [DETR](https://github.com/facebookresearch/detr)
- [RelTR](https://github.com/yrcong/RelTR)
- [EGTR](https://github.com/naver-ai/egtr)
- [Relationformer](https://github.com/suprosanna/relationformer)
- [BGNN](https://github.com/Scarecrow0/BGNN-SGG)
- [PE-Net](https://github.com/VL-Group/PENET)
- [Hydra-SGG](https://github.com/Dreamer312/Hydra-SGG)

这些是作者入口，本次整理未核验远端最新发布状态；实际使用时记录提交、依赖、权重和数据协议。基础 CPU 环境不自动满足研究仓库要求。

综述 PDF 与提取文本均对应当前正在读的 Zhu 等人预印本；不要替换成同名后续期刊版本的作者或年份。其他 PDF 保持原字节，本次重命名不代表更新论文版本。
