# 静态场景图生成论文阅读清单

核查日期：2026 年 10 月 13 日。按当前课题和学习阶段选择，优先读通一个静态基线，再对照多路径、图信息传递和长尾机制。

## 阅读顺序

先读综述的任务与评价部分，随后依次阅读 DETR、RelTR、EGTR。掌握主要计算路径后，按实现问题阅读 Relationformer、BGNN、Logit Adjustment 与 PE-Net。Hydra-SGG 作为进阶选读。

| 顺序 | 论文与发表信息 | 本轮重点 | 作者代码或入口 |
| --- | --- | --- | --- |
| 导读 | [Scene Graph Generation: A comprehensive survey](https://research-repository.uwa.edu.au/en/publications/scene-graph-generation-a-comprehensive-survey/)，Neurocomputing 2024 | 场景图定义、方法分类、任务设置、评价与长尾问题 | 综述用于建立知识地图 |
| 1 | [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)，2020，DETR | object query、集合预测、匹配与检测损失 | [facebookresearch/detr](https://github.com/facebookresearch/detr) |
| 2 | [RelTR: Relation Transformer for Scene Graph Generation](https://arxiv.org/abs/2201.11460)，TPAMI 2023 | 主宾语查询、三元组预测、端到端训练 | [yrcong/RelTR](https://github.com/yrcong/RelTR) |
| 3 | [EGTR: Extracting Graph from Transformer for Scene Graph Generation](https://arxiv.org/abs/2404.02072)，CVPR 2024 | 检测 Transformer 的交互特征、关系提取、连通性与平滑监督 | [naver-ai/egtr](https://github.com/naver-ai/egtr) |
| 4 | [Relationformer: A Unified Framework for Image-to-Graph Generation](https://arxiv.org/abs/2203.10202)，ECCV 2022 | object token 与 relation token、局部和全局语义 | [suprosanna/relationformer](https://github.com/suprosanna/relationformer)，场景图代码在 `scene_graph` 分支 |
| 5 | [Bipartite Graph Network with Adaptive Message Passing for Unbiased Scene Graph Generation](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Bipartite_Graph_Network_With_Adaptive_Message_Passing_for_Unbiased_Scene_CVPR_2021_paper.pdf)，CVPR 2021，BGNN | 实体和关系间的信息传递、连接可靠性、自适应传播 | [作者入口](https://github.com/Scarecrow0/BGNN-SGG)将源码指向 [PySGG](https://github.com/SHTUPLUS/PySGG) |
| 6 | [Long-tail learning via logit adjustment](https://research.google/pubs/long-tail-learning-via-logit-adjustment/)，ICLR 2021 | 类别先验、训练或训练后的得分调整 | 通用长尾方法；先用三个类别做手算 |
| 7 | [Prototype-Based Embedding Network for Scene Graph Generation](https://openaccess.thecvf.com/content/CVPR2023/html/Zheng_Prototype-Based_Embedding_Network_for_Scene_Graph_Generation_CVPR_2023_paper.html)，CVPR 2023，PE-Net | 谓词类别内变化、类间相似、原型对齐与正则 | [VL-Group/PENET](https://github.com/VL-Group/PENET) |
| 选读 | [Hydra-SGG: Hybrid Relation Assignment for One-stage Scene Graph Generation](https://proceedings.iclr.cc/paper_files/paper/2025/hash/10831838f1cdd8fcba90188e88e7a121-Abstract-Conference.html)，ICLR 2025 | 一对一与一对多关系分配、稀疏监督和假负样本 | [作者仓库](https://github.com/Dreamer312/Hydra-SGG)当前标注完整代码与权重待发布，先阅读论文 |

## 每篇要回答的问题

### 1. DETR：查询和匹配怎么工作

先看模型图，区分图像特征、object query 和最后的类别/框预测。用少量预测与标注构造匹配例子，再找分类与框损失的输入。

验收：画出从图片到预测集合的路径；说明 query 数量与真实物体数量为什么可以不同。

### 2. RelTR：怎样把检测扩展到关系预测

对照 DETR，观察耦合的主宾语查询、注意力机制和三元组集合预测。记录检测与关系预测如何进入训练监督。

验收：一页 DETR/RelTR 对照表，标出新增输出、关系特征和匹配/损失的区别。

### 3. EGTR：关系特征从哪里来

重点读注意力交互信息的提取、关系头、连通性预测和 relation smoothing。它直接对应课题中注意力路径与连通性的学习需求，但是否作为 Ours 的实际底座仍需源码确认。

验收：沿一张图片追踪关系特征的来源，分别解释谓词预测和“有无关系”的连通性预测。

### 4. Relationformer：全局关系表示提供什么

阅读 object token 和 relation token 的设计及其交互。将其作为全局上下文路径的对照，核对实际输入输出接口。

验收：说明 relation token 在哪个位置参与对象对的关系推断；不能仅以“用了 Transformer”代替解释。

### 5. BGNN：哪些邻居值得传递信息

重点读二部图、连接可靠性与自适应信息传递，并区分其表征机制和数据重采样机制。该方法的实体—关系传播结构与 PPT 的共享主宾语关系聚合有联系，节点定义和传播对象需要单独核对。

验收：构造三个物体的图，说明一次消息传递从谁到谁、权重怎样控制贡献、更新了哪个特征。

### 6. Logit Adjustment：类别先验怎样改变预测

区分训练损失中的调整与训练后的调整，记录公式、先验来源和调整强度。用三个类别的得分观察排序变化，并分析 R 与 mR 的取舍。

验收：手算一个调整例子，说明头部与尾部类别各受什么影响。PPT 未展开 LA 的具体公式；这篇是相关学习资料，实际对应关系仍需代码确认。

### 7. PE-Net：怎样改善关系表示的混淆

重点看原型对齐、Prototype-guided Learning 和 Prototype Regularization。该工作讨论谓词类别内部的视觉变化与类别间相似，可以帮助细化开题中的表征问题。

验收：区分改善特征表示、改变融合权重和校正分类得分分别发生在哪里；不要把它们的收益混为一项。

### 8. Hydra-SGG：匹配规则会不会限制监督

在集合匹配和 RelTR 类关系查询机制已经理解后，阅读一对一/一对多分配和 Hydra 辅助分支。它提供监督分配方面的对照，论文阅读优先于当前尚未发布完整代码的复现。

验收：说明一个真实关系监督了几个查询，哪些候选会被当作负例，以及训练和推理时的分支使用。

## 和现有课程怎样结合

- Python/PyTorch 基础阶段：每次用 30～60 分钟读摘要、模型图、任务定义和结论，记录术语对应的数据或代码位置。
- 注意力与静态模型阶段：深入 DETR、RelTR、EGTR 的公式、输入输出形状与源码。首先跑通一个基线。
- 改进阶段：按实际改动阅读 Relationformer、BGNN、Logit Adjustment 与 PE-Net，并设计单独对照。
- 实验阶段：核对数据划分、PredCls/SGCls/SGDet、检测器、graph constraint、K、训练预算和 R/mR。不同协议的数值不能直接排序。

论文源码环境按作者说明单独准备。现有 PyTorch 2.6 的 CPU 教学环境验证了基础练习；真实研究仓库还需核对依赖、编译组件、数据和显卡条件。DETR 作者仓库已归档，仍可用于经典架构阅读。[仓库状态](https://github.com/facebookresearch/detr)

## 每篇的一页笔记

```text
Paper / Year:
Problem: 具体要解决什么问题？
Inputs / Outputs: 关键数据、形状与监督是什么？
Method: 相对基线改了哪个环节？
Evidence: 哪项消融或结果支持这个改动？
Connection: 与多路径、图传递或长尾问题怎样连接？
Questions: 哪些公式、代码或协议仍待确认？
```

阅读完成不等于复现完成。读书笔记、模型推理、子集训练与标准实验分别记录。
