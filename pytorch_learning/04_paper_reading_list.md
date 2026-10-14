# 静态场景图论文：按学习顺序阅读

更新日期：2026 年 10 月 14 日。按理解模型所需的前置知识排序，接入每天 5 小时的学习安排。先完成 Python 和 PyTorch 的基础验收，再深入论文公式与代码；基础阶段也可以先浏览综述、摘要和模型图。

## 总顺序

| 顺序 | 论文 | 本轮要理解的问题 |
| --- | --- | --- |
| 0：导读 | [Scene Graph Generation: A comprehensive survey](https://research-repository.uwa.edu.au/en/publications/scene-graph-generation-a-comprehensive-survey/)，Neurocomputing 2024 | 场景图是什么，输入输出和评价条件怎样定义 |
| 1：注意力基础 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762)，2017 | Q/K/V、注意力、多头、残差、位置编码与 encoder/decoder |
| 2：检测基础 | [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)，2020，DETR | 图像检测怎样使用 query、集合预测与匹配 |
| 3：关系预测 | [RelTR: Relation Transformer for Scene Graph Generation](https://arxiv.org/abs/2201.11460)，TPAMI 2023 | 检测怎样扩展为主语、谓词、宾语的联合预测 |
| 4：主要基线阅读 | [EGTR: Extracting Graph from Transformer for Scene Graph Generation](https://arxiv.org/abs/2404.02072)，CVPR 2024 | 注意力交互特征怎样得到关系，连通性怎样参与训练 |
| 5：全局语义对照 | [Relationformer: A Unified Framework for Image-to-Graph Generation](https://arxiv.org/abs/2203.10202)，ECCV 2022 | relation token 怎样提供局部与全局关系信息 |
| 6：图信息传递 | [Bipartite Graph Network with Adaptive Message Passing for Unbiased Scene Graph Generation](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Bipartite_Graph_Network_With_Adaptive_Message_Passing_for_Unbiased_Scene_CVPR_2021_paper.pdf)，CVPR 2021，BGNN | 哪些实体—关系连接适合传播，怎样控制信息贡献 |
| 7：长尾校正 | [Long-tail learning via logit adjustment](https://research.google/pubs/long-tail-learning-via-logit-adjustment/)，ICLR 2021 | 类别先验怎样影响分类得分，训练与后处理调整怎样区分 |
| 8：表示改进 | [Prototype-Based Embedding Network for Scene Graph Generation](https://openaccess.thecvf.com/content/CVPR2023/html/Zheng_Prototype-Based_Embedding_Network_for_Scene_Graph_Generation_CVPR_2023_paper.html)，CVPR 2023，PE-Net | 关系类别内部变化、类间混淆能否通过原型表示改善 |
| 9：进阶选读 | [Hydra-SGG: Hybrid Relation Assignment for One-stage Scene Graph Generation](https://proceedings.iclr.cc/paper_files/paper/2025/hash/10831838f1cdd8fcba90188e88e7a121-Abstract-Conference.html)，ICLR 2025 | 匹配与正负监督的分配会怎样影响训练 |

Attention Is All You Need 是新增的注意力前置阅读。它原本研究序列任务，本轮重点学习可迁移到视觉 Transformer 的机制，机器翻译实验暂按需查阅。

## 第一阶段：任务、训练与注意力

对应主线第 1～2 周。Python 先修的 5 天另按能力检查安排。

### 0. 综述：先建立任务地图

先读任务定义、单阶段/两阶段分类、数据集、评价与长尾问题，再按具体疑问查方法章节。

完成标准：

- 用一个例子写出有方向的主语—谓词—宾语三元组。
- 解释 PredCls、SGCls、SGDet 分别给了模型什么已知信息。
- 列出 R@K 与 mR@K 的区别，以及比较模型时要固定的评价条件。

### 1. Attention Is All You Need：读通一次注意力

开读条件：能做小矩阵乘法，解释 softmax 和线性层的输入输出。先看注意力与模型结构，给公式中的 Q、K、V 和输出标形状，再读多头、残差、归一化与位置编码。

完成标准：手算两三个 token 的单头注意力，用小程序核对加权求和；解释 self-attention 与 cross-attention 的输入来源。完成后再进入 DETR。

## 第二阶段：检测到静态场景图

对应主线第 3 周，并在第 4 周结合数据和评价代码回读。

### 2. DETR：查询、匹配与检测损失

先看模型图，再找图像特征、object query、类别/框输出和集合匹配。核对匹配代价与训练损失各自使用什么输入。

完成标准：画出图片到预测集合的流程，用 3 个预测和 2 个标注做一次匹配例子，说明 query 数量与真实物体数量的区别。

作者代码：[facebookresearch/detr](https://github.com/facebookresearch/detr)。

### 3. RelTR：把物体检测扩展为关系三元组

对照 DETR，看主宾语查询、耦合注意力、三元组输出和匹配监督。重点问“新增了什么预测，哪部分特征支撑谓词判断”。

完成标准：写一页 DETR/RelTR 对照表；追踪一条预测三元组的主语、宾语、谓词得分及训练标签。

作者代码：[yrcong/RelTR](https://github.com/yrcong/RelTR)。

### 4. EGTR：作为主要代码阅读对象

先看关系特征来源，再看关系头、连通性预测和 relation smoothing。结合前向、损失与评价入口读一条完整路径。

完成标准：给主要变量标形状，说明谓词预测与连通性预测的不同任务，并记录单图推理或明确的运行卡点。主线优先跑通一个基线。

作者代码：[naver-ai/egtr](https://github.com/naver-ai/egtr)。将 EGTR 作为学习对象，不认定开题中 Ours 的底座就是 EGTR；取得实际源码后再确认。

## 第三阶段：研究多路径与图传递

对应主线第 5～6 周，前提是已能解释主要基线的数据流与评价条件。

### 5. Relationformer：全局上下文路径的对照

重点阅读 object token、relation token 和对象对关系预测的连接方式，检查接口是否适合当前基线。

完成标准：画出 relation token 参与关系推断的位置；说明它提供的信息如何与实体外观或交互特征互补。先固定特征路径，再比较固定融合和自适应融合。

作者代码：[suprosanna/relationformer](https://github.com/suprosanna/relationformer)，场景图实现位于 `scene_graph` 分支。

### 6. BGNN：先定义邻居，再讨论聚合

重点阅读二部图、连接可靠性与自适应传播。另行记录数据重采样机制，避免把表征改进和采样收益混在一起。

完成标准：用三个物体说明一次传播的来源、方向、权重和更新对象。将它与开题中的共享主宾语关系邻居比较，明确结构差别。

[作者入口](https://github.com/Scarecrow0/BGNN-SGG)将实现指向 [PySGG](https://github.com/SHTUPLUS/PySGG)。

## 第四阶段：长尾与关系表示

对应主线第 7 周；开始前需要能核对 R/mR 和类别频次。

### 7. Logit Adjustment：从简单例子理解先验校正

先区分训练损失调整与训练后调整，核对公式、先验统计和调整强度，用三个类别的 logits 观察排序变化。

完成标准：手算一个校正例子；说明先验从哪里统计、参数在哪个数据划分上选择，并解释整体召回与类别平均召回的取舍。

开题材料未给出 LA 具体公式；这篇是相关机制的学习资料，实际对应关系需从代码确认。[论文与作者信息](https://research.google/pubs/long-tail-learning-via-logit-adjustment/)

### 8. PE-Net：把表示学习与得分校正分开

重点读原型对齐、Prototype-guided Learning 与 Prototype Regularization，理解谓词类别内部视觉变化和类别间相似造成的困难。

完成标准：说明原型表示、门控融合与 LA 分别改动哪个环节；设计独立对照，区分各自贡献。

作者代码：[VL-Group/PENET](https://github.com/VL-Group/PENET)。

## 第五阶段：进阶选读与实验设计

对应主线第 8 周或后续扩展。

### 9. Hydra-SGG：回头检查监督分配

在集合匹配和关系查询已经理解后，阅读一对一/一对多分配、假负样本问题与辅助分支。

完成标准：说明一个真实关系监督了哪些查询，哪些候选被当作负例，并区分训练和推理时的分支使用。

2026 年 10 月 14 日查阅的 [作者仓库](https://github.com/Dreamer312/Hydra-SGG)仍标注完整代码、文档与权重待发布。本轮安排论文阅读，代码发布后再决定是否复现。

## 每篇的阅读步骤与笔记

论文阅读放在每天约 1 小时的资料阅读时段；基础阶段可用其中 30～60 分钟看任务与模型图。每篇分多次读，遇到前置知识卡点先补对应练习。

1. 第一遍：摘要、问题、模型图与结论，标出输入输出。
2. 第二遍：方法公式与关键张量，手算一个小例子。
3. 第三遍：对应源码、损失、后处理和评价条件。
4. 第四遍：消融与失败案例，检查证据是否支持方法结论。

每篇完成一页笔记：

```text
Paper / Year:
Problem: 要解决什么具体问题？
Inputs / Outputs: 数据、形状、标签和监督来源？
Method: 相对基线改动了哪个环节？
Evidence: 哪项消融支持这个改动？
Connection: 与多路径、图传递或长尾课题的联系？
Questions: 还不能解释的公式、代码和协议？
```

模型比较时固定数据划分、任务设置、检测器、graph constraint、K 与训练预算。读书笔记、单图推理、子集训练和完整标准实验分别记录。源码环境按作者依赖单独核对，不直接套用 CPU 教学环境。

返回 [学习总览](00_overview.md) 或 [完整学习方案](02_scene_graph_study_plan.md)。
