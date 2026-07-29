# 网页 ChatGPT 最终写作 Prompt：第3章第一部分（3.1—3.3）

你将为计算机专业学位论文撰写第3章的3.1—3.3节。必须只依据本Prompt给出的项目事实；可以使用材料生成与扩散模型的通用理论，但涉及外部研究的陈述必须标记`[待补参考文献]`，不得编造作者、论文名、年份或编号。

## 论文正式题目

```text
基于深度学习的材料逆向生成
```

该中文题目是学校登记题目，不得修改。暂定英文题目为`Deep Learning-Based Inverse Generation of Materials`，状态为`ENGLISH_TITLE_STATUS=PROVISIONAL`。

## 本章标题

```text
第3章 条件扩散晶体生成基线与评价体系
```

## 论文总体定位

- 论文中的“材料逆向生成”具体指：根据目标属性条件生成满足结构约束的周期晶体候选。
- 本文研究对象是周期晶体材料。
- 本文基础模型类别是预训练条件晶体扩散生成模型。
- 具体实验基线和实现载体是MatterGen。
- MatterGen不是本文提出的方法，本文也没有从零训练完整MatterGen主干。
- 本文贡献是推理阶段的Multi-field Residual-driven Online Adaptive CFG，以及生成后的Learned-Gated E3-PCR。
- 完整方法为A0生成后连接E3-G后处理。

## MatterGen命名规则

首次定义时使用：

> 本文采用预训练MatterGen作为条件晶体扩散生成基线。

说明实现后，优先使用“条件晶体扩散生成基线”或C0。不得把MatterGen已有晶体表示、扩散主干、Predictor–Corrector或原始CFG写成本文贡献，也不得隐藏MatterGen来源。

## 本次只写的章节

### 3.1 材料逆向生成任务定义

需要从材料逆向设计的一般目标过渡到本文的条件晶体候选生成任务，区分正向属性预测与逆向生成，明确输入是目标属性条件、输出是周期晶体候选，说明本文范围不覆盖真实合成闭环。

### 3.2 条件扩散晶体生成方法

用学术中文说明条件扩散生成的一般思想、晶体状态包含的组成/原子位置/晶格信息、条件信息如何影响反向生成，以及Predictor–Corrector与CFG在基线中的作用。通用理论需标`[待补参考文献]`；不要提前写第4章Adaptive CFG公式和实验。

### 3.3 MatterGen基线模型与实验配置

如实说明预训练MatterGen是实验基线和载体，介绍输入输出、三字段晶体表示、属性条件、完整Predictor–Corrector、原始constant CFG、冻结checkpoint以及本文两个扩展的接口位置。避免写成源码说明书。

## 项目事实

- C0：预训练dft_mag_density MatterGen条件晶体扩散生成基线。
- C0使用constant CFG scale=2.0、完整Predictor/Corrector、FP32、batch_size=1。
- A0=C0+Multi-field Residual-driven Online Adaptive CFG。
- E3-A=C0生成结构+Always-on E3-PCR。
- E3-G=C0生成结构+Learned-Gated E3-PCR。
- Full method=A0生成结构+Learned-Gated E3-PCR后处理。
- `dft_mag_density=0.1`是条件输入。
- MatterGen负责生成；MatterSim-5M负责统一代理评价；CHGNet只作为E3-PCR特征和局部更新代理。
- 当前仓库支持冻结配置、checkpoint身份、正式数据和实现commit，但精确原始生成CLI不是可移植归档的一部分。

## 指标定义

本次3.1—3.3不展开实验结果，但术语必须与后文一致：

- 预松弛最大力：`eV/Å`；
- RMSD：松弛前后结构位移，单位Å；
- E-hull：`eV/atom`；
- Stable：代理E-hull不高于0.1 eV/atom；
- NUS：Novel、Unique和Stable同时成立。

上述均属于MatterSim-5M代理评价，不是DFT真值。

## 实验数据资格

- Adaptive CFG正式实验：20000—20255，n=256。
- E3-PCR正式独立实验：40000—40255，n=256；与Gate训练20000—20063交集为0。
- 组合cohort 1：41000—41063，n=64。
- 组合cohort 2：50000—50063，n=64。
- Mixed 256只用于泄漏诊断，不是独立正式结果。

本次3.1—3.3只概括实验协议，不提前报告创新点效果数字。

## 统计方法

后文采用严格配对比较、paired bootstrap 95% CI、Wilcoxon Pratt以及适用的McNemar/Fisher检验。不得在3.1—3.3伪造新的统计口径。

## 允许使用的图表

- 图3-1：本文总体技术路线；
- 图3-2：条件晶体扩散生成与代理评价流程；
- 图3-3：实验数据与seed血缘；
- 表3-1：方法代号及定义；
- 表3-2：正式实验及数据划分。

图中基线模块写“条件晶体扩散生成基线”，图注注明该基线由预训练MatterGen实现。

## 允许使用的结论

- 本文针对目标属性条件下的周期晶体候选生成开展研究。
- 预训练MatterGen是本文的条件晶体扩散生成基线和实现载体。
- 本文创新位于推理控制和生成后质量优化，不位于MatterGen基础主干。
- MatterSim-5M可用于相同流程下的方法间代理相对比较。

## 禁止表述

- 本文提出MatterGen模型。
- 本文训练了完整MatterGen基础模型。
- 本文建立了dft_mag_density数据集。
- 本文已经验证生成材料真实达到目标磁密度。
- 本文完成了DFT稳定性验证。
- MatterSim或CHGNet等价于DFT。
- 代理Stable证明材料可合成。
- 把Mixed 256写成独立验证。
- 编造训练数据规模、主干参数量、生成命令或文献引用。

## 必须保留的局限性

```text
STABILITY_SOURCE=MatterSim-5M surrogate
DFT_VERIFIED=False
PROPERTY_TARGET_VERIFIED=False
```

题目范围宽于实验范围；正文必须明确本文实际研究对象为周期晶体，且没有完成DFT、实验合成或目标磁属性独立真值验证。

## 目标写作风格

- 使用客观、连贯的学术中文，符合计算机专业学位论文。
- 每节采用“问题背景—概念定义—本文落点—与下一节过渡”的逻辑。
- 不使用营销式表达，不把方向性趋势写成证明。
- 专有名词首次出现给出中英文或缩写，之后保持一致。
- 项目事实只使用本Prompt内容；通用理论中的外部事实标`[待补参考文献]`。
- 不在正文堆叠源码路径、commit或SHA256，可将其留给附录与证据包。

## 每节目标字数

- 3.1：800—1200字；
- 3.2：1200—1800字；
- 3.3：1500—2200字。

## 输出格式

1. 只输出`3.1`、`3.2`和`3.3`三节正文，不写3.4以后内容。
2. 保留三级标题仅在确有必要时使用。
3. 不输出写作说明、证据分析、致谢或参考文献表。
4. 需要外部来源的位置统一写`[待补参考文献]`。
5. 3.3结尾用一段话过渡到3.4“目标属性条件任务”，但不提前展开实验结果。
