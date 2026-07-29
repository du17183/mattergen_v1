# Claim Traceability

## C1_ADAPTIVE_CFG_DIRECTIONAL

- 章节/节：第4章，4.8
- 冻结表述：在20000–20255的256个配对样本中，Adaptive CFG相对C0使代理E-hull降低0.003435 eV/atom、Stable提高5.859 pp、NUS提高3.516 pp；总体方向正向，但三项配对统计均未达到显著性。
- 数据：S10_I1_DATA
- 源码：S06_ADAPTIVE_CONTROLLER, S07_ADAPTIVE_CFG
- 图/表：图4-2；表4-2
- seeds/n：20000–20255；n=256
- 统计：E-hull CI [-0.017926,0.011030], raw p=.357; Stable CI [-1.5625,13.2813] pp, p=.146; NUS CI [-2.7344,9.7656] pp, p=.342.
- 限制：MatterSim-5M surrogate; non-significant paired inference; no target-property verification.
- 禁止变体：Adaptive CFG统计显著改善真实热力学稳定性。
- 证据完整：`True`


## C2_E3PCR_FORCE

- 章节/节：第5章，5.9
- 冻结表述：在40000–40255的独立256个样本中，E3-G把预松弛最大力均值从0.342964降至0.263107 eV/Å，相对下降23.28%，配对均值差95% CI为[-0.144966,-0.032453]，Holm校正p=4.19e-10。
- 数据：S17_I2_DATA
- 源码：S13_E3_REFINER, S14_E3_FROZEN_CORE, S15_E3_FORMAL_RUNNER
- 图/表：图5-2；表5-2
- seeds/n：40000–40255；n=256
- 统计：20,000 paired bootstrap; Wilcoxon Pratt; Holm family size 2; raw W/T/L=163/0/93.
- 限制：MatterSim-5M pre-relax force; not DFT or synthesizability evidence.
- 禁止变体：E3-PCR已通过DFT证明提升真实材料稳定性。
- 证据完整：`True`


## C3_GATE_HARM

- 章节/节：第5章，5.10
- 冻结表述：相对Always-on，Learned Gate把覆盖率从100%降至66.406%，harm从25.391%降至18.359%，低力子集harm从29.688%降至17.969%，并保留80.657%的平均降力收益；harm差异McNemar p=0.000534。
- 数据：S17_I2_DATA, S19_GATE_MECHANISM
- 源码：S14_E3_FROZEN_CORE, S15_E3_FORMAL_RUNNER
- 图/表：图5-3；表5-3
- seeds/n：40000–40255；n=256
- 统计：paired exact McNemar; E3-A-only harm=22, E3-G-only harm=4.
- 限制：Always-on平均降力更大；E3-G仍有47个算法语义harm样本。
- 禁止变体：Learned Gate平均降力优于Always-on，且保证每个样本安全。
- 证据完整：`True`


## C4_COMBINATION_COHORT1

- 章节/节：第6章，6.3
- 冻结表述：第一组独立组合cohort（41000–41063，n=64）中，A0+E3-G把预松弛最大力从0.217302降至0.158416 eV/Å，相对下降27.10%，95% CI为[-0.092341,-0.029754]，p=7.74e-5。
- 数据：S21_COHORT1_DATA
- 源码：S06_ADAPTIVE_CONTROLLER, S13_E3_REFINER
- 图/表：图6-2；表6-1
- seeds/n：41000–41063；n=64
- 统计：raw W/T/L=45/0/19; algorithmic 1e-6 W/T/L=34/19/11.
- 限制：独立64样本；不得替代E3-PCR正式256或与cohort 2事后合并。
- 禁止变体：预注册128-seed组合实验的前64个样本。
- 证据完整：`True`


## C5_COMBINATION_COHORT2

- 章节/节：第6章，6.4
- 冻结表述：第二组完全独立cohort（50000–50063，n=64）中，A0+E3-G把预松弛最大力从0.265280降至0.214830 eV/Å，相对下降19.02%，95% CI为[-0.102213,-0.010696]，p=0.000587；算法语义W/T/L=35/18/11。
- 数据：S23_COHORT2_DATA
- 源码：S06_ADAPTIVE_CONTROLLER, S13_E3_REFINER
- 图/表：图6-2、图6-3；表6-1
- seeds/n：50000–50063；n=64
- 统计：raw numeric W/T/L=46/0/18; Gate-off exact structure ties=18.
- 限制：效应小于cohort 1；必须并列报告异质性。
- 禁止变体：只报告更有利的cohort或把两组pooled为预注册128。
- 证据完整：`True`


## C6_LEAKAGE_SAFETY

- 章节/节：第6章，6.6
- 冻结表述：训练重叠没有明显夸大平均最大力改善，但显著高估Gate安全性：overlap harm=0/64，held-out harm=31/192=16.15%，单侧Fisher p=6.87e-5。
- 数据：S25_LEAK_DATA
- 源码：S15_E3_FORMAL_RUNNER
- 图/表：图6-4；表6-2
- seeds/n：20000–20255；n=256
- 统计：2x2 one-sided Fisher exact; held-out is supplementary only.
- 限制：Mixed 256和training overlap不得作为独立正式结果。
- 禁止变体：Mixed 256独立验证或匿名化seed后的独立验证。
- 证据完整：`True`


## C7_MATTERSIM_SURROGATE

- 章节/节：第3章，3.6
- 冻结表述：力、RMSD、E-hull、Stable与NUS均来自MatterSim-5M代理评价，可用于统一相对比较。
- 数据：S03_DATA_DICTIONARY, S05_LIMITATIONS
- 源码：S28_EVAL_ENERGY, S29_EVAL_RMSD
- 图/表：图3-3；表3-2
- seeds/n：all reported cohorts；n=跨实验边界
- 统计：not an effect claim
- 限制：STABILITY_SOURCE=MatterSim-5M surrogate; cannot replace DFT or synthesis evidence.
- 禁止变体：MatterSim评价即DFT验证。
- 证据完整：`True`


## C8_NO_DFT_OR_PROPERTY

- 章节/节：第3章，3.4 and 3.8
- 冻结表述：DFT_VERIFIED=False且PROPERTY_TARGET_VERIFIED=False；dft_mag_density=0.1是条件输入，不是已验证命中结果。
- 数据：S01_MANIFEST, S05_LIMITATIONS, S31_BASE_CONDITION_CONFIG
- 源码：不适用
- 图/表：图3-3；表3-2
- seeds/n：all reported cohorts；n=跨实验边界
- 统计：explicit negative evidence status
- 限制：不能证明真实磁性、热力学稳定性或可合成性。
- 禁止变体：生成结构真实达到目标磁密度并通过DFT验证。
- 证据完整：`True`


## P1_REGISTERED_TITLE_AND_SCOPE

- 章节/节：全篇；第1章与第3章首次明确。
- 冻结表述：学校正式中文题目为“基于深度学习的材料逆向生成”；本文实际研究对象为目标属性条件下的周期晶体候选生成。
- 数据：S33_TITLE, S34_POSITIONING
- 源码：不适用
- 图/表：图3-1；表3-1
- seeds/n：不适用
- 统计：not an effect claim
- 限制：英文题目仍为PROVISIONAL；题目范围宽于当前实验范围。
- 禁止变体：用内部方法题目替代学校登记中文题目。
- 证据完整：`True`

## P2_MATTERGEN_BASELINE_ATTRIBUTION

- 章节/节：第1章简述；第2章相关工作；第3章3.3正式定义。
- 冻结表述：MatterGen是本文采用的预训练条件晶体扩散生成基线、实验框架和实现载体，不是本文提出的方法。
- 数据：S34_POSITIONING, S35_NAMING_POLICY, S36_NUMBERING
- 源码：S08_PC_SAMPLER
- 图/表：图3-1、图3-2；表3-1
- seeds/n：不适用
- 统计：not an effect claim
- 限制：本文没有从零训练MatterGen主干；贡献限于Adaptive CFG和Learned-Gated E3-PCR。
- 禁止变体：本文提出或完整训练了MatterGen。
- 证据完整：`True`
