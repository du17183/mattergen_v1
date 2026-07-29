# 网页 ChatGPT 写作 Prompt：第3章第三部分（3.7—3.9）

继续撰写学校登记题目《基于深度学习的材料逆向生成》的第3章《条件扩散晶体生成基线与评价体系》。本次只输出3.7—3.9。

## 3.7 评价指标与统计方法

使用以下冻结定义：

- \(F_{max}=\max_i\lVert\mathbf F_i\rVert_2\)，预松弛最大力，单位eV/Å；
- RMSD为松弛结构与初始结构的StructureMatcher位移，单位Å；该公式是解释性表达；
- Stable为代理\(E_{hull}\le0.1\) eV/atom；
- NUS=Novel ∧ Unique ∧ Stable；
- Harm为selected相对baseline的最大力差大于\(10^{-6}\) eV/Å。

还需说明composition validity、structure validity、Novel和Unique的评价作用，但不得编造当前证据包没有给出的实现公式。

统计方法：

- 同seed严格配对；
- paired bootstrap 95% CI；
- 连续配对指标使用Wilcoxon Pratt；
- 成对二元差异使用McNemar或精确检验；
- 泄漏harm率使用单侧Fisher exact；
- Win/Tie/Loss必须注明raw continuous或\(10^{-6}\)算法语义口径；
- Adaptive CFG总体方向正向，但配对统计未达到显著性。

## 3.8 数据独立性和真实性控制

说明冻结manifest、逐seed CSV、源码commit、配置和报告之间的追踪关系。必须强调：

- E3-PCR formal、cohort 1和cohort 2与Gate训练seeds交集为0；
- training overlap和Mixed 256只作泄漏诊断；
- 两个64-seed cohort分别报告，不事后pooled；
- 稳定性来源是MatterSim-5M surrogate；
- DFT、目标属性独立验证和实验合成都未完成；
- 不匿名化或改写seed掩盖数据来源。

## 3.9 本章小结

总结材料逆向生成任务、条件晶体扩散基线、MatterGen归属、目标条件、评价和真实性控制，并自然引出第4章“多字段残差驱动的在线自适应条件引导方法”。不要提前详细报告第4章结果。

## 允许使用的图表

- 表3-3：评价指标；
- 表3-4：统计检验方法；
- 图3-3：实验数据与seed血缘。

## 写作边界

```text
STABILITY_SOURCE=MatterSim-5M surrogate
DFT_VERIFIED=False
PROPERTY_TARGET_VERIFIED=False
```

不得把MatterGen写成本研究提出的方法，不得把MatterSim/CHGNet写成DFT，不得把条件0.1写成真实属性命中，不得把Mixed 256写成独立验证，不得补造项目事实或文献。

## 写作风格与字数

- 3.7：1400—2000字；
- 3.8：1000—1500字；
- 3.9：400—600字。

使用客观学术中文。通用理论需引用的位置标`[待补参考文献]`。只输出3.7—3.9正文，不输出写作说明或参考文献表。
