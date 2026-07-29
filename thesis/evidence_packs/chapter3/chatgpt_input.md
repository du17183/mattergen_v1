# 网页ChatGPT写作输入：第3章

你将撰写毕业论文第3章《MatterGen基线、数据与评价体系》。只能依据以下证据，不得补造项目事实；通用理论若需加入，必须标成待补参考文献，不能冒充项目实现。

## 项目术语

C0=原始dft_mag_density MatterGen；A0=C0+Adaptive CFG；E3-A=Always-on E3-PCR；E3-G=Learned-Gated E3-PCR；完整方法=A0+E3-G。MatterGen是生成模型，MatterSim-5M是评价代理，CHGNet是E3-PCR辅助代理。

## 章节结构

- 3.1 条件晶体生成任务定义
- 3.2 MatterGen基线模型
- 3.3 dft_mag_density条件生成设置
- 3.4 实验数据与seed划分
- 3.5 MatterSim-5M代理评价流程
- 3.6 评价指标定义
- 3.7 配对统计方法
- 3.8 数据独立性与真实性控制

## 核心方法事实

- C0为原始dft_mag_density MatterGen，constant CFG scale=2.0；完整Predictor/Corrector、FP32、batch_size=1。
- A0=C0+Multi-field Residual-driven Online Adaptive CFG。
- E3-A/E3-G从同一个C0结构分别执行Always-on或Learned-Gated位置精修。
- MatterGen是生成模型，MatterSim-5M是论文评价代理，CHGNet只用于Gate特征与E3-PCR局部更新；三者不得混同。
- 条件目标为dft_mag_density=0.1，但PROPERTY_TARGET_VERIFIED=False。
- STABILITY_SOURCE=MatterSim-5M surrogate；DFT_VERIFIED=False。

## 公式

- F3_MAX_FORCE: $F_{\max}=\max_i\lVert\mathbf F_i\rVert_2$ (exact)
- F3_RMSD: $\operatorname{RMSD}=\operatorname{MatcherRMSD}(X_{\mathrm{relaxed}},X_{\mathrm{initial}})\;[\AA]$ (interpreted)
- F3_STABLE: $\mathrm{Stable}=\mathbb 1[E_{\mathrm{hull}}\le 0.1\;\mathrm{eV/atom}]$ (exact)
- F3_NUS: $\mathrm{NUS}=\mathrm{Novel}\land\mathrm{Unique}\land\mathrm{Stable}$ (exact)
- F3_HARM: $\mathrm{Harm}=\mathbb 1[F_{\max}^{selected}-F_{\max}^{base}>10^{-6}]$ (exact)

## 参数

- C0 guidance_scale=2.0
- Predictor/Corrector=full
- precision=FP32
- batch_size=1
- dft_mag_density target=0.1
- MatterSim stability threshold=0.1 eV/atom

## 实验结果

- 本章不主张方法效果；只冻结实验身份、评价和证据资格。

## 图表

图：Figure 4。

表：Table 01。

## 允许结论

- MatterSim用于相同流程下的方法间代理相对比较。
- 预松弛最大力、RMSD、E-hull、Stable和NUS可按统一代理口径报告。
- 两个64-seed cohort是两次独立证据。

冻结claim原句：

- 力、RMSD、E-hull、Stable与NUS均来自MatterSim-5M代理评价，可用于统一相对比较。
- DFT_VERIFIED=False且PROPERTY_TARGET_VERIFIED=False；dft_mag_density=0.1是条件输入，不是已验证命中结果。

## 禁止结论

- MatterSim等价于DFT。
- 条件输入0.1证明输出真实磁密度命中。
- 代理Stable证明可合成。
- 把Mixed 256或training overlap写成独立验证。

## 局限性

- 无DFT、无实验合成验证、无目标属性独立验证。
- 同一项目数据域、单一条件checkpoint和统一代理评价器。
- 精确原始生成CLI未作为可移植归档的一部分；可使用冻结配置和manifest，不得编造命令。

STABILITY_SOURCE=MatterSim-5M surrogate；DFT_VERIFIED=False；PROPERTY_TARGET_VERIFIED=False。

## 数据来源标识

- S01_MANIFEST
- S03_DATA_DICTIONARY
- S09_ADAPTIVE_CONFIG
- S10_I1_DATA
- S17_I2_DATA
- S21_COHORT1_DATA
- S23_COHORT2_DATA
- S25_LEAK_DATA

## 正文风格

使用计算机专业学位论文的客观学术中文；先定义、再公式、再算法、再实验、再边界。所有效果注明baseline、seed、n、单位、统计口径和surrogate限制。非显著结果写“方向性趋势”，不写“证明无差异”。

## 目标字数

6000–9000字。当前任务只生成正文草稿；参考文献、学校模板编号和人工审阅标记保留待办。
