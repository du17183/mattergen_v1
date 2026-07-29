# 网页ChatGPT写作输入：第6章

你将撰写毕业论文第6章《组合验证、消融实验与结果讨论》。只能依据以下证据，不得补造项目事实；通用理论若需加入，必须标成待补参考文献，不能冒充项目实现。

## 项目术语

C0=原始条件晶体扩散生成基线，由预训练dft_mag_density MatterGen实现；A0=C0+Adaptive CFG；E3-A=C0生成结构+Always-on E3-PCR；E3-G=C0生成结构+Learned-Gated E3-PCR；完整方法=A0+E3-G。MatterGen是预训练基线而非本文贡献，MatterSim-5M是评价代理，CHGNet是E3-PCR辅助代理。

## 章节结构

- 6.1 两个创新点的功能分工
- 6.2 组合验证设计
- 6.3 独立兼容性实验一
- 6.4 独立复现实验二
- 6.5 Gate消融与风险分析
- 6.6 训练—测试泄漏诊断
- 6.7 代表性No-Go路线
- 6.8 计算开销
- 6.9 真实性与可复现性
- 6.10 局限性讨论

## 核心方法事实

- Adaptive CFG是完整组合方法的共享上游采样模块；E3-PCR是可接C0或A0的独立后生成模块。
- 两个64-seed cohort分别预留并独立报告；不得事后pool为单个预注册128。
- Gate-off为结构级exact fallback；评价数值可能有<1e-6微差，因此报告需区分raw numeric与algorithmic counts。
- 训练重叠诊断故意包含20000–20063；整个Mixed 256无独立资格。
- 代表性No-Go不是创新贡献，而是停止证据和方法边界。

## 公式

- F3_MAX_FORCE: $F_{\max}=\max_i\lVert\mathbf F_i\rVert_2$ (exact)
- F3_HARM: $\mathrm{Harm}=\mathbb 1[F_{\max}^{selected}-F_{\max}^{base}>10^{-6}]$ (exact)
- F5_GATE_RULE: $a=\mathbb 1[c\ge 0.5]$ (exact)
- F5_ACCEPTANCE: $\mathrm{accept}\iff \mathrm{finite\_safe}(X')\land E_{\mathrm{CHGNet}}(X')\le E_{\mathrm{CHGNet}}(X)+10^{-7}$ (exact)

## 参数

- cohort1=41000–41063
- cohort2=50000–50063
- harm epsilon=1e-6
- leak overlap=20000–20063
- leak held-out=20064–20255

## 实验结果

- Cohort 1最大力0.217302→0.158416，-27.10%；CI [-0.092341,-0.029754]；p=7.74e-5；raw 45/0/19，algorithmic 34/19/11。
- Cohort 2最大力0.265280→0.214830，-19.02%；CI [-0.102213,-0.010696]；p=.000587；algorithmic 35/18/11。
- 两组方向一致但效应大小不同，不能声称固定幅度。
- Leakage overlap harm=0/64，held-out=31/192，Fisher p=6.87e-5；安全性明显被高估。
- Corrector Gating约1.506×，但E-hull +0.0224、Stable -9.77 pp、NUS -9.38 pp。
- RP-QTFG离线方向正向但在线RMSD系统恶化，延迟约+30%–49%。
- CG-TDR Gate可学但Teacher residual方向未可靠泛化，收益接近零或RMSD恶化。

## 图表

图：图6-1, 图3-3, 图5-3, 图6-2, 图6-10, 图6-11, 图6-12。

表：表6-1, 表6-1, 表6-1, 表6-2, 表6-3。

## 允许结论

- 两个完全独立cohort均复现正向降力方向。
- 效应大小存在cohort异质性。
- 泄漏显著高估Gate安全性。
- No-Go可用于讨论假设边界。

冻结claim原句：

- 第一组独立组合cohort（41000–41063，n=64）中，A0+E3-G把预松弛最大力从0.217302降至0.158416 eV/Å，相对下降27.10%，95% CI为[-0.092341,-0.029754]，p=7.74e-5。
- 第二组完全独立cohort（50000–50063，n=64）中，A0+E3-G把预松弛最大力从0.265280降至0.214830 eV/Å，相对下降19.02%，95% CI为[-0.102213,-0.010696]，p=0.000587；算法语义W/T/L=35/18/11。
- 训练重叠没有明显夸大平均最大力改善，但显著高估Gate安全性：overlap harm=0/64，held-out harm=31/192=16.15%，单侧Fisher p=6.87e-5。

## 禁止结论

- 预注册128-seed pooled实验。
- 只报告cohort 1。
- Mixed 256独立验证。
- No-Go路线包装成正向创新。
- 创新点一是所有历史分支公共代码。

## 局限性

- 两个组合cohort各n=64。
- 同一数据域与MatterSim评价器。
- 部分历史No-Go原始报告只留服务器/历史分支，归档仅完整保留总结。
- 计算开销证据对E3-PCR组合以小样本/单环境为主，不宜外推部署成本。

STABILITY_SOURCE=MatterSim-5M surrogate；DFT_VERIFIED=False；PROPERTY_TARGET_VERIFIED=False。

## 数据来源标识

- S21_COHORT1_DATA
- S22_COHORT1_REPORT
- S23_COHORT2_DATA
- S24_COHORT2_REPORT
- S25_LEAK_DATA
- S26_LEAK_REPORT
- S32_NEGATIVE_RESULTS

## 正文风格

使用计算机专业学位论文的客观学术中文；先定义、再公式、再算法、再实验、再边界。所有效果注明baseline、seed、n、单位、统计口径和surrogate限制。非显著结果写“方向性趋势”，不写“证明无差异”。

## 目标字数

6000–9000字。当前任务只生成正文草稿；参考文献、学校模板编号和人工审阅标记保留待办。
