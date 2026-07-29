# 网页ChatGPT写作输入：第5章

你将撰写毕业论文第5章《学习门控的晶体生成后质量优化方法》。只能依据以下证据，不得补造项目事实；通用理论若需加入，必须标成待补参考文献，不能冒充项目实现。

## 项目术语

C0=原始条件晶体扩散生成基线，由预训练dft_mag_density MatterGen实现；A0=C0+Adaptive CFG；E3-A=C0生成结构+Always-on E3-PCR；E3-G=C0生成结构+Learned-Gated E3-PCR；完整方法=A0+E3-G。MatterGen是预训练基线而非本文贡献，MatterSim-5M是评价代理，CHGNet是E3-PCR辅助代理。

## 章节结构

- 5.1 生成后局部物理不一致问题
- 5.2 方法总体框架
- 5.3 14维风险特征
- 5.4 129参数Learned Gate
- 5.5 等变位置更新
- 5.6 Trust region与位移限制
- 5.7 Backtracking与安全检查
- 5.8 Exact fallback
- 5.9 正式256-seed实验
- 5.10 Always-on与Random Gate消融
- 5.11 机制分析与局限

## 核心方法事实

- 14特征依次为num_atoms、volume_per_atom、mass_density、minimum_distance、atomic_number_mean/std、cell_condition、CHGNet energy/atom、force RMS/max/mean、stress RMS/maxabs和mag density。
- Gate为StandardScaler+MLPClassifier，14→8→1，tanh隐藏层，129个神经网络参数，阈值0.5。
- 训练样本为A0 seeds 20000–20063；标签为5步位置精修后最大力是否低于基线；8折OOF只作开发诊断。
- Gate只判断是否执行；E3-PCR用CHGNet force-vector方向执行最多5步；Fallback返回原始结构。
- 每步eta=.01、每原子0.02 Å cap，回溯scale为1、1/2、1/4；候选需finite、volume>0.1、min distance>=0.5 Å且CHGNet energy不升。
- 原子种类和晶格不变；最终wrapped累计最大位移检查<=0.10 Å；Gate-off和全拒绝exact fallback。
- 推理不训练MatterGen或CHGNet，且不改变原始MatterGen采样轨迹。

## 公式

- F3_HARM: $\mathrm{Harm}=\mathbb 1[F_{\max}^{selected}-F_{\max}^{base}>10^{-6}]$ (exact)
- F5_STANDARDIZE: $z_j=(x_j-\mu_j)/\sigma_j$ (interpreted)
- F5_GATE_NETWORK: $h=\tanh(W_1z+b_1),\qquad c=\sigma(W_2h+b_2)$ (interpreted)
- F5_PARAMETER_COUNT: $14\times8+8+8\times1+1=129$ (exact)
- F5_GATE_RULE: $a=\mathbb 1[c\ge 0.5]$ (exact)
- F5_POSITION_PROPOSAL: $\Delta x_i^{(b)}=\operatorname{clipnorm}(\eta\,2^{-b}F_i,\ R_{step}2^{-b})$ (exact)
- F5_ACCEPTANCE: $\mathrm{accept}\iff \mathrm{finite\_safe}(X')\land E_{\mathrm{CHGNet}}(X')\le E_{\mathrm{CHGNet}}(X)+10^{-7}$ (exact)
- F5_TRUST_BOUND: $\max_i\lVert x_i^{final}-x_i^{input}\rVert_{MIC}\le 5\times0.02=0.10\;\AA$ (interpreted)

## 参数

- features=14
- hidden=8
- output=1
- parameters=129
- threshold=.5
- steps=5
- eta=.01
- step cap=.02 Å
- cumulative cap=.10 Å
- backtracks=3
- min distance=.5 Å

## 实验结果

- E3-G最大力0.342964→0.263107 eV/Å，-23.28%；CI [-0.144966,-0.032453]；Holm p=4.19e-10；raw W/T/L=163/0/93。
- RMSD 0.049390→0.045937 Å；E-hull基本不变；Stable/NUS/Novel/Unique保持。
- E3-A平均最大力-28.87%，大于E3-G的-23.28%。
- E3-G coverage 66.406%，harm 18.359%，low-force harm 17.969%；E3-A分别100%、25.391%、29.688%。
- E3-G保留80.657% Always-on平均降力收益；harm McNemar p=.000534。
- Random Gate frozen64五次平均相对变化-21.42%，范围[-30.00%,-13.05%]；Learned Gate frozen64为-33.56%，但该比较不是formal256主结论。

## 图表

图：图5-1, 图5-2, 图5-3, 图5-4。

表：表5-2, 表5-3。

## 允许结论

- 独立formal256支持显著预松弛最大力下降。
- Learned Gate以较少覆盖降低总体和低力harm。
- 位置更新安全有界且元素/晶胞保持。

冻结claim原句：

- 在40000–40255的独立256个样本中，E3-G把预松弛最大力均值从0.342964降至0.263107 eV/Å，相对下降23.28%，配对均值差95% CI为[-0.144966,-0.032453]，Holm校正p=4.19e-10。
- 相对Always-on，Learned Gate把覆盖率从100%降至66.406%，harm从25.391%降至18.359%，低力子集harm从29.688%降至17.969%，并保留80.657%的平均降力收益；harm差异McNemar p=0.000534。

## 禁止结论

- Learned Gate平均降力优于Always-on。
- Gate保证所有结构改善。
- E3-PCR是完整晶格/组成松弛器。
- CHGNet输出是真实磁性或DFT验证。

## 局限性

- E3-G仍存在harm样本。
- Gate仅64个训练结构、129参数，对训练重叠敏感。
- 只更新位置，不能修复组成或晶格错误。
- CHGNet是辅助代理，正式评价仍为MatterSim。
- Random Gate来自frozen64而非formal256。

STABILITY_SOURCE=MatterSim-5M surrogate；DFT_VERIFIED=False；PROPERTY_TARGET_VERIFIED=False。

## 数据来源标识

- S16_E3_CONFIG
- S17_I2_DATA
- S18_I2_REPORT
- S19_GATE_MECHANISM
- S20_RANDOM_GATE

## 正文风格

使用计算机专业学位论文的客观学术中文；先定义、再公式、再算法、再实验、再边界。所有效果注明baseline、seed、n、单位、统计口径和surrogate限制。非显著结果写“方向性趋势”，不写“证明无差异”。

## 目标字数

7000–10000字。当前任务只生成正文草稿；参考文献、学校模板编号和人工审阅标记保留待办。
