# 网页ChatGPT写作输入：第4章

你将撰写毕业论文第4章《多字段残差驱动的在线自适应条件引导方法》。只能依据以下证据，不得补造项目事实；通用理论若需加入，必须标成待补参考文献，不能冒充项目实现。

## 项目术语

C0=原始条件晶体扩散生成基线，由预训练dft_mag_density MatterGen实现；A0=C0+Adaptive CFG；E3-A=C0生成结构+Always-on E3-PCR；E3-G=C0生成结构+Learned-Gated E3-PCR；完整方法=A0+E3-G。MatterGen是预训练基线而非本文贡献，MatterSim-5M是评价代理，CHGNet是E3-PCR辅助代理。

## 章节结构

- 4.1 固定CFG的局限
- 4.2 条件与无条件分支
- 4.3 三字段残差定义
- 4.4 EMA残差状态
- 4.5 在线Guidance更新
- 4.6 完整算法流程
- 4.7 计算开销
- 4.8 正式实验结果
- 4.9 讨论与限制

## 核心方法事实

- conditional与unconditional输入先collate为一次joint model forward，再拆分score。
- cell、pos、atomic_numbers残差分别计算RMS，只有在标量化后才求算术平均。
- predictor和corrector各自维护EMA；首个观测直接初始化EMA。
- 当前实现产生一个全局guidance scale，三个字段共享，不是三套独立scale。
- invalid residual/EMA触发stage guidance fallback。
- Adaptive CFG不启用cfg acceleration或Corrector Gating；完整corrector和predictor流程保留。
- 控制器增加三字段RMS归约和常数级标量运算，但不减少或增加MatterGen模型forward次数。

## 公式

- F4_RESIDUAL: $r_{t,k}=s^{cond}_{t,k}-s^{uncond}_{t,k}$ (exact)
- F4_FIELD_RMS: $\delta_{t,k}=\sqrt{\operatorname{mean}(r_{t,k}^{\,2})}$ (exact)
- F4_FIELD_MEAN: $\delta_t=\frac{1}{|\mathcal K_t|}\sum_{k\in\mathcal K_t}\delta_{t,k}$ (exact)
- F4_EMA: $m_{t,p}=\begin{cases}\delta_t,&m_{t-1,p}\ \mathrm{unset}\\ \beta m_{t-1,p}+(1-\beta)\delta_t,&\mathrm{otherwise}\end{cases}$ (exact)
- F4_RATIO: $q_t=\frac{\delta_t}{m_{t,p}+\epsilon}$ (exact)
- F4_MULTIPLIER: $u_t=\operatorname{clip}\!\left(1+\alpha(q_t-1),0.25,4\right)$ (exact)
- F4_GUIDANCE: $g_t=\operatorname{clip}(g_0u_t,g_{\min},g_{\max})$ (exact)
- F4_CFG_FUSION: $s_t^{CFG}=s_t^{uncond}+g_t(s_t^{cond}-s_t^{uncond})$ (exact)

## 参数

- g0=2.0
- alpha=0.50
- beta=0.95
- epsilon=1e-6
- multiplier clip=[0.25,4]
- guidance clip=[0,5]

## 实验结果

- E-hull C0=0.143667，A0=0.140232，差=-0.003435 eV/atom；CI跨0，p=.357。
- Stable C0=41.016%，A0=46.875%，差=+5.859 pp；CI跨0，p=.146。
- NUS C0=22.266%，A0=25.781%，差=+3.516 pp；CI跨0，p=.342。
- 三项方向均正向，但均不得称统计显著。

## 图表

图：图4-1, 图4-2。

表：表4-2。

## 允许结论

- 多字段在线反馈使三项代理指标呈总体正向趋势。
- 算法保留完整Predictor/Corrector。
- 控制器公式可标为代码精确等价。

冻结claim原句：

- 在20000–20255的256个配对样本中，Adaptive CFG相对C0使代理E-hull降低0.003435 eV/atom、Stable提高5.859 pp、NUS提高3.516 pp；总体方向正向，但三项配对统计均未达到显著性。

## 禁止结论

- Adaptive CFG统计显著提升。
- 该方法通过跳步或Corrector Gating加速。
- 三个字段各使用独立guidance scale。
- 代理结果证明真实磁稳定性。

## 局限性

- 配对统计未显著。
- 单checkpoint、单目标和单采样配置。
- 没有独立属性命中验证或DFT。
- 精确控制开销没有作为正式主效果冻结。

STABILITY_SOURCE=MatterSim-5M surrogate；DFT_VERIFIED=False；PROPERTY_TARGET_VERIFIED=False。

## 数据来源标识

- S09_ADAPTIVE_CONFIG
- S10_I1_DATA
- S11_I1_REPORT

## 正文风格

使用计算机专业学位论文的客观学术中文；先定义、再公式、再算法、再实验、再边界。所有效果注明baseline、seed、n、单位、统计口径和surrogate限制。非显著结果写“方向性趋势”，不写“证明无差异”。

## 目标字数

6000–9000字。当前任务只生成正文草稿；参考文献、学校模板编号和人工审阅标记保留待办。
