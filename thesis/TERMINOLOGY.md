# 统一术语

| 中文 | 英文 | 缩写/写法 | 使用规则 |
|---|---|---|---|
| 原始条件 MatterGen | Original conditional MatterGen | C0 | 创新点二独立实验基线 |
| 多字段残差驱动在线自适应 CFG | Multi-field Residual-driven Online Adaptive CFG | A0 / Adaptive CFG | A0 是启用创新点一后的方法 |
| 始终精修 | Always-on refinement | E3-A | 作为降力上限/安全消融，不是最终方法 |
| 学习门控 E3-PCR | Learned-Gated E3-PCR | E3-G | 创新点二最终方法 |
| 安全有界等变后生成晶体精修器 | Safe-Bounded Equivariant Post-Generation Crystal Refiner | E3-PCR | 只改位置，不改元素和晶胞 |
| 精确回退 | Exact fallback | — | Gate-off 或拒绝时返回输入结构 |
| 预松弛最大力 | Pre-relaxation maximum force | max force | 单位 eV/Å |
| 能量凸包上方能量 | Energy above hull | E-hull | 单位 eV/atom |
| 新颖、唯一且稳定 | Novel Unique Stable | NUS | MatterSim 代理稳定性 |
| 训练重叠 | Training overlap | — | Gate 训练 seed 与评估 seed 重合 |
| 独立组合 cohort | Independent combination cohort | Cohort 1/2 | 两组分别报告，不 pooled |
| 代理势 | Surrogate potential | MatterSim-5M | 不能等同 DFT/实验真值 |

“正式/独立/补充/诊断/无效”是证据资格，不是效果大小等级。

