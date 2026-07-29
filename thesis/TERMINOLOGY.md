# 统一术语（最终论文）

| 中文 | 英文 | 缩写/写法 | 使用规则 |
|---|---|---|---|
| 材料逆向生成 | Inverse generation of materials | — | 本文具体指根据目标属性条件生成周期晶体候选 |
| 预训练条件晶体扩散生成基线 | Pretrained conditional crystal diffusion baseline | MatterGen baseline | 由预训练 MatterGen 实现；不是本文提出的方法 |
| 原始条件晶体扩散生成基线 | Original conditional crystal diffusion baseline | C0 | 预训练 MatterGen；constant CFG scale=2.0；完整 Predictor/Corrector |
| 多字段残差驱动在线自适应 CFG | Multi-field Residual-driven Online Adaptive CFG | A0 / Adaptive CFG | A0=C0+创新点一 |
| 始终精修 | Always-on refinement | E3-A | 作为降力上限/安全消融，不是最终方法 |
| 学习门控 E3-PCR | Learned-Gated E3-PCR | E3-G | C0生成结构+创新点二 |
| 安全有界等变后生成晶体精修器 | Safe-Bounded Equivariant Post-Generation Crystal Refiner | E3-PCR | 只改位置，不改元素和晶胞 |
| 完整方法 | Full method | A0+E3-G | Adaptive CFG生成后连接Learned-Gated E3-PCR |
| 精确回退 | Exact fallback | — | Gate-off 或拒绝时返回输入结构 |
| 预松弛最大力 | Pre-relaxation maximum force | max force | 单位 eV/Å |
| 能量凸包上方能量 | Energy above hull | E-hull | 单位 eV/atom |
| 新颖、唯一且稳定 | Novel Unique Stable | NUS | MatterSim 代理稳定性 |
| 训练重叠 | Training overlap | — | Gate 训练 seed 与评估 seed 重合 |
| 独立组合 cohort | Independent combination cohort | Cohort 1/2 | 两组分别报告，不 pooled |
| 代理势 | Surrogate potential | MatterSim-5M | 不能等同 DFT/实验真值 |

“正式/独立/补充/诊断/无效”是证据资格，不是效果大小等级。

正文统一使用 **Learned-Gated E3-PCR**。`Q3`、`Q3 E3-PCR`、`Final refiner` 和 `Gate model`
只保留为历史实验代号，不作为最终论文方法名称。首次说明实现载体后，第4—6章优先使用
“条件晶体扩散生成基线”、C0、A0、E3-A、E3-G和完整方法，避免重复把MatterGen写成论文主题。

