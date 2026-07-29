# 符号表

| 符号 | 含义 | 单位/范围 |
|---|---|---|
| \(s_k^{cond}\) | 字段 \(k\) 的条件 score | 模型字段单位 |
| \(s_k^{uncond}\) | 字段 \(k\) 的无条件 score | 模型字段单位 |
| \(r_k=s_k^{cond}-s_k^{uncond}\) | 三字段 CFG 残差 | 模型字段单位 |
| \(\delta_{t,k}=\operatorname{RMS}(r_{t,k})\) | 时刻 \(t\) 的字段残差 RMS | 非负 |
| \(\delta_t\) | 所有有效字段 RMS 的算术平均 | 非负 |
| \(m_{t,p}\) | predictor/corrector phase \(p\) 独立维护的 EMA | 非负 |
| \(q_t=\delta_t/(m_{t,p}+\epsilon)\) | 当前残差相对 phase 局部基准的比值 | 非负 |
| \(u_t\) | 自适应 multiplier | [0.25,4] |
| \(g_t\) | 三字段共同使用的最终 CFG scale | [0,5] |
| \(g_0\) | 基础 CFG scale | 2.0 |
| \(\alpha\) | 自适应更新强度 | 0.50 |
| \(\beta\) | phase-specific EMA 系数 | 0.95 |
| \(\epsilon\) | 自适应比值数值稳定项 | \(10^{-6}\) |
| \(c\) | Learned Gate confidence | [0,1] |
| \(\tau\) | Gate 阈值 | 0.5 |
| \(\eta\) | E3-PCR 位置更新步长 | 0.01 |
| \(R_{step}\) | 单步位移半径 | 0.02 Å |
| \(R_{cum}\) | 累计 trust-region 上限 | 0.10 Å |
| \(F_{max}\) | 结构预松弛最大原子力 | eV/Å |
| \(\Delta F\) | selected − baseline 最大力 | eV/Å；负值为改善 |
| \(E_{hull}\) | 凸包上方能量 | eV/atom |
| RMSD | 松弛前后结构均方根位移 | Å |

Adaptive CFG 先分别计算三个字段的 RMS，再聚合为一个控制统计量；当前实现不是三个字段
各自使用不同 guidance scale。

本文所有 CI 为 95% CI；“pp”表示百分点而不是相对百分比。

