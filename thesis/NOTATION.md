# 符号表

| 符号 | 含义 | 单位/范围 |
|---|---|---|
| \(s_k^{cond}\) | 字段 \(k\) 的条件 score | 模型字段单位 |
| \(s_k^{uncond}\) | 字段 \(k\) 的无条件 score | 模型字段单位 |
| \(r_k=s_k^{cond}-s_k^{uncond}\) | 三字段 CFG 残差 | 模型字段单位 |
| \(\bar r_k\) | EMA 平滑后的残差统计 | 非负 |
| \(g_k\) | Adaptive CFG 字段尺度 | [0,5] |
| \(\alpha\) | 自适应更新强度 | 0.50 |
| \(\beta\) | EMA 系数 | 0.95 |
| \(c\) | Learned Gate confidence | [0,1] |
| \(\tau\) | Gate 阈值 | 0.5 |
| \(\eta\) | E3-PCR 位置更新步长 | 0.01 |
| \(R_{step}\) | 单步位移半径 | 0.02 Å |
| \(R_{cum}\) | 累计 trust-region 上限 | 0.10 Å |
| \(F_{max}\) | 结构预松弛最大原子力 | eV/Å |
| \(\Delta F\) | selected − baseline 最大力 | eV/Å；负值为改善 |
| \(E_{hull}\) | 凸包上方能量 | eV/atom |
| RMSD | 松弛前后结构均方根位移 | Å |

本文所有 CI 为 95% CI；“pp”表示百分点而不是相对百分比。

