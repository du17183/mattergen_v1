# Final Symbol Check

| Symbol family | Unified rule | Result |
|---|---|---|
| Diffusion state | \(x_t\)、\(x_0\)、\(\hat{x}_0(x_t,t)\) | PASS |
| MatterGen sampler state | \(z_t\) | PASS |
| Fields | \(a_t,p_t,H_t\)；实际分数坐标状态用 \(f_t\) | PASS |
| Cell matrix | 统一使用 \(H\)，行向量约定 \(r_i=f_iH\) | PASS |
| Scores | \(s_{\mathrm{cond}},s_{\mathrm{uncond}}\) | PASS |
| CFG | \(g\)，字段强度 \(g_a,g_p,g_H\) | PASS |
| Innovation 1 residuals | \(\Delta_a,\Delta_p,\Delta_H\) | PASS |
| Innovation 1 branch | \(T=1000\)、\(t_b=400\)、\(K=2\)、\(P_k\)、\(\xi_{t_b}\)、\(y_k\) | PASS |
| Force | \(F_i\)、\(\bar F\)、\(\widetilde F_i\)、\(F_{\mathrm{ref}}\) | PASS |
| Position correction | \(\delta f_i\)、\(\eta\)、\(\delta_{\max}\)、\(c_t^f\) | PASS |
| Relative effect | \(R_m\) | PASS |

本轮修正了第3章中通用字段残差的下标：由泛化的 \(\Delta_f\) 改为 \(\Delta_j,\ j\in\{a,p,H\}\)，避免与分数坐标修正和统一字段记号混淆。正文未发现使用晶胞 \(L\) 代替 \(H\) 的情况；统一符号表中的 \(\Delta_f\) 仅作为禁止误用的说明项保留。

SYMBOL_CONSISTENCY = PASS
