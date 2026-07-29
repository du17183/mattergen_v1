# 第5章实验与消融证据

## 正式三臂256

| 方法 | max force | 相对C0 | RMSD | E-hull | Stable | NUS |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 0.342964 | — | 0.049390 | 0.156136 | 44.531% | 22.266% |
| E3-A | 0.243956 | −28.87% | 0.045057 | 0.156179 | 44.531% | 22.266% |
| E3-G | 0.263107 | −23.28% | 0.045937 | 0.156177 | 44.531% | 22.266% |

E3-G: CI=[−0.144966,−0.032453] eV/Å；raw Wilcoxon Holm-corrected p=4.19e-10；raw W/T/L=163/0/93；algorithmic 1e-6 W/T/L=127/82/47。

## Learned Gate vs Always-on

| 指标 | E3-A | E3-G |
|---|---:|---:|
| Refinement rate | 100% | 66.406% |
| Harm | 25.391% | 18.359% |
| Low-force harm | 29.688% | 17.969% |
| Mean displacement | 0.010968 Å | 0.007580 Å |
| Mean force-gain retention | 100% | 80.657% |

McNemar p=.000534。Always-on平均降力更大；Gate价值是risk–coverage折中。

## Random Gate

frozen64使用5个随机seed，每次固定42/64开启，匹配Learned Gate frozen64覆盖率65.625%。随机相对降力范围−30.00%至−13.05%，均值−21.42%；Learned Gate frozen64为−33.56%。该结果是补充消融，不替代formal256。
