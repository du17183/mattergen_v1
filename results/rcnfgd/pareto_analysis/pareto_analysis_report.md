# RC-NFGD Property–Force Pareto 分析

状态：COMPLETED_EXISTING_DATA；只重分析历史 A6、A1 结果，未重新训练或生成。

**定义**：横轴为冻结属性代理的磁密度 Property MAE；纵轴为终态 MatterSim MaxF，均越小越好。A6 三臂同为 seeds 745000–745031，n=32，逐 seed 配对；A1 两臂 n=256，单独分析，绝不与 A6 混成前沿。前沿是二维**方法均值点估计**，不是显著性判定，也不是逐结构选择器。

| Cohort | Method | n | Property MAE mean ± SD | MatterSim MaxF mean ± SD (eV/Å) | Point-estimate front |
|---|---|---:|---:|---:|---|
| A6 | MatterGen / C0 | 32 | 0.007900 ± 0.006690 | 0.214179 ± 0.239536 | no |
| A6 | RC-NFGD | 32 | 0.007886 ± 0.006537 | 0.163650 ± 0.230165 | yes |
| A6 | POST | 32 | 0.007933 ± 0.006363 | 0.043275 ± 0.138975 | yes |
| A1 | MatterGen / C0 | 256 | 0.009757 ± 0.011904 | 0.226408 ± 0.483886 | yes |
| A1 | RC-NFGD | 256 | 0.009868 ± 0.011940 | 0.158911 ± 0.340756 | yes |

方法均值的 20,000 次 seed bootstrap 95% CI 位于 `pareto_summary.csv`，同一批配对索引用于差值的 CI。以下差值定义为 **前者−后者**，正数表示后者更低；没有多重比较校正，解释为补充性分析。

| Cohort | Comparison | Metric | Mean difference | Paired bootstrap 95% CI |
|---|---|---|---:|---:|
| A6 | C0−F0 | Property MAE | 0.000014 | [-0.000095, 0.000167] |
| A6 | C0−F0 | MaxF | 0.050529 | [0.039821, 0.061093] |
| A6 | C0−F0 | Atomic Force | 0.026279 | [0.021160, 0.031362] |
| A6 | C0−POST | Property MAE | -0.000033 | [-0.000395, 0.000323] |
| A6 | C0−POST | MaxF | 0.170904 | [0.121238, 0.227766] |
| A6 | C0−POST | Atomic Force | 0.086852 | [0.063078, 0.114328] |
| A6 | F0−POST | Property MAE | -0.000048 | [-0.000349, 0.000212] |
| A6 | F0−POST | MaxF | 0.120375 | [0.076924, 0.171350] |
| A6 | F0−POST | Atomic Force | 0.060573 | [0.039603, 0.085314] |
| A1 | C0−F0 | Property MAE | -0.000111 | [-0.000236, -0.000023] |
| A1 | C0−F0 | MaxF | 0.067497 | [0.046658, 0.105063] |
| A1 | C0−F0 | Atomic Force | 0.029330 | [0.023285, 0.039599] |

## 解释与边界

A6 的 POST 在同一 MatterSim MaxF 上具有明显优势；不能声称 RC-NFGD 全面优于后处理。A6 方法均值上，RC-NFGD 与 POST 形成描述性折中，但极小的 MAE 差异必须结合配对 CI 看，不能把点估计前沿升格为确认性性能排序。A1 Formal256 的 Property MAE 方向与 A6 不完全一致，应按独立 cohort 如实保留。

这些力和属性指标都来自代理模型。MatterSim 是 RC-NFGD 所用势，存在 verifier/self-consistency 偏倚；已有 CHGNet 方向一致证据不等于独立 DFT 真值。此 Pareto 分析不能证明材料物理真实稳定性、全域最优性或联合创新点收益。

图件：`figures/rcnfgd/property_force_pareto.pdf` 为 A6 逐结构散点（大描边符号为均值）；`figures/rcnfgd/pareto_frontier.pdf` 为均值及 95% CI，虚线仅连接描述性非支配点。原始来源与逐结构复核可由 `pareto_per_structure.csv` 追溯。
