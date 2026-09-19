# Final Master Numbers

本表是 Chapter 1–6 可进入摘要、正文、答辩材料的主结果入口。数值来自 thesis/cross_chapter/master_numbers.md、master_numbers.csv 和 thesis_release/combined_summary/main_results.csv；本审计不重新计算任何结果。

## 创新点1：Fixed-K2

| 项目 | 冻结值 | 解释 |
|---|---:|---|
| Cohort | C1-128 | 128 个全新配对种子 |
| C0 Property MAE | 0.034796 | 越低越好 |
| Fixed-K2 Property MAE | 0.026332 | 越低越好 |
| 相对降低 | 24.32% | 支持代理属性改善 |
| 绝对改善 | 0.008464 | C0 − Fixed-K2 |
| 95% bootstrap CI | [0.005817, 0.011391] | 20,000 次配对重采样，区间全为正 |
| W/T/L | 53/75/0 | 逐样本改善/持平/损失 |
| E-hull | 0.104296 → 0.081135 | 冻结代理护栏 |
| Stable | 58.59% → 70.31% | 冻结代理护栏 |
| NUS | 32.81% → 37.50% | 冻结代理护栏 |
| Validity | 100% → 100% | 保持 |
| Novel | 71.09% → 65.63% | 观察到的单项下降，必须保留 |
| Linear-K2 | 0.027522 | 未优于 Fixed-K2 |
| Random-K2 | 0.028533 | 描述性比较 |
| Linear vs Fixed | −4.52%，CI [−0.003148, 0.000740]，31/62/35 | NOT_SUPPORTED |
| 部署成本 | 4,400 vs 2,000 score calls，约 2.2× | Phase B 约 3.4× 是离线数据构建成本 |

## 创新点2：RC-NFGD

| 项目 | C0 → RC-NFGD | 解释 |
|---|---:|---|
| Cohort | Formal256 | 256 个全新配对种子 |
| MatterSim MaxF | 0.226408 → 0.158911 | 降低 29.81%，CI [21.42%, 40.88%]，254/0/2 |
| MatterSim Mean Force | 0.098702 → 0.069373 | 降低 29.72%，CI [24.01%, 36.96%]，255/0/1 |
| RMSD | 0.051137 → 0.043893 | 降低 14.17%，CI [4.54%, 25.95%]，216/21/19 |
| Property MAE | 0.009757 → 0.009868 | 误差恶化约 1.14%，不能写成改善 |
| Stable | 80.86% → 80.86% | 保持 |
| Validity | 100% → 100% | 保持 |
| CHGNet MaxF | 0.19389 → 0.16606 | 降低 14.36%，独立代理均值方向 |
| CHGNet Mean Force | 0.08798 → 0.07954 | 降低 9.59%，独立代理均值方向 |
| 阶段可靠性 | \(t=0.02\)，Spearman=0.925587 | 机制与窗口校准，不是因果最优证明 |
| MatterGen score calls | 2,000 → 2,000 | 不增加主网络调用 |
| 在线 MatterSim calls | 0 → 20 | 不能写成 zero overhead 或加速 |

## 固定边界

- INNOVATION1_FIXED_K2 = SUPPORTED。
- LINEAR_K2 > FIXED_K2 = NOT_SUPPORTED。
- INNOVATION2_RC_NFGD = SUPPORTED，仅限冻结代理指标。
- RC_NFGD_PROPERTY_MAE_IMPROVEMENT = NOT_SUPPORTED。
- ONLINE > POST = NOT_SUPPORTED。
- BOUNDED_CORRECTION_AS_CORE = NOT_SUPPORTED。
- JOINT_SYNERGY = NOT_SUPPORTED。
- DFT_VERIFIED = false。
