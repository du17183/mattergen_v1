# 第3—5章统一数字主表

本文件是人工审阅入口；完整机器可读精确值见 `master_numbers.csv`。冻结发布包仍是事实来源，本表不产生新结果。正文显示遵循 `number_formatting_rules.md`。

## 创新点1：最终确认与负确认

| 比较 | 样本 | 统一正文数字 | 状态 | 冻结来源 |
|---|---:|---|---|---|
| Fixed-K2 vs C0，Property MAE | 128对 | 0.034796→0.026332；改善24.32%；绝对CI [0.005817, 0.011391]；W/T/L=53/75/0 | SUPPORTED | `thesis_release/innovation1/results/fixed_vs_c0_bootstrap_20k.json` |
| Fixed-K2 质量护栏 | 128 | E-hull 0.104296→0.081135；Stable 58.59%→70.31%；NUS 32.81%→37.50%；Validity 100%→100% | SUPPORTED_GUARDRAIL | `thesis_release/innovation1/results/c1_metrics.csv` |
| Fixed-K2 分布边界 | 128 | Novel 71.09%→65.63%；Unique 99.22%→100% | MIXED/OBSERVED | 同上 |
| Linear-K2 vs Fixed-K2，C1 | 128对 | 0.027522 vs 0.026332；有利差−0.001190；相对−4.52%；CI [−0.003148, 0.000740]；31/62/35 | NOT_SUPPORTED | `thesis_release/innovation1/results/c1_linear_bootstrap_20k.json` |
| Linear-K2 vs Fixed-K2，Phase B test | 16对 | 0.023634 vs 0.024597；方向性+3.92%；CI [−0.002618, 0.005214] | NOT_SUPPORTED | `thesis_release/innovation1/negative_results/phase_b_allocator_bootstrap_20k.json` |
| Branch-Compatible Oracle-All vs C0 | 12 | 0.028799→0.020041；改善30.41% | SUPPORTED_MECHANISM | `thesis_release/innovation1/results/branch_oracle_metrics.csv` |
| 部署成本 | 每样本 | C0 2 000 score calls；Fixed/Linear 4 400，2.2×；Fixed回退率58.59% | VERIFIED | `thesis_release/combined_summary/compute_summary.csv` |

结论锁：`Fixed-K2 vs C0 = SUPPORTED`；`Linear-K2 > Fixed-K2 = NOT_SUPPORTED`。Oracle 只支持“存在分支空间”，不支持可部署效果。

## 创新点2：正式确认与独立代理评估

| 比较 | 样本 | 统一正文数字 | 状态 | 冻结来源 |
|---|---:|---|---|---|
| 阶段可靠性 | 32 | \(t=0.02\)，Spearman=0.925587 | SUPPORTED_MECHANISM | `thesis_release/innovation2/configs/p0_guidance_config.yaml` |
| P0 MaxF | 16对 | 0.304028→0.253392；改善16.66%；相对CI [10.17%, 31.24%]；15/0/1 | PASS/PILOT | P0 `decision_summary.json` |
| Formal32 MaxF | 32对 | 0.183472→0.134728；改善26.57%；相对CI [14.27%, 52.86%]；32/0/0 | PASS | Formal32 `decision_summary.json` |
| Formal256 MatterSim MaxF | 256对 | 0.226408→0.158911；改善29.81%；相对CI [21.42%, 40.88%]；254/0/2 | SUPPORTED | Formal256 `paired_bootstrap.json` |
| Formal256 MatterSim Mean Force | 256对 | 0.098702→0.069373；改善29.72%；相对CI [24.01%, 36.96%]；255/0/1 | SUPPORTED | 同上 |
| Formal256 RMSD | 256对 | 0.051137→0.043893；改善14.17%；相对CI [4.54%, 25.95%]；216/21/19 | SUPPORTED | 同上 |
| Formal256 Property MAE | 256对 | 0.009757→0.009868；有利变化−1.14%；CI [−2.46%, −0.22%]；95/0/161 | NOT_SUPPORTED_PROPERTY_IMPROVEMENT | 同上 |
| Stable / Validity | 256对 | 80.86%不变 / 100%不变 | SUPPORTED_GUARDRAIL | 同上 |
| CHGNet MaxF | 256对 | 0.193891→0.166057；改善14.36%；绝对CI [0.007052, 0.064868]；172/0/84 | SUPPORTED_INDEPENDENT_SURROGATE | CHGNet `independent_per_structure.csv` |
| CHGNet Mean Force | 256对 | 0.087976→0.079541；改善9.59%；绝对CI [0.002384, 0.019148]；169/0/87 | SUPPORTED_INDEPENDENT_SURROGATE | 同上 |
| 计算成本 | 每样本 | 两者均2 000 score calls；RC-NFGD另有20次 MatterSim；92.663 s vs 92.849 s，不主张加速 | VERIFIED/NO_SPEEDUP_CLAIM | `thesis_release/combined_summary/compute_summary.csv` |

结论锁：`RC-NFGD = SUPPORTED`，但 Property MAE 轻微恶化、CHGNet 尾部混合；`DFT_VERIFIED=false`。

## 机制、消融与联合边界

| 实验 | 统一正文数字 | 允许结论 |
|---|---|---|
| 真实力方向 G1 vs G0 | Mean MaxF 0.141690→0.098500；改善30.48%；CI [24.66%, 38.87%]；31/0/1 | 力方向携带有效信息，不证明理论最优 |
| 有界修正 | T0/T1/T2为0.277138/0.231502/0.060834 | 不支持 bound 是性能核心；其角色是保守控制 |
| 等预算 POST | C0/online/POST为0.214179/0.163650/0.043275 | 不支持在线方法支配 POST |
| A5兼容性 | 力收益保留55.49%，低于70%门槛 | `JOINT_SYNERGY=NOT_SUPPORTED`；且未直接验证 Fixed-K2×RC-NFGD |

## 章节使用原则

- 第3章解释创新点1的方法、负结果链和独立确认；第4章解释创新点2的方法、确认与消融；第5章只做统一比较和综合解释。
- 第5章复述数值时使用本表展示精度，不重新推导算法或另设阈值。
- 所有数值均为冻结代理协议结果；没有 DFT、声子、长时分子动力学或湿实验验证。
