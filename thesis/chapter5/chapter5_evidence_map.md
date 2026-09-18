# Chapter 5 Evidence Map

本文件把第5章每项可核查主张映射到冻结发布包。路径均相对于仓库根目录；表号和图号指 `thesis/chapter5/chapter5_final_draft.md`。`SUPPORTED_MECHANISM` 只表示机制或上界证据，不等于正式可部署性能；所有物理量均为代理评价，`DFT_VERIFIED=false`。

| Section | Claim | Source artifact | Table | Figure | Evidence type | Evidence status | Limitation |
|---|---|---|---|---|---|---|---|
| 5.1.1 | 创新点1历史在线 Adaptive CFG 结果为 MIXED/FAIL | `thesis_release/combined_summary/experiment_status.csv` | 表5-1 | 图5-2 | 探索/机制汇总 | NOT_SUPPORTED | 各实验规模与门槛不同，不能合并估计效应 |
| 5.1.1 | C1 Fixed-K2 是128对全新种子的确认实验 | `thesis_release/innovation1/configs/confirmatory_protocol.md`; `thesis_release/innovation1/seeds/confirmatory_seed_manifest_256.json` | 表5-1 | 图5-1 | 预注册确认协议 | SUPPORTED | C2 仅在门槛通过时运行，实际未运行 |
| 5.1.1 | RC-NFGD 采用 P0→Formal32→Formal256 分阶段冻结验证 | `thesis_release/innovation2/configs/p0_guidance_config.yaml`; `thesis_release/innovation2/configs/formal32_config.yaml`; `thesis_release/innovation2/configs/formal256_config.yaml` | 表5-1 | 图5-4 | 冻结验证协议 | SUPPORTED | 仅当前任务与 checkpoint |
| 5.1.2 | 核心连续指标使用20 000次配对 bootstrap | `thesis_release/innovation1/results/fixed_vs_c0_bootstrap_20k.json`; `thesis_release/innovation2/results/mattersim_late_force_guidance_formal256/paired_bootstrap.json` | 表5-2/5-3 | 图5-3 | 配对统计 | COMPLETE | bootstrap 反映当前 cohort，不保证总体分布 |
| 5.1.3 | 本研究没有 DFT 验证 | `thesis_release/CLAIMS_AND_LIMITATIONS.md`; `thesis_release/innovation2/results/final_decision.json` | 表5-7 | — | 证据边界 | FALSE（DFT claim） | MatterSim/CHGNet 都是 MLIP |
| 5.2.1 | Fixed-K2 Property MAE 0.034796→0.026332 | `thesis_release/combined_summary/main_results.csv`; `thesis_release/innovation1/results/c1_metrics.csv` | 表5-2 | 图5-1 | C1确认 | SUPPORTED | CHGNet 代理属性；当前目标条件 |
| 5.2.1 | Fixed-K2 绝对改善0.008464、相对改善24.32% | `thesis_release/innovation1/results/c1_property_summary.json` | 表5-2 | 图5-1 | C1确认 | SUPPORTED | 不代表跨任务平均效应 |
| 5.2.1 | Fixed-K2 改善95% CI 为[0.005817,0.011391] | `thesis_release/innovation1/results/fixed_vs_c0_bootstrap_20k.json` | 表5-2 | — | 20k配对 bootstrap | SUPPORTED | 区间基于注册的128对样本 |
| 5.2.1 | Fixed-K2 W/T/L=53/75/0 | `thesis_release/innovation1/results/c1_paired_results.csv`; `thesis_release/innovation1/results/c1_selected_outcomes.csv` | 表5-2 | — | 配对结果 | SUPPORTED | 0 loss 受接受/回退规则保证，不是候选永不失败 |
| 5.2.1 | MAE次序 Fixed<Linear<Random<C0 | `thesis_release/innovation1/tables/final_confirmatory_results.csv` | 表5-2 | 图5-1 | 描述+确认 | SUPPORTED（次序） | Linear vs Random 的区间不能支持强优势结论 |
| 5.2.2 | Fixed-K2 E-hull 0.104296→0.081135 | `thesis_release/innovation1/results/c1_metrics.csv` | 表5-2 | — | 冻结护栏 | SUPPORTED_GUARDRAIL | 未报告显著性区间 |
| 5.2.2 | Fixed-K2 Stable 58.59%→70.31% | `thesis_release/innovation1/results/c1_metrics.csv` | 表5-2 | — | 冻结护栏 | SUPPORTED_GUARDRAIL | 代理稳定性，不是实验稳定性 |
| 5.2.2 | Fixed-K2 NUS 32.81%→37.50%，Validity保持100% | `thesis_release/innovation1/results/c1_metrics.csv` | 表5-2 | — | 冻结护栏 | SUPPORTED_GUARDRAIL | 依赖数据库与有效性规则 |
| 5.2.2 | Fixed-K2 Novel 71.09%→65.63% | `thesis_release/innovation1/results/c1_metrics.csv` | 表5-2 | — | 负向描述指标 | OBSERVED | 不应写成全面质量提升 |
| 5.2.2 | Fixed-K2 MaxF和RMSD均值下降 | `thesis_release/innovation1/results/c1_metrics.csv` | — | — | 描述性代理指标 | OBSERVED | 非创新点1预注册主终点 |
| 5.2.2 | Fixed-K2 回退率58.59% | `thesis_release/innovation1/results/c1_compute_accounting.csv`; `thesis_release/innovation1/results/c1_metrics.csv` | 表5-2/5-6 | — | 选择器行为 | VERIFIED | 高回退意味着多数样本保持C0 |
| 5.3.1 | Branch Oracle测试n=12的MAE 0.028799→0.020041 | `thesis_release/innovation1/results/branch_oracle_metrics.csv`; `thesis_release/innovation1/results/branch_oracle_report.md` | 表5-5/5-7 | 图5-2 | 探索性机制上界 | SUPPORTED_MECHANISM | 事后Oracle，小样本，不可部署 |
| 5.3.1 | Branch Oracle相对改善30.41% | `thesis_release/innovation1/results/branch_oracle_status.json` | 表5-5 | 图5-2 | 探索性机制上界 | SUPPORTED_MECHANISM | 不用于最终性能外推 |
| 5.3.1 | Robust V2、V4、V5和Field等未通过冻结门槛 | `thesis_release/innovation1/negative_results/historical_status/` | 表5-1 | 图5-2 | 负结果链 | FAIL/NOT_SUPPORTED | 不同路线回答不同子问题 |
| 5.3.2 | Phase B 为64样本32/16/16划分 | `thesis_release/innovation1/negative_results/phase_b_final_report.md` | 表5-1/5-5 | — | 模型选择/留出测试 | COMPLETE | 留出测试仅16样本 |
| 5.3.2 | Phase B Linear MAE 0.023634 vs Fixed 0.024597 | `thesis_release/innovation1/negative_results/phase_b_test_metrics.csv` | 表5-5 | — | 留出方向性结果 | DIRECTIONAL_ONLY | 置信区间跨零 |
| 5.3.2 | Phase B Linear vs Fixed CI [-0.002618,0.005214] | `thesis_release/innovation1/negative_results/phase_b_allocator_bootstrap_20k.json` | 表5-5 | — | 20k配对 bootstrap | NOT_SUPPORTED | n=16；不可写成已验证优势 |
| 5.3.2 | Linear-K2 确定性重建16/16 Top-2一致 | `thesis_release/innovation1/negative_results/reconstruction_audit.json` | — | — | 重建审计 | RECONSTRUCTED_ARTIFACT | 只恢复对象，不改变历史结论 |
| 5.3.2 | 重建模型哈希已冻结 | `thesis_release/innovation1/negative_results/frozen_linear_k2_parameters.json`; `thesis_release/innovation1/negative_results/reconstruction_audit.json` | — | — | 工程复现 | VERIFIED | 非新的模型选择过程 |
| 5.3.3 | C1 Linear MAE 0.027522，高于 Fixed 0.026332 | `thesis_release/innovation1/results/c1_historical_final_report.md` | 表5-2/5-5 | 图5-1 | 负确认 | NOT_SUPPORTED | Linear仍因回退优于C0 |
| 5.3.3 | Linear vs Fixed增益-0.001190，CI[-0.003148,0.000740] | `thesis_release/innovation1/results/c1_linear_bootstrap_20k.json` | 表5-5 | — | 20k配对 bootstrap | NOT_SUPPORTED | CI跨零且点估计不利 |
| 5.3.3 | Linear vs Fixed W/T/L=31/62/35 | `thesis_release/innovation1/results/c1_linear_bootstrap_20k.json`; `thesis_release/innovation1/results/c1_paired_results.csv` | — | — | 配对结果 | OBSERVED | 不支持额外分配价值 |
| 5.3.3 | C2 未运行且未 rescue tuning | `thesis_release/innovation1/results/c1_continuation_decision.json` | — | — | 预注册停止规则 | VERIFIED | 停止不能解释为缺失正结果 |
| 5.4.1 | Formal256 完成256/256，无样本替换 | `thesis_release/innovation2/results/mattersim_late_force_guidance_formal256/decision_summary.json`; `thesis_release/innovation2/results/mattersim_late_force_guidance_formal256/paired_results.csv` | 表5-3 | 图5-3 | 正式确认 | SUPPORTED | 当前注册 cohort |
| 5.4.1 | MatterSim MaxF 0.226408→0.158911，改善29.81% | `thesis_release/innovation2/tables/formal256_main_results.csv` | 表5-3 | 图5-3 | 正式确认主终点 | SUPPORTED | 同源代理评价 |
| 5.4.1 | MaxF相对CI[21.42%,40.88%]，W/T/L=254/0/2 | `thesis_release/innovation2/results/mattersim_late_force_guidance_formal256/paired_bootstrap.json` | 表5-3 | 图5-3 | 20k配对 bootstrap | SUPPORTED | 非DFT力 |
| 5.4.1 | Mean Force降低29.72%，CI[24.01%,36.96%] | `thesis_release/innovation2/tables/formal256_main_results.csv` | 表5-3 | 图5-3 | 正式确认 | SUPPORTED | 同源代理评价 |
| 5.4.1 | RMSD降低14.17%，CI[4.54%,25.95%] | `thesis_release/innovation2/tables/formal256_main_results.csv` | 表5-3 | 图5-3 | 正式确认 | SUPPORTED | 代理松弛协议 |
| 5.4.1 | MatterSim尾部P90/P95/最大值总体下降 | `thesis_release/innovation2/results/mattersim_late_force_guidance_formal256/final_report.md` | — | — | 描述性尾部 | OBSERVED | 非预注册主检验 |
| 5.4.2 | P0/Formal32/Formal256 MaxF改善16.66%/26.57%/29.81% | `thesis_release/innovation2/tables/effect_scale_consistency.csv` | — | 图5-4 | 独立cohort一致性 | SUPPORTED | 不能解释为随n单调增长 |
| 5.4.2 | 三阶段效应区间均在有利方向 | `thesis_release/innovation2/tables/effect_scale_consistency.csv` | — | 图5-4 | 跨cohort复现 | SUPPORTED | cohort绝对难度不同 |
| 5.4.3 | Property MAE恶化1.14%，CI[-2.46%,-0.22%] | `thesis_release/innovation2/tables/formal256_main_results.csv` | 表5-3/5-7 | — | 正式护栏 | NOT_SUPPORTED（改善claim） | 必须报告属性代价 |
| 5.4.3 | Stable保持80.86%，Validity保持100% | `thesis_release/innovation2/tables/formal256_main_results.csv` | 表5-3 | — | 冻结护栏 | SUPPORTED_GUARDRAIL | 不能代表全部质量指标保持 |
| 5.5 | CHGNet MaxF 0.193891→0.166057，相对改善14.36% | `thesis_release/innovation2/tables/chgnet_validation.csv` | 表5-4 | 图5-5 | 独立代理评估 | SUPPORTED_INDEPENDENT_SURROGATE | 独立于引导，不是DFT独立 |
| 5.5 | CHGNet MaxF绝对CI[0.007052,0.064868] | `thesis_release/innovation2/results/chgnet_formal256/paired_bootstrap.json` | 表5-4 | 图5-5 | 配对 bootstrap | SUPPORTED | 效应小于MatterSim |
| 5.5 | CHGNet Mean Force改善9.59%，绝对CI[0.002384,0.019148] | `thesis_release/innovation2/tables/chgnet_validation.csv` | 表5-4 | 图5-5 | 独立代理评估 | SUPPORTED_INDEPENDENT_SURROGATE | W/T/L含87个loss |
| 5.5 | CHGNet MaxF P90/P95恶化且Mean Force中位数恶化 | `thesis/chapter4/chapter4_final_draft.md`第4.7.2节；`thesis_release/innovation2/results/chgnet_formal256/independent_summary.csv` | — | 图5-5 | 尾部边界 | MIXED | 不支持所有分位数改善；Mean Force尾部由已冻结第4章记录 |
| 5.6 | 阶段诊断在t=0.02得到Spearman 0.925587 | `thesis/chapter4/chapter4_final_draft.md`第4.2节；`thesis_release/innovation2/configs/frozen_method_definition.md` | 表5-5 | — | 机制/校准 | SUPPORTED | 相关性不等于因果最优 |
| 5.6.1 | 方向消融G0/G1/G2/G3/G4均值为0.141690/0.098500/0.153744/0.151428/0.188137 | `thesis/chapter4/chapter4_final_draft.md`第4.8.1节；`thesis_release/innovation2/results/final_report.md` | 表5-5 | — | 方向消融 | SUPPORTED | 控制分支重放G1尺度，有路径依赖 |
| 5.6.1 | G1相对G0改善30.48%，CI[24.66%,38.87%] | `thesis/chapter4/chapter4_final_draft.md`第4.8.1节；`thesis_release/innovation2/results/final_report.md` | 表5-5 | — | 32对配对消融 | SUPPORTED | 小于Formal256证据层级 |
| 5.6.2 | T0/T1/T2 MaxF为0.277138/0.231502/0.060834 | `thesis/chapter4/chapter4_final_draft.md`第4.8.2节 | 表5-5 | — | bounded消融 | NOT_SUPPORTED（必要性） | T2同时移除两个限制，不能分开归因 |
| 5.6.2 | bound 保留为保守数值控制而非性能核心 | `thesis_release/CLAIMS_AND_LIMITATIONS.md` | 表5-5/5-7 | — | 结论边界 | NOT_SUPPORTED（性能核心） | 跨环境安全性未确认 |
| 5.6.3 | 等预算C0/online/POST Mean MaxF为0.214179/0.163650/0.043275 | `thesis/chapter4/chapter4_final_draft.md`第4.8.3节；`thesis_release/innovation2/results/final_report.md` | 表5-5/5-7 | — | 等预算消融 | ONLINE_DOMINANCE_NOT_SUPPORTED | n=32且同一MatterSim指标 |
| 5.6.3 | 在线F0不是最低代理力方案 | `thesis_release/CLAIMS_AND_LIMITATIONS.md` | 表5-7 | — | 负机制结论 | NOT_SUPPORTED | 不否定在线反馈相对C0有效 |
| 5.7 | Fixed-K2每样本4 400 score调用、3次终点评价、2.2× | `thesis_release/innovation1/results/c1_compute_accounting.csv`; `thesis_release/combined_summary/compute_summary.csv` | 表5-6 | 图5-6 | 计算审计 | VERIFIED | 不同硬件实际时间需重测 |
| 5.7 | Linear-K2 与 Fixed-K2 匹配2.2×预算 | `thesis_release/combined_summary/compute_summary.csv` | 表5-6 | 图5-6 | 公平预算比较 | VERIFIED | 只匹配记录的主要部署调用 |
| 5.7 | Phase B采集3.4×是数据构建而非部署 | `thesis_release/innovation1/results/c1_acquisition_accounting.json`; `thesis_release/combined_summary/compute_summary.csv` | 表5-6 | 图5-6 | 成本分解 | VERIFIED | 不可用于夸大/缩小部署成本 |
| 5.7 | RC-NFGD与C0均为2 000 score调用，另加20次MatterSim | `thesis_release/combined_summary/compute_summary.csv`; `thesis_release/innovation2/configs/frozen_method_definition.md` | 表5-6 | — | 计算审计 | VERIFIED | 神经势调用成本随设备/规模变化 |
| 5.7 | 记录时间92.663s vs 92.849s，不能宣称加速 | `thesis_release/combined_summary/compute_summary.csv` | 表5-6 | — | 运行时间描述 | NO_SPEEDUP_CLAIM | 运行波动且batch-size-1 |
| 5.8 | A5 C0/A0/B0/AB MaxF为0.218922/0.239629/0.171833/0.192794 | `thesis/chapter4/chapter4_final_draft.md`第4.8.4节；`thesis_release/innovation2/results/final_report.md` | 表5-5 | — | 兼容性实验 | FAIL | 测试历史Adaptive CFG，不是Fixed-K2完整组合 |
| 5.8 | 组合只保留55.49%单独力改善，低于70%门槛 | `thesis/chapter4/chapter4_final_draft.md`第4.8.4节；`thesis_release/innovation2/results/final_report.md` | 表5-5/5-7 | — | 预注册兼容门槛 | FAIL | 不能外推未来所有组合 |
| 5.8 | 两项创新应独立报告，不宣称协同 | `thesis_release/CLAIMS_AND_LIMITATIONS.md` | 表5-7 | — | 最终解释 | NOT_SUPPORTED（synergy） | 共同论文主线不等于联合模型 |
| 5.9.1 | Linear重建发生在新确认seed注册之前 | `thesis_release/innovation1/negative_results/reconstruction_audit.json`; `thesis_release/innovation1/configs/protocol_amendment.md` | — | — | 防泄漏审计 | VERIFIED | 依赖归档时间与哈希记录 |
| 5.9.2 | 方法适用范围限于当前checkpoint/目标/代理 | `thesis_release/CLAIMS_AND_LIMITATIONS.md`; `thesis_release/innovation1/README.md`; `thesis_release/innovation2/README.md` | 表5-7 | — | 外部有效性边界 | LIMITED | 跨属性、模型、材料域需重新确认 |
| 5.9.3 | Fixed-K2最终状态SUPPORTED | `thesis_release/innovation1/README.md`; `thesis_release/combined_summary/main_results.csv` | 表5-7 | 图5-1 | 最终确认 | SUPPORTED | surrogate-only，2.2×成本 |
| 5.9.3 | Linear-K2最终状态NOT_SUPPORTED | `thesis_release/innovation1/results/c1_continuation_decision.json` | 表5-7 | — | 负确认 | NOT_SUPPORTED | 不否定未来不同学习器 |
| 5.9.3 | RC-NFGD最终状态SUPPORTED | `thesis_release/innovation2/results/final_decision.json`; `thesis_release/innovation2/README.md` | 表5-7 | 图5-3/5-4/5-5 | 正式确认+独立代理评估 | SUPPORTED | 属性代价、尾部混合、无DFT |
| 5.9.3 | 联合协同NOT_SUPPORTED | `thesis_release/CLAIMS_AND_LIMITATIONS.md`; `thesis_release/innovation2/results/final_report.md` | 表5-7 | — | 兼容性负结果 | NOT_SUPPORTED | 未直接确认Fixed-K2×RC-NFGD |

## Evidence audit summary

- 章节主张均可追溯到 `thesis_release/` 冻结文件或已冻结的第3/4章结果段。
- 使用的发布图件为 I1-F1、I1-F5、I1-F6、I2-F3、I2-F4、I2-F5；未生成新图。
- 没有把探索性 Oracle、P0、机制消融或单个均值提升为最终确认结论。
- 明确保留 Linear-K2、bounded 核心解释、online-over-POST、联合协同和 DFT 验证的负结论。
- 本章没有新增外部文献事实，`CITATION_NEEDED_COUNT=0`。
