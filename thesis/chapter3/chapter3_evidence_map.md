# Chapter 3 论断—证据映射

本文件用于审计《第3章 参考轨迹保留的预算约束多分支条件引导方法》。`thesis_release/`、`mattergen/` 和 `experiments/` 路径相对于最终 release worktree；`research_archive/` 路径相对于归档分支 `archive/thesis-exploration-2026` 的冻结 HEAD `848d5f03346b06e03df39f1f8bf258a394f15f7b`。SUPPORTED 表示论断受到对应冻结数据支持，但仅适用于记录的 cohort、指标和代理评价协议；MECHANISM_ONLY 表示机制或上界证据；NEGATIVE 表示预注册检验未支持相应假设；LIMITATION 表示必须保留的解释边界。

| Section | Claim | Source file | Table/Figure | Evidence status |
|---|---|---|---|---|
| 章首、3.1 | MatterGen 联合生成原子类型、周期坐标和晶胞，可进行属性条件生成 | 外部文献：Zeni et al., Nature 2025, DOI 10.1038/s41586-025-08628-5 | 参考文献[1] | EXTERNAL_VERIFIED |
| 3.1.1 | CFG 组合条件与无条件得分 | 外部文献：Ho & Salimans, arXiv:2207.12598 | 参考文献[2] | EXTERNAL_VERIFIED |
| 3.1.1 | 本研究 C0 的固定 CFG 为 2.0 | thesis_release/innovation1/configs/frozen_method_manifest.json | 式（标准 CFG） | SUPPORTED |
| 3.2.1 | V1 分字段计算残差，但输出单一共享全局 CFG | mattergen/diffusion/sampling/guidance_schedule.py；mattergen/diffusion/sampling/classifier_free_guidance.py；research_archive/innovation1/adaptive_cfg_v1/results/final_report.md | 3.2.1 公式 | SUPPORTED_IMPLEMENTATION |
| 3.2.1 | V1 使用字段均值、分阶段 EMA、\(\alpha=0.5,\beta=0.95,\varepsilon=10^{-6}\) 和 [0,5] 边界 | research_archive/innovation1/adaptive_cfg_v1/results/final_report.md；mattergen/diffusion/sampling/guidance_schedule.py | 3.2.1 公式 | SUPPORTED_IMPLEMENTATION |
| 3.2.2 | V1 Formal256 的 E-hull、Stable、NUS 方向改善，但三项 CI 均跨零 | thesis_release/innovation1/negative_results/historical_status/adaptive_v1_formal.json；research_archive/innovation1/adaptive_cfg_v1/results/final_report.md | 表3-1；图3-1 | SUPPORTED_MIXED |
| 3.2.2 | V1 Formal256 没有可核验的属性预测，不能报告该批 Property MAE | research_archive/innovation1/adaptive_cfg_v1/results/final_report.md | 正文限制 | LIMITATION |
| 3.2.2 | fresh8 未复现质量改善方向 | research_archive/innovation1/adaptive_cfg_v1/results/final_report.md；thesis_release/innovation1/negative_results/historical_status/adaptive_v1_per_seed_metrics.csv | 表3-1；图3-1 | SUPPORTED_MIXED |
| 3.3 | V3 候选为 1.75/2.0/2.25，32 seeds、256 短状态、64 完整状态 | research_archive/innovation1/oracle_v3/config/oracle_config.yaml；research_archive/innovation1/oracle_v3/results/oracle_summary.json | 3.3.1 | SUPPORTED |
| 3.3 | 完整终点 Oracle 相对 CFG2 改善 27.22% | research_archive/innovation1/oracle_v3/results/oracle_headroom.csv；thesis_release/innovation1/negative_results/historical_status/oracle_v3.json | 表3-1；图3-1 | MECHANISM_ONLY |
| 3.3 | 短视野标签用于完整终点时恶化 6.90%，短/完整标签一致率 62.5% | research_archive/innovation1/oracle_v3/results/oracle_summary.json；research_archive/innovation1/oracle_v3/results/final_report.md | 3.3.2 | SUPPORTED_NEGATIVE |
| 3.4.1 | V2 使用时间归一化、中位数聚合、置信回退、[1.5,2.5] 边界和 0.05 限速 | research_archive/innovation1/robust_v2/config/robust_adaptive_cfg_v2_config.yaml；research_archive/innovation1/robust_v2/config/algorithm_description.md | 表3-1 | SUPPORTED_IMPLEMENTATION |
| 3.4.1 | V2 fresh P0 未降低旧 V1 的 E-hull/Stable/Property 伤害率，属性护栏失败 | research_archive/innovation1/robust_v2/results/p0_final_report.md；research_archive/innovation1/robust_v2/results/harm_metrics.csv | 表3-1；图3-1 | NEGATIVE |
| 3.4.2 | Risk V4 无操作点同时达到覆盖率≥15%和属性伤害率≤10% | research_archive/innovation1/risk_v4/results/final_report.md；thesis_release/innovation1/negative_results/historical_status/risk_v4.json | 表3-1 | NEGATIVE |
| 3.4.2 | Safe V5 验证门槛失败，未运行测试/P0/Formal256 | research_archive/innovation1/safe_v5/results/final_report.md；thesis_release/innovation1/negative_results/historical_status/safe_v5.json | 表3-1 | NEGATIVE |
| 3.4.3 | Stage 低 CFG 改善属性，但破坏部分质量；脉冲未优于恒定 1.9 | research_archive/innovation1/stage_cfg/results/p0_paired_results.csv；research_archive/innovation1/stage_cfg/results/p0_bootstrap.json；research_archive/innovation1/stage_cfg/results/p0_decision_summary.json | 表3-2 | SUPPORTED_MIXED |
| 3.4.3 | 三字段均为 2.0 时字段化实现精确复现基准 | research_archive/innovation1/field_cfg/config/frozen_manifest.json；research_archive/innovation1/field_cfg/code/run_generation.py | 3.4.3 | SUPPORTED_IMPLEMENTATION |
| 3.4.3 | 属性贡献分布于 atomic+position，质量伤害分布于 position+cell | research_archive/innovation1/field_cfg/results/field_mechanism.csv；research_archive/innovation1/field_cfg/results/selected_policy.json | 3.4.3 | MECHANISM_ONLY |
| 3.4.3 | Field calibration 无候选通过全部门槛，P0/Formal256 未运行 | research_archive/innovation1/field_cfg/results/decision_summary.json；research_archive/innovation1/field_cfg/results/final_report.md | 表3-1；图3-1 | NEGATIVE |
| 3.5.2 | 第400步保存完整状态和 RNG，各分支恢复同一快照 | thesis_release/innovation1/method/confirmatory_sampler.py；thesis_release/innovation1/method/phase_b_sampler.py | 算法3-1 | SUPPORTED_IMPLEMENTATION |
| 3.5.2 | C1 的独立全程 C0 与共享前缀 C0 全部精确一致 | thesis_release/innovation1/results/c1_acquisition_accounting.json；thesis_release/innovation1/results/c1_generation_summary.json | 算法3-1 | SUPPORTED |
| 3.5.2 | Fixed-K2 为 4 400 score calls/seed，即 C0 的 2.2× | thesis_release/combined_summary/compute_summary.csv；thesis_release/innovation1/results/c1_compute_accounting.csv | 图3-6 | SUPPORTED |
| 3.5.3 | 候选库为 GPulse/APulse/PPulse/CPulse，分支点400、持续100步、脉冲1.9 | thesis_release/innovation1/method/run_confirmation_generation.py；thesis_release/innovation1/configs/frozen_method_manifest.json | 表3-3 | SUPPORTED_IMPLEMENTATION |
| 3.5.3 | 最终 Fixed-K2 为 GPulse+PPulse | thesis_release/innovation1/configs/frozen_method_manifest.json | 表3-3 | SUPPORTED |
| 3.5.4 | SAFE-A 要求属性严格改善、Validity/Stable 不低于 C0、E-hull 不高于 C0+0.01 | thesis_release/innovation1/method/analyze_confirmation.py；thesis_release/innovation1/configs/frozen_method_manifest.json | 终点选择公式；算法3-1 | SUPPORTED_IMPLEMENTATION |
| 3.6 | 合法共享前缀的候选仅 GPulse/APulse/PPulse/CPulse | thesis_release/innovation1/results/branch_oracle_report.md；thesis_release/innovation1/method/run_branch_compatible_audit.py | 表3-4 | SUPPORTED_IMPLEMENTATION |
| 3.6 | 分支兼容测试集 C0=0.02879873、Oracle-All=0.02004107，改善30.41% | thesis_release/innovation1/results/branch_oracle_metrics.csv；thesis_release/innovation1/results/branch_oracle_report.md | 表3-4 | MECHANISM_ONLY |
| 3.6 | Oracle-K2 恢复100%优化空间；Phase A Fixed-K2 为 CPulse+GPulse、恢复72.56% | thesis_release/innovation1/results/branch_oracle_k_results.csv；thesis_release/innovation1/results/branch_fixed_k_results.csv | 表3-4 | MECHANISM_ONLY |
| 3.6 | Phase A 与最终 C1 的 Fixed-K2 组合不同，不应混淆 | thesis_release/innovation1/results/branch_oracle_status.json；thesis_release/innovation1/negative_results/phase_b_decision_summary.json；thesis_release/innovation1/configs/frozen_method_manifest.json | 3.6.2 | SUPPORTED |
| 3.7 | C1 使用128个完全新种子，历史重叠为0 | thesis_release/innovation1/seeds/confirmatory_seed_manifest_256.json；thesis_release/innovation1/configs/frozen_execution_manifest.json | 表3-5 | SUPPORTED |
| 3.7 | Fixed-K2 Property MAE 0.034796→0.026332，相对改善24.32% | thesis_release/combined_summary/main_results.csv；thesis_release/innovation1/tables/final_confirmatory_results.csv | 表3-5；图3-2 | SUPPORTED |
| 3.7 | Fixed-K2 绝对增益0.008464，20k bootstrap CI [0.005817,0.011391] | thesis_release/innovation1/results/fixed_vs_c0_bootstrap_20k.json；thesis_release/innovation1/results/c1_paired_results.csv | 图3-3、图3-4 | SUPPORTED |
| 3.7 | Fixed-K2 配对结果为53改善、75持平/回退、0损失 | thesis_release/innovation1/results/fixed_vs_c0_bootstrap_20k.json；thesis_release/innovation1/results/c1_selected_outcomes.csv | 图3-3 | SUPPORTED_BY_SELECTOR |
| 3.7 | Fixed-K2 的 E-hull、Stable、NUS 改善，Validity 保持100% | thesis_release/combined_summary/main_results.csv；thesis_release/innovation1/results/c1_metrics.csv | 表3-5；图3-5 | SUPPORTED_GUARDRAIL |
| 3.8.1 | Phase B 为64个新种子，32/16/16 划分，179个原始特征 | thesis_release/innovation1/negative_results/phase_b_final_report.md；thesis_release/innovation1/configs/frozen_method_manifest.json | 3.8.1 | SUPPORTED |
| 3.8.1 | Phase B Linear-K2=0.023634、Fixed-K2=0.024597，3.92%方向收益但 CI 跨0 | thesis_release/innovation1/negative_results/phase_b_test_metrics.csv；thesis_release/innovation1/negative_results/phase_b_allocator_bootstrap_20k.json | 3.8.1 | NOT_SUPPORTED |
| 3.8.2 | 重建实现历史 Top-2 16/16 精确一致，最大指标误差1.11e-16 | thesis_release/innovation1/negative_results/reconstruction_audit.json；thesis_release/innovation1/negative_results/reconstruction_test_scores.csv | 3.8.2 | SUPPORTED_RECONSTRUCTION |
| 3.8.2 | 重建 artifact SHA-256 已冻结 | thesis_release/innovation1/configs/frozen_method_manifest.json；thesis_release/REPRODUCIBILITY.md | 3.8.2 | SUPPORTED_INTEGRITY |
| 3.8.3 | C1 Linear-K2=0.027522，高于 Fixed-K2=0.026332；相对增益-4.52% | thesis_release/combined_summary/main_results.csv；thesis_release/innovation1/results/c1_continuation_decision.json | 表3-5；图3-2 | NOT_SUPPORTED |
| 3.8.3 | Linear-vs-Fixed mean=-0.001190，CI [-0.003148,0.000740]，W/T/L=31/62/35 | thesis_release/innovation1/results/c1_linear_bootstrap_20k.json；thesis_release/innovation1/results/c1_paired_results.csv | 3.8.3 | NOT_SUPPORTED |
| 3.8.3 | C1 未通过继续门槛，C2=NOT_RUN | thesis_release/innovation1/results/c1_continuation_decision.json；thesis_release/innovation1/configs/confirmatory_protocol.md | 3.8.3 | SUPPORTED_STOP_RULE |
| 3.9 | Fixed-K2 依赖终点评价器，当前无 DFT 验证 | thesis_release/CLAIMS_AND_LIMITATIONS.md；thesis_release/README.md | 局限性 | LIMITATION |
| 3.9 | 未证明 Fixed-K2 优于所有等预算通用 best-of-N | thesis_release/CLAIMS_AND_LIMITATIONS.md | 局限性 | LIMITATION |
| 3.9 | Adaptive CFG 稳定优于固定 CFG 的论断不受支持 | thesis_release/CLAIMS_AND_LIMITATIONS.md；thesis_release/combined_summary/experiment_status.csv | 讨论；图3-1 | NOT_SUPPORTED |
| 3.9 | Linear-K2 优于 Fixed-K2 的论断不受支持 | thesis_release/CLAIMS_AND_LIMITATIONS.md；thesis_release/combined_summary/main_results.csv | 讨论；表3-5 | NOT_SUPPORTED |
| 章首、3.9 | MatterSim 为代理评价器，需补齐论文总参考文献条目 | experiments/metric_framework_reassessment/final_report.md 中的 arXiv:2405.04967 链接 | 参考文献[3] | CITATION_NEEDED |

## 图表来源清单

| 论文编号 | Release ID | 文件 | 用途 |
|---|---|---|---|
| 图3-1 | I1-F6 | thesis_release/innovation1/figures/I1_F6_evidence_progression.png | 研究证据演化 |
| 图3-2 | I1-F1 | thesis_release/innovation1/figures/I1_F1_c1_property_mae.png | C1 四方法 Property MAE |
| 图3-3 | I1-F2 | thesis_release/innovation1/figures/I1_F2_fixed_vs_c0_paired.png | Fixed-K2 与 C0 配对结果 |
| 图3-4 | I1-F3 | thesis_release/innovation1/figures/I1_F3_fixed_gain_bootstrap.png | 20k bootstrap 增益 |
| 图3-5 | I1-F4 | thesis_release/innovation1/figures/I1_F4_quality_guardrails.png | C1 质量护栏 |
| 图3-6 | I1-F5 | thesis_release/innovation1/figures/I1_F5_compute_comparison.png | 计算量比较 |
| 表3-1 | — | thesis_release/combined_summary/experiment_status.csv 及历史状态文件 | 自适应 CFG 探索摘要 |
| 表3-2 | — | research_archive/innovation1/stage_cfg/results/p0_paired_results.csv | Stage P0 |
| 表3-3 | — | thesis_release/innovation1/method/run_confirmation_generation.py | 分支候选定义 |
| 表3-4 | — | thesis_release/innovation1/results/branch_oracle_*.csv | 分支 Oracle 上界 |
| 表3-5 | — | thesis_release/innovation1/tables/final_confirmatory_results.csv | C1-128 最终结果 |

## 审计摘要

- EVIDENCE_MAP = COMPLETE
- CITATION_NEEDED_COUNT = 1
- UNRESOLVED_EVIDENCE_COUNT = 0
- CLAIM_EVIDENCE_CONSISTENCY = PASS
