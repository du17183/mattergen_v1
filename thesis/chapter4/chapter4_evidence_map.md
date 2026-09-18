# Chapter 4 论断—证据映射

本文件用于审计《第4章 基于阶段可靠性校准的后期神经力场引导扩散方法》。最终 release 证据位于写作分支基点 5b572c61c70c4287147616beae7f72d28955cf98 的 thesis_release/；探索归档位于 archive/thesis-exploration-2026 冻结 HEAD 848d5f03346b06e03df39f1f8bf258a394f15f7b 的 research_archive/。SUPPORTED 只表示论断在记录的 cohort、指标与代理评价协议下得到支持，不表示 DFT、实验或普适性确认。SUPPORTED_IMPLEMENTATION 表示源代码与冻结配置直接支持；SUPPORTED_INDEPENDENT_SURROGATE 表示仅由未参与力引导的 CHGNet 神经势支持；NOT_SUPPORTED 与 LIMITATION 必须在正文中保留。

| Section | Claim | Source artifact | Figure | Table | Evidence status | Limitation |
|---|---|---|---|---|---|---|
| 章首、4.1 | MatterGen 联合去噪原子类型、周期坐标和晶胞并支持属性条件生成 | 外部文献：Zeni et al., Nature 2025, DOI 10.1038/s41586-025-08628-5 | — | — | EXTERNAL_VERIFIED | 不将原论文的实验验证外推到本章样本 |
| 4.1.1 | 扩散模型从噪声反演数据，score-SDE 可用 predictor-corrector 求解 | 外部文献：Ho et al., NeurIPS 2020；Song et al., ICLR 2021 | — | — | EXTERNAL_VERIFIED | 仅为方法背景 |
| 4.1.1 | C0 为目标磁性密度0.2、固定CFG 2.0、1000步、每步1次corrector | thesis_release/innovation2/configs/formal32_config.yaml；configs/formal256_config.yaml | I2-F1 | 表4-2 | SUPPORTED_IMPLEMENTATION | 限当前checkpoint与条件任务 |
| 4.1.1 | MaxF/MeanF 是神经势代理受力，不是DFT或可合成性证明 | thesis_release/CLAIMS_AND_LIMITATIONS.md；innovation2/results/final_report.md | — | 表4-4、表4-5 | LIMITATION | DFT_VERIFIED=false |
| 4.2.1 | 诊断为32 seeds，16 calibration + 16 validation，时间点0.10/0.05/0.02/0.01及终态 | 原始冻结诊断：experiments/mattersim_late_force_guidance_p0/diagnostic_config.yaml；diagnostic_report.md | — | 表4-1 | SUPPORTED | 归档为summary-only，原始路径记录于SOURCE.md |
| 4.2.1 | t=0.00行实际是epsilon predictor后的model_t=0.001，不用于阶段选择 | thesis_release/innovation2/results/mattersim_late_force_guidance_p0/final_report.md；原始 diagnostic_report.md | — | 表4-1 | SUPPORTED | 不是数学t=0求值 |
| 4.2.1 | 160次诊断评价100%有限/有效、明显OOD为0 | 原始 diagnostic_report.md；diagnostic_summary.json | — | 表4-1 | SUPPORTED | OOD由冻结启发式阈值定义 |
| 4.2.2 | 最佳非终态t=0.02，预测干净结构MaxF–最终MaxF Spearman=0.925587，p=3.422e-14，n=32 | thesis_release/innovation2/configs/p0_guidance_config.yaml；原始 diagnostic_correlations.csv；diagnostic_summary.json | — | 表4-1 | SUPPORTED | 支持排序可靠性，不是物理精确性 |
| 4.2.2 | t=0.02验证子集Spearman=0.982353 | 原始 diagnostic_correlations.csv | — | 表4-1 | SUPPORTING | n=16，仅作子集一致性说明 |
| 4.2.2 | t=0.02能量/原子与最终E-hull Spearman=0.269428，未达0.30辅助门槛 | 原始 diagnostic_summary.json；diagnostic_config.yaml | — | 表4-1 | NOT_SUPPORTED_AUXILIARY | 不外推能量可靠性 |
| 4.3.1 | 原子类型的预测干净结构由argmax输出加离散offset得到 | thesis_release/innovation2/method/late_force_sampler.py 中的 _clean_atoms | I2-F1 | — | SUPPORTED_IMPLEMENTATION | 输出变量沿用代码中的score命名 |
| 4.3.1 | 位置的预测干净结构为wrap(f_t+sigma_f^2 s_f) | thesis_release/innovation2/method/late_force_sampler.py 中的 _clean_atoms | I2-F1 | — | SUPPORTED_IMPLEMENTATION | 限实际position SDE |
| 4.3.1 | 晶胞的预测干净结构含alpha、sigma与limit mean校正 | thesis_release/innovation2/method/late_force_sampler.py 中的 _clean_atoms | I2-F1 | — | SUPPORTED_IMPLEMENTATION | 不能用统一x0公式替代 |
| 4.3.2 | MatterSim通过能量梯度关系提供原子力，实际工程调用inference | 外部文献：Yang et al., arXiv:2405.04967；thesis_release/innovation2/method/late_force_sampler.py | I2-F1 | — | EXTERNAL_VERIFIED + SUPPORTED_IMPLEMENTATION | MatterSim是MLIP，不是DFT |
| 4.3.2 | 修正前去除平均力以消除整体平移 | thesis_release/innovation2/method/late_force_sampler.py 中的 bounded_cartesian_force_correction | I2-F1 | — | SUPPORTED_IMPLEMENTATION | 不等价于完整动量/应力约束 |
| 4.3.3 | 行向量坐标映射delta_f=delta_r@inv(clean_cell) | thesis_release/innovation2/method/late_force_sampler.py 中的 cartesian_to_fractional_correction；configs/frozen_method_definition.md | I2-F1 | — | SUPPORTED_IMPLEMENTATION | 预测干净晶胞无效时执行安全回退 |
| 4.3.4 | correction scale为eta/max(max centered-force norm,F_ref)，再应用hard cap | thesis_release/innovation2/method/late_force_sampler.py 中的 bounded_cartesian_force_correction | I2-F1 | 表4-2 | SUPPORTED_IMPLEMENTATION | cap不是已证明的性能核心 |
| 4.3.4 | F_ref=0.0779514922 eV/Å、eta=0.005 Å、cap=0.01 Å | thesis_release/innovation2/configs/formal32_config.yaml；configs/p0_guidance_config.yaml | I2-F1 | 表4-2 | SUPPORTED | 参数在正式cohort前冻结 |
| 4.3.4 | 候选最小周期距离需≥0.5 Å；异常使用原score | thesis_release/innovation2/method/late_force_sampler.py 中的 position_correction_is_safe 与异常分支 | I2-F1 | 表4-2 | SUPPORTED_IMPLEMENTATION | 是数值回退，不是物理安全保证 |
| 4.3.5 | delta_score=delta_fractional/predictor score coefficient，使本步增加精确delta_f | thesis_release/innovation2/method/late_force_sampler.py 中的 _evaluate_exact_score | I2-F1 | — | SUPPORTED_IMPLEMENTATION | 系数非有限或过小时执行安全回退 |
| 4.3.5 | 只改position predictor，atomic/cell/stress/corrector不改 | thesis_release/innovation2/README.md；method/late_force_sampler.py；configs/frozen_method_definition.md | I2-F1 | 表4-2 | SUPPORTED_IMPLEMENTATION | 不支持多字段物理引导表述 |
| 4.3.5 | 20次触发时间为0.020至0.001 | thesis_release/innovation2/configs/formal32_config.yaml 中的 actual_trigger_model_t | I2-F1 | 表4-2 | SUPPORTED | 受1000步离散网格限制 |
| 4.4 | 阶段阈值、参考力、lambda=1、位移尺度在确认实验前冻结且无sweep | thesis_release/innovation2/configs/p0_guidance_config.yaml；configs/formal32_config.yaml；configs/formal256_implementation_lock.json | I2-F1 | 表4-2 | SUPPORTED_PROTOCOL | 不能把正式结果解释为后验调参 |
| 4.5.1 | P0 MaxF 0.304028→0.253392，16.66%，CI [10.17%,31.24%]，15/0/1 | thesis_release/innovation2/results/mattersim_late_force_guidance_p0/decision_summary.json；final_report.md | I2-F4 | 表4-3 | SUPPORTED_P0 | n=16，只作先导证据 |
| 4.5.1 | P0 MeanF、RMSD改善；E-hull +0.000809；Stable/NUS/Validity保持 | thesis_release/innovation2/results/mattersim_late_force_guidance_p0/final_report.md | — | 表4-3 | SUPPORTED_P0_GUARDRAIL | 全部为代理评价 |
| 4.5.1 | P0 320/320引导接受、0次安全回退、runtime ratio 0.9998 | thesis_release/innovation2/results/mattersim_late_force_guidance_p0/decision_summary.json；final_report.md | — | — | SUPPORTED | 运行时受硬件波动影响 |
| 4.5.2 | Formal32 MaxF 0.183472→0.134728，26.57%，CI [14.27%,52.86%]，32/0/0 | thesis_release/innovation2/results/mattersim_late_force_guidance_formal32/decision_summary.json；bootstrap_results.json | I2-F4 | 表4-3 | SUPPORTED | 独立32-seed cohort |
| 4.5.2 | Formal32 MeanF、RMSD改善；Property MAE轻微恶化；runtime ratio 1.00336 | thesis_release/innovation2/results/mattersim_late_force_guidance_formal32/final_report.md；bootstrap_results.json | — | 表4-3 | SUPPORTED_MIXED | 不声称属性改善或加速 |
| 4.6.1 | Formal256无丢种子，256/256配对完成 | thesis_release/innovation2/results/mattersim_late_force_guidance_formal256/final_report.md；decision_summary.json | I2-F2、I2-F3 | 表4-4 | SUPPORTED_PROTOCOL | 限注册cohort |
| 4.6.1 | MatterSim MaxF 0.226408→0.158911，29.81%，CI [21.42%,40.88%]，254/0/2 | thesis_release/innovation2/results/mattersim_late_force_guidance_formal256/paired_bootstrap.json；tables/formal256_main_results.csv | I2-F2、I2-F3 | 表4-4 | SUPPORTED | MatterSim surrogate；非DFT |
| 4.6.1 | Mean Force降低29.72%，CI [24.01%,36.96%]，255/0/1 | 同上 | I2-F2、I2-F3 | 表4-4 | SUPPORTED | MatterSim surrogate |
| 4.6.1 | RMSD降低14.17%，CI [4.54%,25.95%]，216/21/19 | 同上 | I2-F2、I2-F3 | 表4-4 | SUPPORTED | MatterSim relaxation proxy |
| 4.6.1 | Property MAE 0.009757→0.009868，恶化1.14%，95/0/161 | thesis_release/innovation2/results/mattersim_late_force_guidance_formal256/paired_bootstrap.json；tables/formal256_main_results.csv | I2-F6 | 表4-4 | NOT_SUPPORTED_PROPERTY_IMPROVEMENT | 必须披露不利方向 |
| 4.6.1 | Stable 80.86%不变，Validity 100%不变 | thesis_release/innovation2/results/mattersim_late_force_guidance_formal256/quality_metrics.csv；tables/formal256_main_results.csv | I2-F6 | 表4-4 | SUPPORTED_GUARDRAIL | 不代表全部质量属性改善 |
| 4.6.2 | P0/Formal32/Formal256的MaxF效应方向一致 | thesis_release/innovation2/tables/effect_scale_consistency.csv | I2-F4 | 表4-3、表4-4 | SUPPORTED | 不同cohort，不是时间序列 |
| 4.6.2 | Formal256 MatterSim P90/P95/max及high-MaxF比例下降 | thesis_release/innovation2/results/mattersim_late_force_guidance_formal256/quality_metrics.csv | I2-F3 | — | SUPPORTING_DESCRIPTIVE | 非预注册独立主检验 |
| 4.6.2 | E-hull增加0.000469，NUS增加0.390625 pp | thesis_release/innovation2/results/mattersim_late_force_guidance_formal256/paired_bootstrap.json；quality_metrics.csv | I2-F6 | 表4-4 | SUPPORTED_MIXED | E-hull轻微不利，NUS轻微有利 |
| 4.6.3 | 两臂均2000 score calls；RC-NFGD另有20次MatterSim调用；runtime ratio 0.9980 | thesis_release/combined_summary/compute_summary.csv；innovation2/results/.../quality_metrics.csv | — | — | SUPPORTED_COMPUTE | 不构成加速证明 |
| 4.7.1 | CHGNet为未参与力引导的独立代理评估器 | thesis_release/innovation2/results/chgnet_formal256/decision_summary.json；外部文献Deng et al. 2023 | I2-F5 | 表4-5 | SUPPORTED_INDEPENDENT_SURROGATE | 参与磁性护栏；训练数据可能重叠；非DFT |
| 4.7.1 | CHGNet MaxF降低14.36%，相对CI [4.32%,27.73%]，172/0/84 | thesis_release/innovation2/results/chgnet_formal256/paired_bootstrap.json；research_archive/innovation2/chgnet_validation/results/paired_statistics.csv | I2-F5 | 表4-5 | SUPPORTED_INDEPENDENT_SURROGATE | 效应弱于MatterSim且损失84例 |
| 4.7.1 | CHGNet Mean Force降低9.59%，相对CI [3.08%,19.10%]，169/0/87 | thesis_release/innovation2/tables/chgnet_validation.csv；research_archive/innovation2/chgnet_validation/results/paired_statistics.csv | I2-F5 | 表4-5 | SUPPORTED_INDEPENDENT_SURROGATE | 不是DFT或实验验证 |
| 4.7.2 | CHGNet MaxF P90/P95上升，Mean Force中位数上升，尾部表现混合 | research_archive/innovation2/chgnet_validation/results/distribution_statistics.csv；final_report.md | I2-F5 | 表4-5 | SUPPORTED_MIXED | 禁止“所有分位数改善” |
| 4.8.1 | G1真实方向优于G0/G2/G3/G4的描述性次序 | thesis_release/innovation2/results/ablations/mattersim_force_direction_ablation/final_report.md；decision_summary.json | — | 表4-6 | SUPPORTED_DIRECTION | 控制分支重放G1幅度，存在路径依赖 |
| 4.8.1 | G1 vs G0 MaxF降低30.48%，CI [24.66%,38.87%]，31/0/1 | 同上 | — | 表4-6 | SUPPORTED_DIRECTION | 不证明理论最优方向 |
| 4.8.1 | G2为随机零净平移，G3为原子打乱，G4为配对G1反向 | thesis_release/innovation2/method/generate_variant_pair.py；method/experimental_sampler.py | — | 表4-6 | SUPPORTED_IMPLEMENTATION | G4不是自身轨迹闭环反梯度 |
| 4.8.2 | T2去除范数饱和与硬cap后MatterSim Mean MaxF低于T1 | thesis_release/innovation2/results/ablations/mattersim_trust_region_ablation/final_report.md；research_archive/innovation2/bounded_correction/results/paired_statistics.csv | — | — | NOT_SUPPORTED_BOUND_AS_CORE | T2一次改变两个幅度机制 |
| 4.8.2 | T2的独立CHGNet MaxF未稳定优于T1 | research_archive/innovation2/bounded_correction/results/paired_statistics.csv | — | — | MIXED | T1–T2 CI跨零 |
| 4.8.2 | bound仅定位为保守步长/异常控制 | thesis_release/CLAIMS_AND_LIMITATIONS.md；innovation2/results/ablations/mattersim_trust_region_ablation/decision_summary.json | — | 表4-2 | CLAIM_BOUNDARY | 禁止称为已验证性能核心 |
| 4.8.3 | 等20次MatterSim调用时POST Mean MaxF 0.043275，优于F0 0.163650 | thesis_release/innovation2/results/ablations/mattersim_equal_budget_post_generation/final_report.md；decision_summary.json | — | — | SUPPORTED_COMPARATOR | n=32；评价为MatterSim |
| 4.8.3 | F0不能宣称优于等预算生成后修正 | thesis_release/CLAIMS_AND_LIMITATIONS.md | — | — | NOT_SUPPORTED_ONLINE_SUPERIORITY | 两方法研究目标不同 |
| 4.8.4 | Adaptive CFG + RC-NFGD兼容性未达到双保持阈值 | thesis_release/innovation2/results/ablations/adaptive_cfg_force_guidance_compatibility/decision_summary.json；final_report.md | — | — | FAIL_COMPATIBILITY | 最终模型为F0 standalone |
| 4.8.4、4.9 | 创新点1与2协同不受支持 | thesis_release/CLAIMS_AND_LIMITATIONS.md | — | — | NOT_SUPPORTED_SYNERGY | 两项贡献独立报告 |
| 4.9 | Innovation2最终状态SUPPORTED | thesis_release/innovation2/README.md；results/final_decision.json；research_archive/innovation2/final_rc_nfgd/README.md | I2-F1–I2-F6 | 表4-1–表4-6 | SUPPORTED | 仅支持冻结代理协议下主claim |
| 4.9 | DFT_VERIFIED=false | thesis_release/innovation2/results/final_decision.json；CLAIMS_AND_LIMITATIONS.md；research_archive/innovation2/dft_verification/README.md | — | — | FALSE_DFT_CLAIM | 禁止DFT validated、physically proven |
| 4.9 | 不支持SOTA、普适最优或优于全部替代方案 | thesis_release/CLAIMS_AND_LIMITATIONS.md；innovation2/results/final_report.md | — | — | CLAIM_BOUNDARY | 未进行文献全覆盖和普适基准验证 |

## 图表来源清单

| 论文编号 | Release ID | 文件 | 主要用途 | 边界 |
|---|---|---|---|---|
| 图4-1 | I2-F1 | thesis_release/innovation2/figures/I2_F1_rc_nfgd_pipeline.png | RC-NFGD在线流程 | 概念图，无性能数字 |
| 图4-2 | I2-F2 | thesis_release/innovation2/figures/I2_F2_formal256_primary_metrics.png | Formal256主指标 | MatterSim代理 |
| 图4-3 | I2-F3 | thesis_release/innovation2/figures/I2_F3_paired_improvement_distributions.png | 配对改善分布 | 尾部依赖cohort |
| 图4-4 | I2-F4 | thesis_release/innovation2/figures/I2_F4_effect_consistency.png | P0/Formal32/Formal256方向一致性 | 不同cohort非时间序列 |
| 图4-5 | I2-F6 | thesis_release/innovation2/figures/I2_F6_quality_guardrails.png | Stable/Validity/Property护栏 | Property MAE轻微恶化 |
| 图4-6 | I2-F5 | thesis_release/innovation2/figures/I2_F5_mattersim_vs_chgnet.png | MatterSim与CHGNet跨势比较 | 非DFT一致性或因果验证 |
| 表4-1 | — | 原始 diagnostic_report.md；diagnostic_correlations.csv | 阶段可靠性 | 代理相关性 |
| 表4-2 | — | thesis_release/innovation2/configs/formal32_config.yaml；frozen_method_definition.md | 冻结方法配置 | 限当前任务 |
| 表4-3 | — | P0/Formal32 decision_summary.json；effect_scale_consistency.csv | 小规模复现 | cohort较小 |
| 表4-4 | — | thesis_release/innovation2/tables/formal256_main_results.csv | Formal256完整主表 | 含不利Property MAE |
| 表4-5 | — | thesis_release/innovation2/tables/chgnet_validation.csv；archive paired_statistics.csv | 独立代理评估 | CHGNet非DFT |
| 表4-6 | — | force-direction ablation final_report.md | 方向消融 | G2/G3/G4轨迹重放限制 |

## 引用审计

| Reference | 核对来源 | 状态 |
|---|---|---|
| MatterGen | Nature 639, 624–632 (2025), DOI 10.1038/s41586-025-08628-5 | VERIFIED |
| DDPM | NeurIPS 2020 proceedings, Ho, Jain, Abbeel | VERIFIED |
| Score-SDE | ICLR 2021 / arXiv:2011.13456, Song et al. | VERIFIED |
| MatterSim | arXiv:2405.04967, Yang et al. | VERIFIED |
| CHGNet | Nature Machine Intelligence 5, 1031–1041 (2023), DOI 10.1038/s42256-023-00716-3 | VERIFIED |

## 最终一致性检查

- Unsupported SOTA claim: **NONE**
- DFT or physical-proof claim: **NONE**
- Hidden Property MAE degradation: **NO**
- False online-over-post claim: **NONE**
- False Innovation1+2 synergy claim: **NONE**
- Citation-needed marker count: **0**
- Unresolved evidence count: **0**
- CLAIM_EVIDENCE_CONSISTENCY: **PASS**
