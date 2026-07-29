# Master Source Index

| source_id | 类型 | 相对路径 | commit | 用于章节 | 用于结论 | 数据资格 |
| --- | --- | --- | --- | --- | --- | --- |
| S01_MANIFEST | manifest | `thesis_archive/FINAL_EXPERIMENT_MANIFEST.json` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | 3,4,5,6 | all experiment identity, seed, n, branch and evidence qualification | authoritative archive manifest |
| S02_CLAIMS | report | `thesis/PAPER_CLAIMS_FINAL.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | 3,4,5,6 | C1–C6 frozen wording and scientific boundaries | authoritative writing claim register |
| S03_DATA_DICTIONARY | report | `thesis_archive/DATA_DICTIONARY.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | 3,5,6 | archived column definitions and units | authoritative archive schema |
| S04_LINEAGE | report | `thesis_archive/EXPERIMENT_LINEAGE.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | 3,4,5,6 | method identity, branch lineage, negative-result boundaries | authoritative lineage narrative |
| S05_LIMITATIONS | report | `thesis/PAPER_LIMITATIONS.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | 3,4,5,6 | MatterSim surrogate, no DFT/property verification, leakage and cohort limits | mandatory disclosure |
| S06_ADAPTIVE_CONTROLLER | source code | `mattergen/diffusion/sampling/guidance_schedule.py` | `5de00419eea2d8a9be303638f2db8ece15a22366` | 4 | phase-specific EMA, multiplier, clipping and fallback | formal implementation snapshot |
| S07_ADAPTIVE_CFG | source code | `mattergen/diffusion/sampling/classifier_free_guidance.py` | `5de00419eea2d8a9be303638f2db8ece15a22366` | 4 | field residual RMS, joint cond/uncond forward and CFG lerp | formal implementation snapshot |
| S08_PC_SAMPLER | source code | `mattergen/diffusion/sampling/pc_sampler.py` | `5de00419eea2d8a9be303638f2db8ece15a22366` | 3,4 | corrector then predictor calls; full path retained | formal implementation snapshot |
| S09_ADAPTIVE_CONFIG | config | `thesis_archive/configs/adaptive_cfg_final.yaml` | `5de00419eea2d8a9be303638f2db8ece15a22366` | 3,4 | g0=2, alpha=.5, beta=.95, eps=1e-6, [0,5], FP32, B1, full PC | frozen configuration |
| S10_I1_DATA | per-seed data | `thesis_archive/data/innovation1/per_seed_metrics.csv` | `5de00419eea2d8a9be303638f2db8ece15a22366` | 3,4 | Adaptive CFG paired 256 metrics | formal independent data; seeds 20000–20255 |
| S11_I1_REPORT | report | `thesis_archive/reports/innovation1/formal_final_report.json` | `20255f1a857cd763a7ef2bf2f24c1889c98c4d1c` | 4,6 | Adaptive effect estimates, CI, tests and Corrector Gating No-Go | frozen formal report |
| S12_I1_FIGURE | figure | `thesis/figures/generated/pdf/fig05_adaptive_cfg_results.pdf` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | 4 | Adaptive paired E-hull and Stable/NUS effects | generated only from archived data |
| S13_E3_REFINER | source code | `research/postgen_fastgate/refiner_eval.py` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | 5 | 14 features, Gate training, position proposal, safety, backtracking | available at formal snapshot; frozen source identity b65f42a8792004c7c820e59fa4413e1310e06143 |
| S14_E3_FROZEN_CORE | source code | `research/q3_frozen64.py` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | 3,5,6 | formal feature extraction, learned/always arms, exact fallback and metrics | frozen core reused by formal runner |
| S15_E3_FORMAL_RUNNER | source code | `research/q3_formal256.py` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | 3,5,6 | formal contract, seed audit, paired statistics and mechanism checks | formal run snapshot; recorded code commit 5293b4b71be88b6663bbe349f3b57694a916835f |
| S16_E3_CONFIG | config | `thesis_archive/configs/e3_pcr_final.yaml` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | 5 | 129 parameters, 14→8→1, threshold, trust and backtracking constants | frozen configuration |
| S17_I2_DATA | per-seed data | `thesis_archive/data/innovation2/per_seed_metrics.csv` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | 3,5 | C0/E3-A/E3-G paired 256 metrics and gate behavior | formal independent data; seeds 40000–40255 |
| S18_I2_REPORT | report | `thesis_archive/reports/innovation2/final_report.md` | `41479015c5c3edc389601c4b7cc44a6db5e115cd` | 5 | E3-PCR force, quality and sensitivity results | frozen formal report |
| S19_GATE_MECHANISM | summary JSON | `reports/q3_e3_pcr/formal256/gate_mechanism_summary.json` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | 5,6 | coverage, harm, low-force harm, gain retention and displacement | formal mechanism evidence |
| S20_RANDOM_GATE | table | `reports/q3_e3_pcr/frozen64/random_gate_ablation.csv` | `b65f42a8792004c7c820e59fa4413e1310e06143` | 5,6 | five random gates at equal 42/64 coverage; result range and mean | frozen64 supplementary ablation, not formal256 |
| S21_COHORT1_DATA | per-seed data | `thesis_archive/data/compatibility_1/per_seed_metrics.csv` | `ba2303c284210fdae0a35bb0153a8ef3af45a54c` | 3,6 | first independent A0+E3-G cohort | independent; seeds 41000–41063 |
| S22_COHORT1_REPORT | report | `thesis_archive/reports/compatibility/final_report.md` | `e358ee39a8cdd2a061a18bfaddbe88316b455048` | 6 | cohort 1 effect, CI, p and quality | independent compatibility report |
| S23_COHORT2_DATA | per-seed data | `thesis_archive/data/compatibility_2/per_seed_metrics.csv` | `22e1db74a59476562f1f746cd4210b9420cbdf05` | 3,6 | second independent A0+E3-G cohort | fully independent; seeds 50000–50063 |
| S24_COHORT2_REPORT | report | `thesis_archive/reports/replication/final_report.md` | `85485bc956fce1cf7d01c55baaa92c0b69fd745e` | 6 | cohort 2 effect, CI, p, semantic W/T/L and quality | independent replication report |
| S25_LEAK_DATA | per-seed data | `thesis_archive/data/leakage_diagnostic/per_seed_metrics.csv` | `01e9b2c30e5c58e05eaae908ba291c518b977d03` | 3,6 | training-overlap versus held-out Gate safety | diagnostic only; mixed 256 invalid for independent claims |
| S26_LEAK_REPORT | report | `thesis_archive/reports/leakage_diagnostic/final_report.md` | `d5bf7d00ab51a2a0b319203443391e3463e7a91b` | 6 | Fisher test and leakage interpretation | diagnostic report; held-out 192 supplementary only |
| S27_EVAL_STRUCTURE | source code | `mattergen/evaluation/metrics/structure.py` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | 3 | novel, unique, composition and structure validity | official evaluator implementation in repository snapshot |
| S28_EVAL_ENERGY | source code | `mattergen/evaluation/metrics/energy.py` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | 3 | E-hull, Stable and NUS | official evaluator implementation in repository snapshot |
| S29_EVAL_RMSD | source code | `mattergen/evaluation/utils/metrics_structure_summary.py` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | 3 | relaxation RMSD source structures | official evaluator implementation in repository snapshot |
| S30_RMSD_UTIL | source code | `mattergen/evaluation/utils/utils.py` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | 3 | RMSDStructureMatcher conversion to angstrom | official evaluator implementation in repository snapshot |
| S31_BASE_CONDITION_CONFIG | config | `configs/q3_e3_pcr_frozen64.json` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | 3,5 | dft_mag_density target 0.1 and immutable refinement fields | frozen evaluation config |
| S32_NEGATIVE_RESULTS | report | `docs/experiments/negative_results_summary.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | 6 | representative and supplementary No-Go routes | archive-level synthesis; some original server reports not in GitHub |

所有路径均为仓库相对路径。正式源码事实必须同时使用路径和commit。
