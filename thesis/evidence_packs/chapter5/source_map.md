# 第5章来源映射

| source_id | 类型 | 仓库相对路径 | commit | 支持内容 | 资格 |
| --- | --- | --- | --- | --- | --- |
| S01_MANIFEST | manifest | `thesis_archive/FINAL_EXPERIMENT_MANIFEST.json` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | all experiment identity, seed, n, branch and evidence qualification | authoritative archive manifest |
| S02_CLAIMS | report | `thesis/PAPER_CLAIMS_FINAL.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | C1–C6 frozen wording and scientific boundaries | authoritative writing claim register |
| S03_DATA_DICTIONARY | report | `thesis_archive/DATA_DICTIONARY.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | archived column definitions and units | authoritative archive schema |
| S04_LINEAGE | report | `thesis_archive/EXPERIMENT_LINEAGE.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | method identity, branch lineage, negative-result boundaries | authoritative lineage narrative |
| S05_LIMITATIONS | report | `thesis/PAPER_LIMITATIONS.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | MatterSim surrogate, no DFT/property verification, leakage and cohort limits | mandatory disclosure |
| S13_E3_REFINER | source code | `research/postgen_fastgate/refiner_eval.py` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | 14 features, Gate training, position proposal, safety, backtracking | available at formal snapshot; frozen source identity b65f42a8792004c7c820e59fa4413e1310e06143 |
| S14_E3_FROZEN_CORE | source code | `research/q3_frozen64.py` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | formal feature extraction, learned/always arms, exact fallback and metrics | frozen core reused by formal runner |
| S15_E3_FORMAL_RUNNER | source code | `research/q3_formal256.py` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | formal contract, seed audit, paired statistics and mechanism checks | formal run snapshot; recorded code commit 5293b4b71be88b6663bbe349f3b57694a916835f |
| S16_E3_CONFIG | config | `thesis_archive/configs/e3_pcr_final.yaml` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | 129 parameters, 14→8→1, threshold, trust and backtracking constants | frozen configuration |
| S17_I2_DATA | per-seed data | `thesis_archive/data/innovation2/per_seed_metrics.csv` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | C0/E3-A/E3-G paired 256 metrics and gate behavior | formal independent data; seeds 40000–40255 |
| S18_I2_REPORT | report | `thesis_archive/reports/innovation2/final_report.md` | `41479015c5c3edc389601c4b7cc44a6db5e115cd` | E3-PCR force, quality and sensitivity results | frozen formal report |
| S19_GATE_MECHANISM | summary JSON | `reports/q3_e3_pcr/formal256/gate_mechanism_summary.json` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | coverage, harm, low-force harm, gain retention and displacement | formal mechanism evidence |
| S20_RANDOM_GATE | table | `reports/q3_e3_pcr/frozen64/random_gate_ablation.csv` | `b65f42a8792004c7c820e59fa4413e1310e06143` | five random gates at equal 42/64 coverage; result range and mean | frozen64 supplementary ablation, not formal256 |
| S31_BASE_CONDITION_CONFIG | config | `configs/q3_e3_pcr_frozen64.json` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | dft_mag_density target 0.1 and immutable refinement fields | frozen evaluation config |

正式commit内源码应使用 `git show <commit>:<relative_path>` 查看；不能只依赖当前工作树或README。
