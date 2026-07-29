# 第3章来源映射

| source_id | 类型 | 仓库相对路径 | commit | 支持内容 | 资格 |
| --- | --- | --- | --- | --- | --- |
| S01_MANIFEST | manifest | `thesis_archive/FINAL_EXPERIMENT_MANIFEST.json` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | all experiment identity, seed, n, branch and evidence qualification | authoritative archive manifest |
| S02_CLAIMS | report | `thesis/PAPER_CLAIMS_FINAL.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | C1–C6 frozen wording and scientific boundaries | authoritative writing claim register |
| S03_DATA_DICTIONARY | report | `thesis_archive/DATA_DICTIONARY.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | archived column definitions and units | authoritative archive schema |
| S04_LINEAGE | report | `thesis_archive/EXPERIMENT_LINEAGE.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | method identity, branch lineage, negative-result boundaries | authoritative lineage narrative |
| S05_LIMITATIONS | report | `thesis/PAPER_LIMITATIONS.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | MatterSim surrogate, no DFT/property verification, leakage and cohort limits | mandatory disclosure |
| S08_PC_SAMPLER | source code | `mattergen/diffusion/sampling/pc_sampler.py` | `5de00419eea2d8a9be303638f2db8ece15a22366` | corrector then predictor calls; full path retained | formal implementation snapshot |
| S09_ADAPTIVE_CONFIG | config | `thesis_archive/configs/adaptive_cfg_final.yaml` | `5de00419eea2d8a9be303638f2db8ece15a22366` | g0=2, alpha=.5, beta=.95, eps=1e-6, [0,5], FP32, B1, full PC | frozen configuration |
| S10_I1_DATA | per-seed data | `thesis_archive/data/innovation1/per_seed_metrics.csv` | `5de00419eea2d8a9be303638f2db8ece15a22366` | Adaptive CFG paired 256 metrics | formal independent data; seeds 20000–20255 |
| S14_E3_FROZEN_CORE | source code | `research/q3_frozen64.py` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | formal feature extraction, learned/always arms, exact fallback and metrics | frozen core reused by formal runner |
| S15_E3_FORMAL_RUNNER | source code | `research/q3_formal256.py` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | formal contract, seed audit, paired statistics and mechanism checks | formal run snapshot; recorded code commit 5293b4b71be88b6663bbe349f3b57694a916835f |
| S17_I2_DATA | per-seed data | `thesis_archive/data/innovation2/per_seed_metrics.csv` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | C0/E3-A/E3-G paired 256 metrics and gate behavior | formal independent data; seeds 40000–40255 |
| S21_COHORT1_DATA | per-seed data | `thesis_archive/data/compatibility_1/per_seed_metrics.csv` | `ba2303c284210fdae0a35bb0153a8ef3af45a54c` | first independent A0+E3-G cohort | independent; seeds 41000–41063 |
| S23_COHORT2_DATA | per-seed data | `thesis_archive/data/compatibility_2/per_seed_metrics.csv` | `22e1db74a59476562f1f746cd4210b9420cbdf05` | second independent A0+E3-G cohort | fully independent; seeds 50000–50063 |
| S25_LEAK_DATA | per-seed data | `thesis_archive/data/leakage_diagnostic/per_seed_metrics.csv` | `01e9b2c30e5c58e05eaae908ba291c518b977d03` | training-overlap versus held-out Gate safety | diagnostic only; mixed 256 invalid for independent claims |
| S27_EVAL_STRUCTURE | source code | `mattergen/evaluation/metrics/structure.py` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | novel, unique, composition and structure validity | official evaluator implementation in repository snapshot |
| S28_EVAL_ENERGY | source code | `mattergen/evaluation/metrics/energy.py` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | E-hull, Stable and NUS | official evaluator implementation in repository snapshot |
| S29_EVAL_RMSD | source code | `mattergen/evaluation/utils/metrics_structure_summary.py` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | relaxation RMSD source structures | official evaluator implementation in repository snapshot |
| S30_RMSD_UTIL | source code | `mattergen/evaluation/utils/utils.py` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | RMSDStructureMatcher conversion to angstrom | official evaluator implementation in repository snapshot |
| S31_BASE_CONDITION_CONFIG | config | `configs/q3_e3_pcr_frozen64.json` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | dft_mag_density target 0.1 and immutable refinement fields | frozen evaluation config |

正式commit内源码应使用 `git show <commit>:<relative_path>` 查看；不能只依赖当前工作树或README。
