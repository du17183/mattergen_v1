# 第6章来源映射

| source_id | 类型 | 仓库相对路径 | commit | 支持内容 | 资格 |
| --- | --- | --- | --- | --- | --- |
| S01_MANIFEST | manifest | `thesis_archive/FINAL_EXPERIMENT_MANIFEST.json` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | all experiment identity, seed, n, branch and evidence qualification | authoritative archive manifest |
| S02_CLAIMS | report | `thesis/PAPER_CLAIMS_FINAL.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | C1–C6 frozen wording and scientific boundaries | authoritative writing claim register |
| S03_DATA_DICTIONARY | report | `thesis_archive/DATA_DICTIONARY.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | archived column definitions and units | authoritative archive schema |
| S04_LINEAGE | report | `thesis_archive/EXPERIMENT_LINEAGE.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | method identity, branch lineage, negative-result boundaries | authoritative lineage narrative |
| S05_LIMITATIONS | report | `thesis/PAPER_LIMITATIONS.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | MatterSim surrogate, no DFT/property verification, leakage and cohort limits | mandatory disclosure |
| S11_I1_REPORT | report | `thesis_archive/reports/innovation1/formal_final_report.json` | `20255f1a857cd763a7ef2bf2f24c1889c98c4d1c` | Adaptive effect estimates, CI, tests and Corrector Gating No-Go | frozen formal report |
| S14_E3_FROZEN_CORE | source code | `research/q3_frozen64.py` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | formal feature extraction, learned/always arms, exact fallback and metrics | frozen core reused by formal runner |
| S15_E3_FORMAL_RUNNER | source code | `research/q3_formal256.py` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | formal contract, seed audit, paired statistics and mechanism checks | formal run snapshot; recorded code commit 5293b4b71be88b6663bbe349f3b57694a916835f |
| S19_GATE_MECHANISM | summary JSON | `reports/q3_e3_pcr/formal256/gate_mechanism_summary.json` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | coverage, harm, low-force harm, gain retention and displacement | formal mechanism evidence |
| S20_RANDOM_GATE | table | `reports/q3_e3_pcr/frozen64/random_gate_ablation.csv` | `b65f42a8792004c7c820e59fa4413e1310e06143` | five random gates at equal 42/64 coverage; result range and mean | frozen64 supplementary ablation, not formal256 |
| S21_COHORT1_DATA | per-seed data | `thesis_archive/data/compatibility_1/per_seed_metrics.csv` | `ba2303c284210fdae0a35bb0153a8ef3af45a54c` | first independent A0+E3-G cohort | independent; seeds 41000–41063 |
| S22_COHORT1_REPORT | report | `thesis_archive/reports/compatibility/final_report.md` | `e358ee39a8cdd2a061a18bfaddbe88316b455048` | cohort 1 effect, CI, p and quality | independent compatibility report |
| S23_COHORT2_DATA | per-seed data | `thesis_archive/data/compatibility_2/per_seed_metrics.csv` | `22e1db74a59476562f1f746cd4210b9420cbdf05` | second independent A0+E3-G cohort | fully independent; seeds 50000–50063 |
| S24_COHORT2_REPORT | report | `thesis_archive/reports/replication/final_report.md` | `85485bc956fce1cf7d01c55baaa92c0b69fd745e` | cohort 2 effect, CI, p, semantic W/T/L and quality | independent replication report |
| S25_LEAK_DATA | per-seed data | `thesis_archive/data/leakage_diagnostic/per_seed_metrics.csv` | `01e9b2c30e5c58e05eaae908ba291c518b977d03` | training-overlap versus held-out Gate safety | diagnostic only; mixed 256 invalid for independent claims |
| S26_LEAK_REPORT | report | `thesis_archive/reports/leakage_diagnostic/final_report.md` | `d5bf7d00ab51a2a0b319203443391e3463e7a91b` | Fisher test and leakage interpretation | diagnostic report; held-out 192 supplementary only |
| S32_NEGATIVE_RESULTS | report | `docs/experiments/negative_results_summary.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | representative and supplementary No-Go routes | archive-level synthesis; some original server reports not in GitHub |

正式commit内源码应使用 `git show <commit>:<relative_path>` 查看；不能只依赖当前工作树或README。
