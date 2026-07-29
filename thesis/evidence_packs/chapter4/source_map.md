# 第4章来源映射

| source_id | 类型 | 仓库相对路径 | commit | 支持内容 | 资格 |
| --- | --- | --- | --- | --- | --- |
| S01_MANIFEST | manifest | `thesis_archive/FINAL_EXPERIMENT_MANIFEST.json` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | all experiment identity, seed, n, branch and evidence qualification | authoritative archive manifest |
| S02_CLAIMS | report | `thesis/PAPER_CLAIMS_FINAL.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | C1–C6 frozen wording and scientific boundaries | authoritative writing claim register |
| S04_LINEAGE | report | `thesis_archive/EXPERIMENT_LINEAGE.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | method identity, branch lineage, negative-result boundaries | authoritative lineage narrative |
| S05_LIMITATIONS | report | `thesis/PAPER_LIMITATIONS.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | MatterSim surrogate, no DFT/property verification, leakage and cohort limits | mandatory disclosure |
| S06_ADAPTIVE_CONTROLLER | source code | `mattergen/diffusion/sampling/guidance_schedule.py` | `5de00419eea2d8a9be303638f2db8ece15a22366` | phase-specific EMA, multiplier, clipping and fallback | formal implementation snapshot |
| S07_ADAPTIVE_CFG | source code | `mattergen/diffusion/sampling/classifier_free_guidance.py` | `5de00419eea2d8a9be303638f2db8ece15a22366` | field residual RMS, joint cond/uncond forward and CFG lerp | formal implementation snapshot |
| S08_PC_SAMPLER | source code | `mattergen/diffusion/sampling/pc_sampler.py` | `5de00419eea2d8a9be303638f2db8ece15a22366` | corrector then predictor calls; full path retained | formal implementation snapshot |
| S09_ADAPTIVE_CONFIG | config | `thesis_archive/configs/adaptive_cfg_final.yaml` | `5de00419eea2d8a9be303638f2db8ece15a22366` | g0=2, alpha=.5, beta=.95, eps=1e-6, [0,5], FP32, B1, full PC | frozen configuration |
| S10_I1_DATA | per-seed data | `thesis_archive/data/innovation1/per_seed_metrics.csv` | `5de00419eea2d8a9be303638f2db8ece15a22366` | Adaptive CFG paired 256 metrics | formal independent data; seeds 20000–20255 |
| S11_I1_REPORT | report | `thesis_archive/reports/innovation1/formal_final_report.json` | `20255f1a857cd763a7ef2bf2f24c1889c98c4d1c` | Adaptive effect estimates, CI, tests and Corrector Gating No-Go | frozen formal report |
| S12_I1_FIGURE | figure | `thesis/figures/generated/pdf/fig05_adaptive_cfg_results.pdf` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | Adaptive paired E-hull and Stable/NUS effects | generated only from archived data |

正式commit内源码应使用 `git show <commit>:<relative_path>` 查看；不能只依赖当前工作树或README。
