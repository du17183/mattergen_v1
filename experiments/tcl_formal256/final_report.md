# TCL FORMAL256 FINAL: FAIL

`DFT_VERIFIED=False`

本报告只描述冻结 MatterGen checkpoint、MatterSim-5M 和项目官方 Alex-MP 参考评价的结果。E-hull、Stable、force 与 relaxation 均为 MatterSim/项目代理指标，不是 DFT 验证结果。

## 1. 实验身份与执行完整性

| 项目 | 结果 |
|---|---|
| Branch | `experiment/tcl-formal256` |
| Final HEAD | 本报告所在的最终结果 commit；精确 SHA 见交付消息中的 `git rev-parse HEAD` |
| P2 parent | `b9fd99019657df7c605976255ef07d774b61f4f9` |
| Preregistration commit | `3b09c1ab289b8551f65221f53c9c30f3b7d11822` |
| Formal seed range | `83000–83255`（连续 256 paired seeds） |
| Historical overlap | none |
| Formal training steps | 0 |
| C0 generation | 256/256（100%） |
| FT0 generation | 256/256（100%） |
| TCL generation | 256/256（100%） |
| C0 MatterSim | 256/256（100%） |
| FT0 MatterSim | 256/256（100%） |
| TCL MatterSim | 256/256（100%） |
| Technical reruns / replacements | 0 / 0 |
| DFT verification | `False` |

冻结 checkpoint SHA-256：

- C0 base `last.ckpt`: `01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e`
- FT0 `model.pt`: `1cd254568463ae4d7774193fa0a0e97925cdad3b6c505c1e5f5dd4e7968f6960`
- TCL `model.pt`: `8dd9879f11edea7f25f587ae72f928a6f3ab2e5c81fc9400c13642364ad90443`

采样为 1000 steps、corrector 1、CFG 2.0、目标 `dft_mag_density=0.1`、batch size 1。生成真实文件的时间跨度为 20,836.4 s（约 5 h 47 min）；各样本记录耗时均值 C0/FT0/TCL 分别为 117.277/116.812/116.860 s。MatterSim-5M 使用 `fmax=0.05`，三组分别在 GPU 0/1/2 执行；正式质量评价复用这些松弛结果并按冻结 P1 管线在 CPU 上使用同一 Alex-MP 参考。

## 2. 256-seed absolute quality

### 2.1 E-hull（eV/atom，越低越好）

| Method | Mean | Median | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 0.112332 | 0.089138 | 0.217593 | 0.281351 | 0.492795 | 1.337847 |
| FT0 | 0.151019 | 0.133035 | 0.286929 | 0.318499 | 0.479347 | 0.515808 |
| TCL | 0.114443 | 0.096450 | 0.234420 | 0.275644 | 0.445646 | 0.927617 |

### 2.2 Stable / NUS / Novel / Unique

| Method | Stable | NUS | Novel | Unique |
|---|---:|---:|---:|---:|
| C0 | 142/256 (55.47%) | 82/256 (32.03%) | 193/256 (75.39%) | 252/256 (98.44%) |
| FT0 | 89/256 (34.77%) | 32/256 (12.50%) | 192/256 (75.00%) | 228/256 (89.06%) |
| TCL | 131/256 (51.17%) | 73/256 (28.52%) | 196/256 (76.56%) | 253/256 (98.83%) |

### 2.3 RMSD（Å，越低越好）

| Method | Mean | Median | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 0.049300 | 0.020270 | 0.113653 | 0.175984 | 0.389339 | 1.685046 |
| FT0 | 0.115658 | 0.041076 | 0.269372 | 0.489682 | 1.242785 | 1.515905 |
| TCL | 0.082882 | 0.025116 | 0.156276 | 0.399002 | 1.137961 | 1.367040 |

### 2.4 Atomic force（eV/Å，越低越好）

| Method | Mean | Median | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| C0 | 0.148648 | 0.086821 | 0.449401 | 0.816486 | 1.341585 |
| FT0 | 0.301859 | 0.234867 | 0.790991 | 1.377514 | 4.619371 |
| TCL | 0.182402 | 0.102990 | 0.620748 | 1.118313 | 1.603575 |

### 2.5 Maximum atomic force per structure（eV/Å，越低越好）

| Method | Mean | Median | P90 | P95 | P99 | Max | >1 | >2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.337212 | 0.160266 | 0.824932 | 1.048538 | 2.200286 | 4.243141 | 16 (6.25%) | 3 (1.17%) |
| FT0 | 0.600446 | 0.419229 | 1.072391 | 1.538754 | 3.945332 | 8.433249 | 30 (11.72%) | 10 (3.91%) |
| TCL | 0.387907 | 0.187198 | 0.883580 | 1.406524 | 2.413364 | 5.597732 | 24 (9.38%) | 5 (1.95%) |

### 2.6 Relaxation steps（越低越好）

| Method | Mean | Median | P90 | P95 | P99 | Max | >100 | >200 | >300 | >400 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 47.082 | 28.5 | 88.0 | 142.5 | 395.1 | 656 | 18 (7.03%) | 8 (3.12%) | 5 (1.95%) | 3 (1.17%) |
| FT0 | 81.887 | 40.0 | 213.5 | 295.0 | 650.0 | 1043 | 45 (17.58%) | 27 (10.55%) | 13 (5.08%) | 9 (3.52%) |
| TCL | 62.750 | 33.5 | 109.0 | 184.25 | 551.7 | 1321 | 27 (10.55%) | 12 (4.69%) | 10 (3.91%) | 8 (3.12%) |

完整机器可读 absolute 表见 `quality_results.csv`。

## 3. Question A：TCL 是否优于 FT0？

20,000 次 paired bootstrap，seed `20260912`。Delta 定义为 TCL−FT0；E-hull/RMSD/force/steps 负值有利，rate 正值有利。

| Endpoint | Delta | 95% CI | W/T/L | Raw one-sided p | Holm result |
|---|---:|---:|---:|---:|---|
| E-hull mean (primary) | -0.036576 eV/atom | [-0.053110, -0.020384] | 155/0/101 | 0.000050 | SUPPORTED, adjusted p=0.000250 |
| Stable | +16.406 pp | [8.594, 24.219] | 76/146/34 | 0.000050 | secondary |
| NUS (primary) | +16.016 pp | [9.375, 22.656] | 63/171/22 | 0.000050 | SUPPORTED, adjusted p=0.000250 |
| Novel | +1.562 pp | [-5.469, 8.594] | 46/168/42 | 0.352782 | secondary, unsupported |
| Unique | +9.766 pp | [5.859, 13.672] | 27/227/2 | 0.000050 | secondary |
| RMSD mean (primary) | -0.032777 Å | [-0.067411, 0.000653] | 150/0/106 | 0.032798 | SUPPORTED, adjusted p=0.032798 |
| Atomic-force mean (primary) | -0.119456 eV/Å | [-0.172239, -0.073327] | 183/0/73 | 0.000200 | SUPPORTED, adjusted p=0.000600 |
| Structure max-force mean (primary) | -0.212539 eV/Å | [-0.333815, -0.100624] | 171/0/85 | 0.000650 | SUPPORTED, adjusted p=0.001300 |
| Max-force >1 | -2.344 pp | [-7.812, 2.734] | 27/208/21 | 0.212739 | secondary, unsupported |
| Max-force >2 | -1.953 pp | [-5.078, 0.781] | 10/241/5 | 0.121644 | secondary, unsupported |
| Relaxation steps mean | -19.137 | [-38.152, 0.090] | 164/3/89 | 0.024249 | secondary |
| Max-force P95 | -0.132229 eV/Å | [-0.911381, 0.250417] | N/A | N/A | secondary quantile |
| Max-force P99 | -1.531968 eV/Å | [-5.182175, 1.062370] | N/A | N/A | secondary quantile |

五个 primary endpoint 的 Holm 结论全部为 `SUPPORTED`。因此 P1/P2 的 TCL>FT0 核心 signal 在全新 256 paired seeds 上正式第三次复现；E-hull、NUS、RMSD、atomic-force mean、structure max-force mean 五个方向全部有利。

## 4. Question B：TCL 是否守住 C0？

以下 margins 是查看 Formal256 outcomes 之前冻结的本研究 engineering/scientific tolerances，不是材料领域 universal standards。

| NI endpoint | TCL | C0 | Delta/ratio | 95% CI | Frozen margin / rule | Result |
|---|---:|---:|---:|---:|---|---|
| E-hull mean | 0.114443 | 0.112332 | +0.002111 | [-0.017894, 0.021309] | upper CI < +0.025 | PASS |
| Stable | 51.17% | 55.47% | -4.297 pp | [-12.500, 3.906] | lower CI > -10 pp | NOT ESTABLISHED |
| NUS | 28.52% | 32.03% | -3.516 pp | [-11.719, 4.297] | lower CI > -10 pp | NOT ESTABLISHED |
| RMSD mean | 0.082882 | 0.049300 | +0.033581 Å | [0.007844, 0.060115] | upper CI < +0.020 Å | NOT ESTABLISHED |
| Atomic-force mean | 0.182402 | 0.148648 | ratio 1.227077 | [1.026305, 1.468996] | upper CI < 1.20 | NOT ESTABLISHED |
| Structure max-force mean | 0.387907 | 0.337212 | ratio 1.150335 | [0.917018, 1.445893] | upper CI < 1.20 | NOT ESTABLISHED |
| Structure max-force P95 | 1.406524 | 1.048538 | ratio 1.341415 | [0.894937, 1.666662] | upper CI < 1.30 | NOT ESTABLISHED |
| Structure max-force P99 | 2.413364 | 2.200286 | ratio 1.096841 | [0.483372, 2.419155] | upper CI < 1.50 | NOT ESTABLISHED |
| Max-force >1 | 9.38% | 6.25% | +3.125 pp | [-1.172, 7.422] | upper CI < +5 pp | NOT ESTABLISHED |
| Max-force >2 | 1.95% | 1.17% | +0.781 pp | [-1.172, 3.125] | upper CI < +2 pp | NOT ESTABLISHED |

主要 non-inferiority **没有建立**：10 个预注册端点仅 E-hull mean 通过。Stable/NUS 等若只是 CI 略宽尚可视为精度问题，但 RMSD mean 的 CI 完全高于 0，atomic-force ratio 的 CI 完全高于 1，构成相对 C0 的可信系统性 geometry/force 恶化证据，而不是单纯不确定。

## 5. Robustness 与所有严重 outlier seeds

阈值冻结为 RMSD>0.5 Å、structure max-force>1/>2 eV/Å、relaxation steps>200/>300/>400。所有样本均保留。

| Method | Unique outlier seeds | RMSD>0.5 | Force>1 | Force>2 | Steps>200 | >300 | >400 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 18 | 1 | 16 | 3 | 8 | 5 | 3 |
| FT0 | 50 | 13 | 30 | 10 | 27 | 13 | 9 |
| TCL | 30 | 10 | 24 | 5 | 12 | 10 | 8 |

完整 method-specific outlier seed 清单：

- C0（18）：83014, 83018, 83049, 83057, 83099, 83129, 83140, 83156, 83163, 83166, 83171, 83198, 83214, 83224, 83238, 83241, 83247, 83254。
- FT0（50）：83003, 83008, 83014, 83026, 83029, 83039, 83051, 83057, 83064, 83068, 83069, 83070, 83072, 83081, 83084, 83086, 83087, 83088, 83089, 83093, 83094, 83102, 83105, 83109, 83117, 83121, 83129, 83132, 83138, 83145, 83148, 83156, 83163, 83164, 83166, 83173, 83174, 83185, 83187, 83191, 83194, 83196, 83219, 83233, 83238, 83240, 83241, 83243, 83251, 83254。
- TCL（30）：83008, 83015, 83023, 83043, 83049, 83068, 83069, 83075, 83077, 83090, 83094, 83104, 83109, 83110, 83118, 83127, 83129, 83130, 83135, 83138, 83150, 83176, 83180, 83182, 83202, 83203, 83214, 83215, 83233, 83249。

逐条 formula、RMSD、max force、steps 和复合 flag 共 98 行，完整保存在 `tail_analysis.csv`，上面的分组 seed 清单无遗漏。

- Novel/Unique：无 collapse。TCL 分别为 76.56%/98.83%，均不低于 C0 的 75.39%/98.44%。
- Geometry tail：不安全。TCL 的 RMSD>0.5 Å 为 10/256，对 C0 为 1/256；P95/P99 为 0.399/1.138 Å，对 C0 为 0.176/0.389 Å；RMSD mean 的 TCL−C0 CI 也排除 0 并指向恶化。
- Force tail：相对 FT0 改善，但没有守住 C0。TCL 的 force>1/>2 为 24/5，对 C0 为 16/3，P95 为 1.407 对 1.049 eV/Å。比例/分位数 CI 较宽，不能单独宣称明确 tail collapse，但同 atomic-force mean 的可信恶化共同构成风险。
- Relaxation tail：TCL 比 FT0 好，但比 C0 重；TCL >300/>400 为 10/8，对 C0 为 5/3，且 TCL max=1321、C0 max=656。

## 6. P0/P1/P2/Formal replication

Delta 均为 TCL−FT0；低优指标负值为有利，高优 rate 正值为有利。

| Metric | P0 | P1 | P2 | Formal delta [95% CI] | Replication |
|---|---:|---:|---:|---:|---|
| E-hull | -0.009535 | -0.017797 | -0.038217 | -0.036576 [-0.053110, -0.020384] | yes_all_four |
| Stable (pp) | 0.000 | +28.125 | +14.062 | +16.406 [8.594, 24.219] | yes_p1_p2_formal |
| NUS (pp) | +25.000 | +21.875 | +23.438 | +16.016 [9.375, 22.656] | yes_all_four |
| Novel (pp) | +25.000 | -9.375 | +10.938 | +1.562 [-5.469, 8.594] | yes_p2_to_formal_only |
| Unique (pp) | 0.000 | +3.125 | +6.250 | +9.766 [5.859, 13.672] | yes_p1_p2_formal |
| RMSD | -0.019737 | -0.026404 | -0.047980 | -0.032777 [-0.067411, 0.000653] | yes_all_four |
| Atomic-force mean | -0.083695 | -0.092882 | -0.153130 | -0.119456 [-0.172239, -0.073327] | yes_all_four |
| Structure max-force mean | -0.242409 | -0.151221 | -0.396061 | -0.212539 [-0.333815, -0.100624] | yes_all_four |
| Relaxation steps | -9.125 | -22.156 | -23.375 | -19.137 [-38.152, 0.090] | yes_all_four |

结论：TCL 相对 FT0 的 P1/P2 signal 已正式复现，而且五个 primary endpoints 均通过预注册 Holm inference；失败并非来自 TCL 对 FT0 无效。

## 7. 最终判定与建议

**最终结论：FAIL。**

原因严格对应冻结 FAIL 规则：

1. TCL 相对 FT0 的效果强且复现，因此不满足“signal 消失”型失败。
2. 但 TCL 相对 C0 的主体 non-inferiority 没有建立：10 项仅 1 项 PASS，不是 BORDERLINE 所述的“少数 CI 较宽”。
3. RMSD mean `+0.033581 Å`，95% CI `[+0.007844,+0.060115]`，相对 C0 有可信 geometry degradation；RMSD>0.5 Å 频率 10/256 对 1/256，P99 明显增大。
4. Atomic-force mean ratio `1.227`，95% CI `[1.026,1.469]`，相对 C0 有可信 force degradation；max-force tail 也整体偏重，虽其单项 CI 较宽。
5. Novel/Unique 没有 mode collapse，这一项通过，但不能抵消 C0 geometry/force robustness 失败。

因此**不推荐**“冻结 TCL 为硕士论文最终第二创新”。按预注册规则应停止当前 TCL 版本，并永久锁定 Formal seeds `83000–83255`；不得在这些 seeds 上调参、删 outlier 或再次声称独立 formal。当前不启动 DML、TCL+DML、TCL V2、第二个 Formal256 或任何自动调参。

若后续另立开发路线，应只在全新的 development seeds 上针对 C0 geometry/force 保真分析；新的方法冻结后必须使用另一组从未看过的 formal seeds。该建议不改变本次 `FAIL` 结论。

## 8. 结果文件

- `quality_results.csv`：三组完整 absolute quality。
- `superiority_results.csv`：TCL−FT0 全部 superiority endpoints。
- `paired_statistics.csv`：成对 bootstrap 原始统计。
- `noninferiority_results.csv`：10 个冻结 C0 NI endpoints。
- `tail_analysis.csv`：98 条严重 outlier 逐 seed 明细。
- `replication_table.csv`：P0/P1/P2/Formal 复现表。
- `generation_results.csv`：768 条生成结果与 checkpoint 哈希。
