# EXP-06 两创新相互作用

执行状态：COMPLETED_EXISTING；只补充审计和统计，不重复生成。

| Method   | Cohort   | Seed range    |   n |   Property MAE |      Hit |         MaxF |   Atomic Force |          RMSD |        E-hull |   Stable |    Novel |   Unique |      NUS |   Validity |     Runtime | Method label       |
|:---------|:---------|:--------------|----:|---------------:|---------:|-------------:|---------------:|--------------:|--------------:|---------:|---------:|---------:|---------:|-----------:|------------:|:-------------------|
| C0       | A5       | 744000-744063 |  64 | 0.01056134316  | 0.65625  | 0.2189222919 |  0.1037979554  | 0.05853177012 | 0.1203421311  | 0.671875 | 0.5625   |   0.8125 | 0.21875  |          1 | 113.586407  | Official MatterGen |
| A0       | A5       | 744000-744063 |  64 | 0.009705702117 | 0.6875   | 0.2396292507 |  0.1168202236  | 0.0491622024  | 0.09085319657 | 0.828125 | 0.453125 |   0.6875 | 0.265625 |          1 | 107.0054801 | Adaptive CFG       |
| B0       | A5       | 744000-744063 |  64 | 0.01015056249  | 0.671875 | 0.1718331202 |  0.08002332531 | 0.05540295111 | 0.1190488095  | 0.671875 | 0.5625   |   0.8125 | 0.21875  |          1 | 107.9443632 | RC-NFGD            |
| AB       | A5       | 744000-744063 |  64 | 0.009727538997 | 0.6875   | 0.1927939179 |  0.09133041418 | 0.05809121202 | 0.09084618242 | 0.828125 | 0.453125 |   0.6875 | 0.265625 |          1 | 118.0970499 | AB                 |

ADAPTIVE_CFG_PLUS_F0=NOT_COMPATIBLE，限定为“未通过冻结双收益保留率要求”，不等于所有指标全无改善。原 A5 执行完成、审计 PASS，科学判定 FAIL。

定义属性保留率为 (MAE_C0-MAE_AB)/(MAE_C0-MAE_A0)，力保留率为 (MaxF_C0-MaxF_AB)/(MaxF_C0-MaxF_B0)。本批分别 97.4479% 和 55.4870%；后一项低于冻结 70% 要求。AB 相对 C0 的属性、NUS 和 MaxF 仍有点估计改善，不能把组合负结果描述成“全面失败”。

A0 单独改善本批磁密度 MAE，但 MatterSim MaxF 均值上升；B0 单独降低 MaxF。两个分支作用目标不同。结果提示相互作用或耦合，但未完成因果分解。最终论文采用从 MatterGen 分出的两条独立方法主线，不要求串联成组合最终模型，不开展 rescue tuning。

本文所有数字由 `collect_results.py` 自动读取并按 seed 合并，不从提示词近似数抄写。完整精度见 [主结果 CSV](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/main_results.csv)、[逐种子数据](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/all_per_seed_metrics.csv)、[统计 CSV](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/paired_statistics.csv)。

历史判定、阈值和置信区间保存在 [原始判定归档](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/original_decisions.json)；原始来源、Git commit、SHA256、评价器和配置见 [复现清单](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/reproducibility_manifest.json)。创新点一原始统计来源：[Formal256 报告](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/sources/a367f85efe8027bf8ef2db41cc48ec2fd3d60582/thesis_archive/reports/innovation1/formal_final_report.json)。

统一边界：SURROGATE_PROPERTY_EVAL=True；DFT_VERIFIED=False。Stable/E-hull/力均为代理评价，不是实验合成或 DFT 真值证明。不同 cohort 不作绝对指标直接排名。
