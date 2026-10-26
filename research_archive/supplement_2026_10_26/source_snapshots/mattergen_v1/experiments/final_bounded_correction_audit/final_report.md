# EXP-05 有界修正消融

执行状态：COMPLETED_EXISTING；只补充审计和统计，不重复生成。

| Method   | Cohort   | Seed range    |   n |   Property MAE |   Hit |          MaxF |   Atomic Force |           RMSD |        E-hull |   Stable |   Novel |   Unique |   NUS |   Validity |     Runtime | Method label   |
|:---------|:---------|:--------------|----:|---------------:|------:|--------------:|---------------:|---------------:|--------------:|---------:|--------:|---------:|------:|-----------:|------------:|:---------------|
| T0       | A4       | 742000-742031 |  32 | 0.009207650988 | 0.625 | 0.2771384391  |  0.1000159395  | 0.01460213665  | 0.07689125473 |     0.75 |   0.375 |  0.71875 | 0.125 |          1 | 84.49223987 | T0             |
| T1       | A4       | 742000-742031 |  32 | 0.009233885604 | 0.625 | 0.2315021824  |  0.07690814055 | 0.01292418678  | 0.07686946298 |     0.75 |   0.375 |  0.71875 | 0.125 |          1 | 85.31104014 | T1             |
| T2       | A4       | 742000-742031 |  32 | 0.009425633173 | 0.625 | 0.06083426119 |  0.03028154693 | 0.005175944986 | 0.0768602126  |     0.75 |   0.375 |  0.71875 | 0.125 |          1 | 84.71428832 | T2             |

| baseline   | method   | metric           |   n |   baseline_mean |   method_mean |   delta_method_minus_baseline |   paired_median_delta |   delta_ci95_low |   delta_ci95_high |   wins |   ties |   losses |
|:-----------|:---------|:-----------------|----:|----------------:|--------------:|------------------------------:|----------------------:|-----------------:|------------------:|-------:|-------:|---------:|
| T0         | T1       | MaxF             |  32 |    0.2771384391 | 0.2315021824  |               -0.04563625663  |       -0.04587053722  |   -0.06125353272 |  -0.03419560641   |     32 |      0 |        0 |
| T0         | T1       | Independent MaxF |  32 |    0.1476795066 | 0.1413343544  |               -0.006345152285 |       -0.005782496691 |   -0.01277049441 |  -0.0002307575049 |     22 |      0 |       10 |
| T0         | T2       | MaxF             |  32 |    0.2771384391 | 0.06083426119 |               -0.2163041779   |       -0.04634514398  |   -0.4851938262  |  -0.06306857      |     32 |      0 |        0 |
| T0         | T2       | Independent MaxF |  32 |    0.1476795066 | 0.1489696311  |                0.001290124488 |       -0.002002915841 |   -0.02341950832 |   0.02827847068   |     19 |      0 |       13 |
| T1         | T2       | MaxF             |  32 |    0.2315021824 | 0.06083426119 |               -0.1706679212   |       -0.005975751805 |   -0.4254891515  |  -0.02625318666   |     27 |      0 |        5 |
| T1         | T2       | Independent MaxF |  32 |    0.1413343544 | 0.1489696311  |                0.007635276773 |        2.38821131e-05 |   -0.01380382195 |   0.03278843079   |     11 |      0 |       21 |

BOUNDING_NOT_SUPPORTED_AS_CORE_MECHANISM。T0=Base，T1=冻结 F0，T2=去除范数相关饱和及硬上限，仍保留相同方向、阶段、score 映射和几何安全检查。T2 的均值更低，当前结果不支持“bounded 优于 unbounded”或“bound 是收益的必要原因”。

Bounded correction 应定位为机械可验证的安全/幅度约束设计，而不是实验已证明的性能核心或稳定性提升。T2 同时去掉归一化饱和与硬 cap，不是只删去本已冗余的 0.01 Å cap；该对照不能进一步分离两者。保留负结果，不调整 cap，不重新调参。

本文所有数字由 `collect_results.py` 自动读取并按 seed 合并，不从提示词近似数抄写。完整精度见 [主结果 CSV](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/main_results.csv)、[逐种子数据](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/all_per_seed_metrics.csv)、[统计 CSV](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/paired_statistics.csv)。

历史判定、阈值和置信区间保存在 [原始判定归档](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/original_decisions.json)；原始来源、Git commit、SHA256、评价器和配置见 [复现清单](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/reproducibility_manifest.json)。创新点一原始统计来源：[Formal256 报告](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/sources/a367f85efe8027bf8ef2db41cc48ec2fd3d60582/thesis_archive/reports/innovation1/formal_final_report.json)。

统一边界：SURROGATE_PROPERTY_EVAL=True；DFT_VERIFIED=False。Stable/E-hull/力均为代理评价，不是实验合成或 DFT 真值证明。不同 cohort 不作绝对指标直接排名。
