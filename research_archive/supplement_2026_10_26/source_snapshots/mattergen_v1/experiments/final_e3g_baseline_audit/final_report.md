# EXP-08 E3-G 强后处理基线

执行状态：COMPLETED_EXISTING；只补充审计和统计，不重复生成。

| Method   | Cohort       | Seed range   |   n | Property MAE   | Hit   |         MaxF | Atomic Force   |          RMSD |       E-hull |    Stable |   Novel |   Unique |        NUS |   Validity | Runtime   | Method label       |
|:---------|:-------------|:-------------|----:|:---------------|:------|-------------:|:---------------|--------------:|-------------:|----------:|--------:|---------:|-----------:|-----------:|:----------|:-------------------|
| C0       | E3 Formal256 | 40000-40255  | 256 | NA             | NA    | 0.3429642629 | NA             | 0.04938950866 | 0.1561356966 | 0.4453125 | 0.71875 | 0.984375 | 0.22265625 |          1 | NA        | Official MatterGen |
| E3-A     | E3 Formal256 | 40000-40255  | 256 | NA             | NA    | 0.2439559598 | NA             | 0.04505749632 | 0.1561788554 | 0.4453125 | 0.71875 | 0.984375 | 0.22265625 |          1 | NA        | E3-A               |
| E3-G     | E3 Formal256 | 40000-40255  | 256 | NA             | NA    | 0.2631070179 | NA             | 0.04593743919 | 0.1561766907 | 0.4453125 | 0.71875 | 0.984375 | 0.22265625 |          1 | NA        | E3-G               |

E3-G role=STRONG POST-GENERATION BASELINE / FALLBACK，不作为第二创新主线。其阶段是“生成完成后，learned gate 决策，再作基于 CHGNet 力的局部位置精修”，而 RC-NFGD 在最后一段 reverse diffusion 内使用 MatterSim 力修正位置 score 后继续采样。两者的力源及调用定义并不相同；仅 A6 的 POST 与 F0 构成同一 MatterSim 实际调用预算的直接比较。

E3-G 相对本 cohort 的 C0 预松弛 MaxF 改善 23.2844%；原始绝对差 E3-G-C0=-0.07985724494 eV/Å，95% CI [-0.1449657996, -0.03245265378] eV/Å；原始严格符号 wins/ties/losses=163/0/93，Holm p=4.188017323e-10。这是绝对差 CI，不得误标成相对百分比 CI。

Gate 启用率=66.4062%，启用数=170/256；平均接受步数=3.3203125。Always-on harm rate=25.3906%，gated harm rate=18.3594%。Harm 采用 1e-6 eV/Å 容忍阈值，而原始主表 wins 使用严格符号，不能混同。接受步数不是全部原子势实际调用数。

E3-G cohort=40000–40255、target=0.1；F0 Formal256 cohort=740000–740255、target=0.2。即便生成 checkpoint 一样，cohort 和条件目标不同也不允许绝对 MaxF 排名；相对改善仅是各自配对基线下的描述，不是直接统计优越比较。归档仅有 relaxation elapsed，不能当成端到端生成+精修用时。历史 Mag MAE/Hit/原子平均力/端到端时间缺项标为 NA。

本文所有数字由 `collect_results.py` 自动读取并按 seed 合并，不从提示词近似数抄写。完整精度见 [主结果 CSV](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/main_results.csv)、[逐种子数据](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/all_per_seed_metrics.csv)、[统计 CSV](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/paired_statistics.csv)。

历史判定、阈值和置信区间保存在 [原始判定归档](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/original_decisions.json)；原始来源、Git commit、SHA256、评价器和配置见 [复现清单](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/reproducibility_manifest.json)。创新点一原始统计来源：[Formal256 报告](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/sources/a367f85efe8027bf8ef2db41cc48ec2fd3d60582/thesis_archive/reports/innovation1/formal_final_report.json)。

统一边界：SURROGATE_PROPERTY_EVAL=True；DFT_VERIFIED=False。Stable/E-hull/力均为代理评价，不是实验合成或 DFT 真值证明。不同 cohort 不作绝对指标直接排名。
