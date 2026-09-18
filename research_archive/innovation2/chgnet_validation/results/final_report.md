# EXP-02 独立势 Formal256 验证

执行状态：COMPLETED_EXISTING；只补充审计和统计，不重复生成。

Independent MaxF：0.1938906373 → 0.1660573042；相对改善 14.3552%，配对 95% CI [4.3186%, 27.7288%]；改善/持平/变差 172/0/84。

Independent mean force：0.08797635633 → 0.07954108791；相对改善 9.5881%，配对 95% CI [3.0817%, 19.0993%]；改善/持平/变差 169/0/87。

| Cohort   | Method   | Metric                 |   n | units       |          mean |        median |          p90 |          p95 |          max |
|:---------|:---------|:-----------------------|----:|:------------|--------------:|--------------:|-------------:|-------------:|-------------:|
| A1       | C0       | Independent MaxF       | 256 | eV/Angstrom | 0.1938906373  | 0.08886908657 | 0.4058529686 | 0.6612338398 | 5.072047194  |
| A1       | C0       | Independent mean force | 256 | eV/Angstrom | 0.08797635633 | 0.04633585335 | 0.2206215315 | 0.2844048319 | 1.558912665  |
| A1       | F0       | Independent MaxF       | 256 | eV/Angstrom | 0.1660573042  | 0.08518204525 | 0.425399163  | 0.6962488835 | 1.81443494   |
| A1       | F0       | Independent mean force | 256 | eV/Angstrom | 0.07954108791 | 0.04816591003 | 0.2133071407 | 0.2764061578 | 0.8369934774 |

CROSS_MLIP_GENERALIZATION=SUPPORTED。C0/F0 共 512 个结构已完整评估，冻结 CHGNet 0.3.0 原始原子力向量复算两种力指标通过。该证据反驳“改善仅体现于同一个 MatterSim 打分器”的最强说法，但不能排除训练数据重叠，也不等价于真实能量面保证。CHGNet 同时用于磁性护栏，因此独立性是“未参与力引导”的评价器独立性，不是完全未被项目使用过的盲测模型。

跨势改善并不覆盖所有分位数：CHGNet MaxF P90 从 0.4058529686 增至 0.425399163，P95 从 0.6612338398 增至 0.6962488835 eV/Å；原子平均力的结构间中位数从 0.04633585335 增至 0.04816591003 eV/Å。支持的是预先指定的平均力相关终点，而不是全分布或每个尾部指标都一致改善。

本文所有数字由 `collect_results.py` 自动读取并按 seed 合并，不从提示词近似数抄写。完整精度见 [主结果 CSV](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/main_results.csv)、[逐种子数据](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/all_per_seed_metrics.csv)、[统计 CSV](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/paired_statistics.csv)。

历史判定、阈值和置信区间保存在 [原始判定归档](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/original_decisions.json)；原始来源、Git commit、SHA256、评价器和配置见 [复现清单](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/reproducibility_manifest.json)。创新点一原始统计来源：[Formal256 报告](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/sources/a367f85efe8027bf8ef2db41cc48ec2fd3d60582/thesis_archive/reports/innovation1/formal_final_report.json)。

统一边界：SURROGATE_PROPERTY_EVAL=True；DFT_VERIFIED=False。Stable/E-hull/力均为代理评价，不是实验合成或 DFT 真值证明。不同 cohort 不作绝对指标直接排名。
