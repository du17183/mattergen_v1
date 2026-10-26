# 最终统计口径及结果

配对单位为 seed，所有注册 seed 都保留。连续指标以“method-baseline”的差及其 95% CI 报告；较小更好的指标，相对改善=(baseline均值-method均值)/baseline均值。bootstrap 每次一起重采样配对，再计算均值比，而非平均逐 seed 百分比。比例类同时给 fraction 差，可乘 100 表示 pp。

历史正式检验不被覆盖：P0 bootstrap seed=20260913，Formal32=20260914，A1/A3/A4/A5/A6及其独立势=20260915，E3=20260728；这些已记录流程均为 20,000 次。I1 原始报告的随机种子/次数没有完整归档，本轮保留其原始 CI/p，不臆造参数。另行提供可复现的补充 20,000 次 paired percentile bootstrap（seed=20260915），标注为 retrospective/supplementary，不能替换原始冻结判定。

补充区间未做多重比较校正。NUS/Unique 的 bootstrap 固定原 cohort 标签，不在重复采样的集合上重跑去重，因此区间是条件于本批定义的标签；不要外推成全生成分布的唯一性保证。Wins 默认容忍差 1e-12；E3 历史严格符号统计和 harm 阈值另行保留。不存在看结果后更换 evaluator、删掉失败样本或改 primary endpoint 的操作。

## 创新点一原始正式统计

| Method   | Cohort                | Seed range   |   n | Property MAE   | Hit   | MaxF   | Atomic Force   |          RMSD |       E-hull |     Stable |      Novel |     Unique |        NUS |   Validity |     Runtime | Method label       |
|:---------|:----------------------|:-------------|----:|:---------------|:------|:-------|:---------------|--------------:|-------------:|-----------:|-----------:|-----------:|-----------:|-----------:|------------:|:-------------------|
| C0       | Innovation1 Formal256 | 20000-20255  | 256 | NA             | NA    | NA     | NA             | 0.06457459745 | 0.1436665541 | 0.41015625 | 0.73828125 | 0.95703125 | 0.22265625 |          1 | 204.4782496 | Official MatterGen |
| A0       | Innovation1 Formal256 | 20000-20255  | 256 | NA             | NA    | NA     | NA             | 0.07382172668 | 0.1402316092 | 0.46875    | 0.765625   | 0.98046875 | 0.2578125  |          1 | 209.545581  | Adaptive CFG       |

原始正式统计（保留原检验，不替换为新 bootstrap）：

| Metric            |           A0-C0 |   Original 95% CI low |   Original 95% CI high |        Raw p |   Holm p | Wins/Ties/Losses   |
|:------------------|----------------:|----------------------:|-----------------------:|-------------:|---------:|:-------------------|
| E-hull (eV/atom)  | -0.003434944937 |        -0.01792562011 |           0.0110300358 | 0.3573553962 |        1 | 137/1/118          |
| Stable (fraction) |  0.05859375     |        -0.015625      |           0.1328125    | 0.1461770286 |        1 | 54/163/39          |
| NUS (fraction)    |  0.03515625     |        -0.02734375    |           0.09765625   | 0.3424709984 |        1 | 40/185/31          |

冻结门槛要求 E-hull 恶化不超过 0.02 eV/atom，Stable 与 NUS 下降不超过 3 pp，并且至少两个主指标方向改善。原始门槛通过，FORMAL_INNOVATION1_CONFIRMED=True。

必须区别“预设方向性门槛确认”和“统计显著优越”：三个主指标原始 CI 均跨零，不能宣称显著改善，也未执行严格的统计非劣效检验。RMSD 从 0.06457459745 增至 0.07382172668 Å，应如实报告质量维度间并非一致受益。

该 Formal256 未保存属性预测与 pre-relax 力原始数据，Property MAE、Hit、pre-relax MaxF/mean force 标为 NA。归档的 maximum_force_ev_ang 是松弛后力，绝不能填入 pre-relax MaxF 列。新的小样本或 A5 不能回填历史 Formal256。

## F0 正式主结果

MaxF：0.2264083679 → 0.1589114891；相对改善 29.8120%，配对 95% CI [21.4166%, 40.8758%]；改善/持平/变差 254/0/2。

Atomic Force：0.0987023058 → 0.06937279273；相对改善 29.7151%，配对 95% CI [24.0121%, 36.9627%]；改善/持平/变差 255/0/1。

RMSD：0.05113694678 → 0.04389322436；相对改善 14.1653%，配对 95% CI [4.5382%, 25.9530%]；改善/持平/变差 216/21/19。

| Method   | Cohort   | Seed range    |   n |   Property MAE |        Hit |         MaxF |   Atomic Force |          RMSD |        E-hull |     Stable |      Novel |     Unique |        NUS |   Validity |     Runtime | Method label       |
|:---------|:---------|:--------------|----:|---------------:|-----------:|-------------:|---------------:|--------------:|--------------:|-----------:|-----------:|-----------:|-----------:|-----------:|------------:|:-------------------|
| C0       | A1       | 740000-740255 | 256 | 0.009756888912 | 0.67578125 | 0.2264083679 |  0.0987023058  | 0.05113694678 | 0.09129777084 | 0.80859375 | 0.484375   | 0.51171875 | 0.2109375  |          1 | 92.84878176 | Official MatterGen |
| F0       | A1       | 740000-740255 | 256 | 0.00986816916  | 0.67578125 | 0.1589114891 |  0.06937279273 | 0.04389322436 | 0.09176719068 | 0.80859375 | 0.48828125 | 0.515625   | 0.21484375 |          1 | 92.66313072 | RC-NFGD            |

所有正式种子保留，未因高 MaxF 或不利结果删除样本。冻结质量护栏通过，但不等于每个质量指标都改善：磁密度 MAE 与 E-hull 的均值有小幅增加，NUS 小幅增加，Stable/Hit/几何 Validity 未变。主结论限定为本 checkpoint、目标与采样设置下的代理预松弛力改善。

## 独立评价

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

其余全部配对比较见 paired_statistics.csv。

本文所有数字由 `collect_results.py` 自动读取并按 seed 合并，不从提示词近似数抄写。完整精度见 [主结果 CSV](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/main_results.csv)、[逐种子数据](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/all_per_seed_metrics.csv)、[统计 CSV](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/paired_statistics.csv)。

历史判定、阈值和置信区间保存在 [原始判定归档](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/original_decisions.json)；原始来源、Git commit、SHA256、评价器和配置见 [复现清单](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/reproducibility_manifest.json)。创新点一原始统计来源：[Formal256 报告](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/sources/a367f85efe8027bf8ef2db41cc48ec2fd3d60582/thesis_archive/reports/innovation1/formal_final_report.json)。

统一边界：SURROGATE_PROPERTY_EVAL=True；DFT_VERIFIED=False。Stable/E-hull/力均为代理评价，不是实验合成或 DFT 真值证明。不同 cohort 不作绝对指标直接排名。
