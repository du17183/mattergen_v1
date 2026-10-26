# 效率及计算成本

Adaptive CFG 复用条件/无条件神经网络输出，仅增加按字段归约和两个标量 EMA 状态。额外归约计算随得分张量总元素数增长，控制器常驻状态为 O(1)；并不新增训练参数或神经网络前向，但 GPU→CPU 标量同步可能产生实际开销，不能把“无额外 forward”写成“零开销”。

RC-NFGD 在 1000-step reverse diffusion 的最后 20 个 predictor 增加 20 次 MatterSim 力计算；相对采样工作量为 T_base+20*T_potential+映射/安全检查开销，不把原子势计算成本当作常数。正式 n=256 的真实调用总数为 5120，失败/回退为零。峰值显存是每个采样段的 PyTorch allocated 统计，不是 GPU 总显存，也不代表训练成本。

已有运行时间来自共享服务器。P0 和 A5 有明确其他任务竞争记录，Formal256 各组运行顺序也不是随机交叉的隔离微基准。不得从 F0 偶尔稍快推出加速结论，不剔除慢 seed，不比较不同 cohort 的绝对运行时。表中 Runtime 为记录的采样/势加载范围，不包含离线评价、MatterSim 松弛或 reference 构建。E3 的松弛耗时另列；其生成+精修时间缺失。
| Cohort                | Method     |   n | Runtime            | Runtime scope                                                              |
|:----------------------|:-----------|----:|:-------------------|:---------------------------------------------------------------------------|
| Innovation1 Formal256 | C0         | 256 | 204.47824956220154 | sampling+force-potential load where applicable; excludes evaluator runtime |
| Innovation1 Formal256 | A0         | 256 | 209.5455809699938  | sampling+force-potential load where applicable; excludes evaluator runtime |
| E3 Formal256          | C0         | 256 | NA                 | NA (only relaxation time archived)                                         |
| E3 Formal256          | E3-A       | 256 | NA                 | NA (only relaxation time archived)                                         |
| E3 Formal256          | E3-G       | 256 | NA                 | NA (only relaxation time archived)                                         |
| A1                    | C0         | 256 | 92.84878176247821  | sampling+force-potential load where applicable; excludes evaluator runtime |
| A1                    | F0         | 256 | 92.66313072172193  | sampling+force-potential load where applicable; excludes evaluator runtime |
| Formal32              | C0         |  32 | 86.26528826166032  | sampling+force-potential load where applicable; excludes evaluator runtime |
| Formal32              | F0         |  32 | 86.55536163669422  | sampling+force-potential load where applicable; excludes evaluator runtime |
| P0                    | B0         |  16 | 150.28391521199592 | sampling+force-potential load where applicable; excludes evaluator runtime |
| P0                    | F0         |  16 | 150.25152837080168 | sampling+force-potential load where applicable; excludes evaluator runtime |
| A6                    | C0         |  32 | 84.55636092574969  | sampling+force-potential load where applicable; excludes evaluator runtime |
| A6                    | F0         |  32 | 85.3786244604944   | sampling+force-potential load where applicable; excludes evaluator runtime |
| A6                    | POST       |  32 | 85.38156319449809  | sampling+force-potential load where applicable; excludes evaluator runtime |
| A3                    | G0         |  32 | 100.41072964905834 | sampling+force-potential load where applicable; excludes evaluator runtime |
| A3                    | G1         |  32 | 101.98852241943132 | sampling+force-potential load where applicable; excludes evaluator runtime |
| A3                    | G2         |  32 | 91.63984835685005  | sampling+force-potential load where applicable; excludes evaluator runtime |
| A3                    | G3         |  32 | 90.03129307006384  | sampling+force-potential load where applicable; excludes evaluator runtime |
| A3                    | G4         |  32 | 92.85361612390261  | sampling+force-potential load where applicable; excludes evaluator runtime |
| A4                    | T0         |  32 | 84.49223986574907  | sampling+force-potential load where applicable; excludes evaluator runtime |
| A4                    | T1         |  32 | 85.31104013625554  | sampling+force-potential load where applicable; excludes evaluator runtime |
| A4                    | T2         |  32 | 84.71428831612502  | sampling+force-potential load where applicable; excludes evaluator runtime |
| A5                    | C0         |  64 | 113.58640703084257 | sampling+force-potential load where applicable; excludes evaluator runtime |
| A5                    | A0         |  64 | 107.00548012142644 | sampling+force-potential load where applicable; excludes evaluator runtime |
| A5                    | B0         |  64 | 107.94436322420461 | sampling+force-potential load where applicable; excludes evaluator runtime |
| A5                    | AB         |  64 | 118.0970498689394  | sampling+force-potential load where applicable; excludes evaluator runtime |
| Innovation1 minimal8  | C0         |   8 | 87.94825957150897  | sampling+force-potential load where applicable; excludes evaluator runtime |
| Innovation1 minimal8  | A0         |   8 | 85.13987012962752  | sampling+force-potential load where applicable; excludes evaluator runtime |
| Innovation1 minimal8  | POS_ONLY   |   8 | 85.1854813193786   | sampling+force-potential load where applicable; excludes evaluator runtime |
| Innovation1 minimal8  | NO_EMA     |   8 | 86.54553200975352  | sampling+force-potential load where applicable; excludes evaluator runtime |
| Innovation1 minimal8  | SHARED_EMA |   8 | 86.46036521362112  | sampling+force-potential load where applicable; excludes evaluator runtime |
| Innovation1 minimal8  | NO_CLIP    |   8 | 86.54959585087636  | sampling+force-potential load where applicable; excludes evaluator runtime |


本文所有数字由 `collect_results.py` 自动读取并按 seed 合并，不从提示词近似数抄写。完整精度见 [主结果 CSV](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/main_results.csv)、[逐种子数据](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/all_per_seed_metrics.csv)、[统计 CSV](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/paired_statistics.csv)。

历史判定、阈值和置信区间保存在 [原始判定归档](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/original_decisions.json)；原始来源、Git commit、SHA256、评价器和配置见 [复现清单](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/reproducibility_manifest.json)。创新点一原始统计来源：[Formal256 报告](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/sources/a367f85efe8027bf8ef2db41cc48ec2fd3d60582/thesis_archive/reports/innovation1/formal_final_report.json)。

统一边界：SURROGATE_PROPERTY_EVAL=True；DFT_VERIFIED=False。Stable/E-hull/力均为代理评价，不是实验合成或 DFT 真值证明。不同 cohort 不作绝对指标直接排名。


## 调用及显存明细

| Cohort                | Method     |   n | runtime_mean_s     | runtime_median_s   | runtime_p95_s      | actual_mattersim_calls_mean   | failed_mattersim_calls_sum   | peak_allocated_bytes_max   | mattergen_score_calls_mean   |
|:----------------------|:-----------|----:|:-------------------|:-------------------|:-------------------|:------------------------------|:-----------------------------|:---------------------------|:-----------------------------|
| A1                    | C0         | 256 | 92.84878176247821  | 84.53575774451019  | 90.27791075126152  | 0.0                           | 0.0                          | 385280000.0                | 2000.0                       |
| A1                    | F0         | 256 | 92.66313072172193  | 85.1868188335211   | 89.57930388003297  | 20.0                          | 0.0                          | 496613888.0                | 2000.0                       |
| A3                    | G0         |  32 | 100.41072964905834 | 85.34286535248975  | 180.8565893381892  | 0.0                           | 0.0                          | 377844224.0                | 2000.0                       |
| A3                    | G1         |  32 | 101.98852241943132 | 85.70083767347387  | 210.3739221433206  | 20.0                          | 0.0                          | 490876416.0                | 2000.0                       |
| A3                    | G2         |  32 | 91.63984835685005  | 84.93057565652998  | 94.48392458780144  | 20.0                          | 0.0                          | 527714304.0                | 2000.0                       |
| A3                    | G3         |  32 | 90.03129307006384  | 85.37757213000441  | 105.81697968380175 | 20.0                          | 0.0                          | 563915776.0                | 2000.0                       |
| A3                    | G4         |  32 | 92.85361612390261  | 85.15296800847864  | 116.20711981311904 | 20.0                          | 0.0                          | 600720896.0                | 2000.0                       |
| A4                    | T0         |  32 | 84.49223986574907  | 84.44544225750724  | 86.08509265113389  | 0.0                           | 0.0                          | 375750144.0                | 2000.0                       |
| A4                    | T1         |  32 | 85.31104013625554  | 85.28489554900443  | 87.62601455424155  | 20.0                          | 0.0                          | 481179136.0                | 2000.0                       |
| A4                    | T2         |  32 | 84.71428831612502  | 84.56231029995251  | 86.643275951955    | 20.0                          | 0.0                          | 517581824.0                | 2000.0                       |
| A5                    | C0         |  64 | 113.58640703084257 | 84.81410407699877  | 367.9600322281475  | 0.0                           | 0.0                          | 374427648.0                | 2000.0                       |
| A5                    | A0         |  64 | 107.00548012142644 | 85.10206485851086  | 203.97352488466927 | 0.0                           | 0.0                          | 377738240.0                | 2000.0                       |
| A5                    | B0         |  64 | 107.94436322420461 | 85.57204449700657  | 296.12070215999415 | 20.0                          | 0.0                          | 484913152.0                | 2000.0                       |
| A5                    | AB         |  64 | 118.0970498689394  | 86.32381983599043  | 406.9661665515973  | 20.0                          | 0.0                          | 520492032.0                | 2000.0                       |
| A6                    | C0         |  32 | 84.55636092574969  | 84.30301278299885  | 86.7273269214871   | 0.0                           | 0.0                          | 374434816.0                | 2000.0                       |
| A6                    | F0         |  32 | 85.3786244604944   | 85.17910447498434  | 88.02303719312768  | 20.0                          | 0.0                          | 464584192.0                | 2000.0                       |
| A6                    | POST       |  32 | 85.38156319449809  | 85.13883938349318  | 87.57932059717423  | 20.0                          | 0.0                          | 496968704.0                | 2000.0                       |
| P0                    | B0         |  16 | 150.28391521199592 | 86.88415964398882  | 595.848918421485   | NA                            | NA                           | NA                         | 2000.0                       |
| P0                    | F0         |  16 | 150.25152837080168 | 87.35649484099122  | 591.510540714502   | NA                            | NA                           | NA                         | 2000.0                       |
| Formal32              | C0         |  32 | 86.26528826166032  | 85.32890651052003  | 92.59572544189577  | NA                            | NA                           | NA                         | NA                           |
| Formal32              | F0         |  32 | 86.55536163669422  | 85.9532018515165   | 90.24037657007575  | NA                            | NA                           | NA                         | NA                           |
| Innovation1 Formal256 | C0         | 256 | 204.47824956220154 | 201.05363212898374 | 231.01261819450883 | NA                            | NA                           | NA                         | NA                           |
| Innovation1 Formal256 | A0         | 256 | 209.5455809699938  | 206.7527083009918  | 235.60800894451677 | NA                            | NA                           | NA                         | NA                           |
| Innovation1 minimal8  | C0         |   8 | 87.94825957150897  | 84.8909371389891   | 96.32734220681596  | NA                            | NA                           | NA                         | 2000.0                       |
| Innovation1 minimal8  | A0         |   8 | 85.13987012962752  | 84.35753332503373  | 89.04058349709958  | NA                            | NA                           | NA                         | 2000.0                       |
| Innovation1 minimal8  | POS_ONLY   |   8 | 85.1854813193786   | 84.50932348152855  | 89.21759936938179  | NA                            | NA                           | NA                         | 2000.0                       |
| Innovation1 minimal8  | NO_EMA     |   8 | 86.54553200975352  | 84.58705181151163  | 94.47721600134973  | NA                            | NA                           | NA                         | 2000.0                       |
| Innovation1 minimal8  | SHARED_EMA |   8 | 86.46036521362112  | 84.3761668149964   | 94.99546294219908  | NA                            | NA                           | NA                         | 2000.0                       |
| Innovation1 minimal8  | NO_CLIP    |   8 | 86.54959585087636  | 84.45038769952953  | 94.59207025694487  | NA                            | NA                           | NA                         | 2000.0                       |
| E3 Formal256          | C0         | 256 | NA                 | NA                 | NA                 | NA                            | NA                           | NA                         | NA                           |
| E3 Formal256          | E3-A       | 256 | NA                 | NA                 | NA                 | NA                            | NA                           | NA                         | NA                           |
| E3 Formal256          | E3-G       | 256 | NA                 | NA                 | NA                 | NA                            | NA                           | NA                         | NA                           |

字段缺失时不把 guidance event 或接受步数冒充 actual force-call count。
