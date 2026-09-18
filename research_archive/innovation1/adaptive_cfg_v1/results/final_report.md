# EXP-03 创新点一最终审计

## 方法定义

在同一次条件/无条件得分计算后，对字段 $f\in\{a,p,L\}$ 计算

$$d_{f,t}=\sqrt{\operatorname{mean}((s_{c,f,t}-s_{u,f,t})^2)} ,\qquad d_t=\tfrac13\sum_f d_{f,t}.$$

这不是将不同形状张量拼接后的总体 RMS，而是每个字段单独降维后等权取均值。源码顺序为“先字段聚合、后阶段 EMA”。对 $q\in\{predictor,corrector\}$ 分别维护

$$m_{q,t}=\begin{cases}d_t,&\text{该阶段首次观测}\\\beta m_{q,prev}+(1-\beta)d_t,&\text{否则}\end{cases}.$$

先用当前观测更新 EMA，再求比值 $r_t=d_t/(m_{q,t}+\epsilon)$，并计算

$$u_t=\operatorname{clip}(1+\alpha(r_t-1),0.25,4),\quad g_t=\operatorname{clip}(g_0u_t,g_{min},g_{max}),$$

$$s_{f,t}=\operatorname{lerp}(s_{u,f,t},s_{c,f,t},g_t),\quad \forall f.$$

所有字段共用同一个 $g_t$。不是 field-specific CFG，不是 predictor/corrector 使用同一 EMA。冻结参数：$g_0=2.0$，$\alpha=0.5$，$\beta=0.95$，$\epsilon=1e-06$，最终 scale 区间 [0.0, 5.0]。本方法使用 adaptive 而非 stage_adaptive，不叠加 piecewise 日程。非法残差/非有限统计回退至基础引导；每次采样开始重置 EMA。

源码：[guidance_schedule.py](/mnt/datasets-livsyn/dxl/mattergen_v1/mattergen/diffusion/sampling/guidance_schedule.py)、[classifier_free_guidance.py](/mnt/datasets-livsyn/dxl/mattergen_v1/mattergen/diffusion/sampling/classifier_free_guidance.py)。

## 正式结果与统计

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

## 最小组件消融

| Method     | Cohort               | Seed range    |   n |   Property MAE |   Hit |         MaxF |   Atomic Force |          RMSD |        E-hull |   Stable |   Novel |   Unique |   NUS |   Validity |     Runtime | Method label       |
|:-----------|:---------------------|:--------------|----:|---------------:|------:|-------------:|---------------:|--------------:|--------------:|---------:|--------:|---------:|------:|-----------:|------------:|:-------------------|
| C0         | Innovation1 minimal8 | 746000-746007 |   8 |  0.03130602264 | 0     | 0.1430724719 |  0.0787133245  | 0.05436011206 | 0.04337111739 |    1     |   0.25  |        1 | 0.25  |          1 | 87.94825957 | Official MatterGen |
| A0         | Innovation1 minimal8 | 746000-746007 |   8 |  0.019654905   | 0.25  | 0.1957595922 |  0.09344980933 | 0.02932295027 | 0.08587734637 |    0.625 |   0.625 |        1 | 0.25  |          1 | 85.13987013 | Adaptive CFG       |
| POS_ONLY   | Innovation1 minimal8 | 746000-746007 |   8 |  0.0182864115  | 0.375 | 0.3547381141 |  0.1243656971  | 0.1696977689  | 0.0744592501  |    0.75  |   0.5   |        1 | 0.25  |          1 | 85.18548132 | POS_ONLY           |
| NO_EMA     | Innovation1 minimal8 | 746000-746007 |   8 |  0.02420114261 | 0.25  | 0.1851860062 |  0.09282927433 | 0.02947688465 | 0.07720059693 |    0.75  |   0.25  |        1 | 0.125 |          1 | 86.54553201 | NO_EMA             |
| SHARED_EMA | Innovation1 minimal8 | 746000-746007 |   8 |  0.0268228898  | 0.375 | 0.1887286576 |  0.07892723579 | 0.1337852435  | 0.08839360838 |    0.5   |   0.5   |        1 | 0.25  |          1 | 86.46036521 | SHARED_EMA         |
| NO_CLIP    | Innovation1 minimal8 | 746000-746007 |   8 |  0.01965682958 | 0.25  | 0.1957855405 |  0.09348149226 | 0.01853671636 | 0.08588073986 |    0.625 |   0.625 |        1 | 0.25  |          1 | 86.54959585 | NO_CLIP            |


| baseline   | method     | metric       |   n |   baseline_mean |   method_mean |   delta_method_minus_baseline |   paired_median_delta |   delta_ci95_low |   delta_ci95_high |   wins |   ties |   losses |
|:-----------|:-----------|:-------------|----:|----------------:|--------------:|------------------------------:|----------------------:|-----------------:|------------------:|-------:|-------:|---------:|
| C0         | A0         | E-hull       |   8 |   0.04337111739 | 0.08587734637 |               0.04250622899   |       0.05116707079   |  0.005317632327  |   0.07863243756   |      1 |      0 |        7 |
| C0         | A0         | Stable       |   8 |   1             | 0.625         |              -0.375           |       0               | -0.75            |  -0.125           |      0 |      5 |        3 |
| C0         | A0         | NUS          |   8 |   0.25          | 0.25          |               0               |       0               | -0.5             |   0.5             |      2 |      4 |        2 |
| C0         | A0         | Property MAE |   8 |   0.03130602264 | 0.019654905   |              -0.01165111765   |      -0.007722479943  | -0.02636458619   |   0.003609074682  |      6 |      0 |        2 |
| A0         | POS_ONLY   | E-hull       |   8 |   0.08587734637 | 0.0744592501  |              -0.01141809627   |      -0.0180899269    | -0.04679114584   |   0.02832721971   |      6 |      0 |        2 |
| A0         | POS_ONLY   | Stable       |   8 |   0.625         | 0.75          |               0.125           |       0               |  0               |   0.375           |      1 |      7 |        0 |
| A0         | POS_ONLY   | NUS          |   8 |   0.25          | 0.25          |               0               |       0               | -0.375           |   0.375           |      1 |      6 |        1 |
| A0         | POS_ONLY   | Property MAE |   8 |   0.019654905   | 0.0182864115  |              -0.001368493493  |       0.01071282392   | -0.01961581521   |   0.01395281734   |      3 |      0 |        5 |
| A0         | NO_EMA     | E-hull       |   8 |   0.08587734637 | 0.07720059693 |              -0.008676749445  |      -0.0007255356175 | -0.03846575559   |   0.017766544     |      4 |      0 |        4 |
| A0         | NO_EMA     | Stable       |   8 |   0.625         | 0.75          |               0.125           |       0               | -0.25            |   0.5             |      2 |      5 |        1 |
| A0         | NO_EMA     | NUS          |   8 |   0.25          | 0.125         |              -0.125           |       0               | -0.5             |   0.25            |      1 |      5 |        2 |
| A0         | NO_EMA     | Property MAE |   8 |   0.019654905   | 0.02420114261 |               0.004546237617  |       0.0003480448645 | -0.004512744339  |   0.01519382922   |      4 |      0 |        4 |
| A0         | SHARED_EMA | E-hull       |   8 |   0.08587734637 | 0.08839360838 |               0.002516262003  |       0.03004540541   | -0.06238619948   |   0.05563819486   |      2 |      0 |        6 |
| A0         | SHARED_EMA | Stable       |   8 |   0.625         | 0.5           |              -0.125           |       0               | -0.625           |   0.375           |      2 |      3 |        3 |
| A0         | SHARED_EMA | NUS          |   8 |   0.25          | 0.25          |               0               |       0               | -0.375           |   0.375           |      1 |      6 |        1 |
| A0         | SHARED_EMA | Property MAE |   8 |   0.019654905   | 0.0268228898  |               0.007167984803  |       0.004927497505  | -0.01870551716   |   0.03498888384   |      3 |      0 |        5 |
| A0         | NO_CLIP    | E-hull       |   8 |   0.08587734637 | 0.08588073986 |               3.393491109e-06 |       0               | -4.212061564e-07 |   1.043081284e-05 |      3 |      3 |        2 |
| A0         | NO_CLIP    | Stable       |   8 |   0.625         | 0.625         |               0               |       0               |  0               |   0               |      0 |      8 |        0 |
| A0         | NO_CLIP    | NUS          |   8 |   0.25          | 0.25          |               0               |       0               |  0               |   0               |      0 |      8 |        0 |
| A0         | NO_CLIP    | Property MAE |   8 |   0.019654905   | 0.01965682958 |               1.924585683e-06 |      -3.373510291e-09 | -1.333938094e-06 |   7.112822023e-06 |      5 |      0 |        3 |


| method     |   events |   clipped_events |   fallback_events |   minimum_scale |   maximum_scale |   predictor_mean |   corrector_mean |
|:-----------|---------:|-----------------:|------------------:|----------------:|----------------:|-----------------:|-----------------:|
| A0         |    16000 |                2 |                 0 |     1.284239528 |     5           |      2.014148261 |      2.014323206 |
| NO_CLIP    |    16000 |                0 |                 0 |     1.284239528 |     5.445056519 |      2.014196511 |      2.014357372 |
| NO_EMA     |    16000 |                0 |                 0 |     1.999993122 |     1.999999891 |      1.999997607 |      1.999997611 |
| POS_ONLY   |    16000 |               56 |                 0 |     1.062373145 |     5           |      2.151808611 |      2.152878984 |
| SHARED_EMA |    16000 |                2 |                 0 |     1.245159573 |     5           |      2.009239825 |      2.008979123 |

全部 8 个新配对 seed=746000–746007 在产生结果前注册。六组为 C0、完整 A0、仅 position 残差、移除 EMA 时间平滑(beta=0)、共享 predictor/corrector EMA、移除 multiplier/final 两层 clip。只作组件去除，不调其余参数；没有学习新模型。目标=0.1，和 I1 Formal256 一致，但该批不能并入正式256或补写为256属性验证。

机制轨迹逐行独立复算：2000 次 score 决策/样本，五个非 C0 组共 80,000 条，残差、EMA、ratio 和最终 scale 对应公式均匹配。去除 EMA 后 ratio 约为1，接近固定 CFG，但因 epsilon 并非位级严格相同。共享 EMA 保留原有 predictor/corrector 标签，但两阶段交替更新同一状态。POS_ONLY 仍向所有字段应用一个共享 scale。

表格展示所有方向，包括不利结果；不因 n=8 的波动改动 Formal256 冻结方法。即使完整 A0 不是每项指标最优，也不做参数回扫。Clip 是否在本批被触发以轨迹统计为准；无触发不构成未来必然稳定的证据。组件消融完成不等于每个组件的必要性均成立。

### 本次真实结果的解释

新增批次没有复现原 Formal256 的质量改善方向：完整 A0 的 E-hull 从 0.04337111739 增至 0.08587734637 eV/atom，差值 0.04250622899，补充未校正 bootstrap 95% CI [0.005317632327, 0.07863243756]。Stable 从 100.0000% 降至 62.5000%，NUS 均为 25.0000%；属性 MAE 从 0.03130602264 降至 0.019654905，其改善区间仍跨零。结果显示属性与结构质量的权衡，不能认定完整多字段/EMA 版本在本批全部主指标上优于简化版本。n=8 的 percentile bootstrap 和离散比例区间统计效能/覆盖率均有限，不将其单独升级为普遍显著性结论；同样不能因样本小而隐去负方向。创新点一继续保留为用户冻结的方法路线，但其效果证据必须标注为有限、跨批次不一致，而不是稳定复现的普遍提升。

## 效率

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


## 局限

1. 创新点一的 Formal256 满足历史方向性门槛，但三个主指标的 CI 均跨零；不能写成普遍或显著优越。其历史目标属性没有独立预测验证。
2. 新增 n=8 组件消融只补充机制诊断，统计效能有限；无法证明每个组件在所有任务上必要。逐字段 RMS 再等权平均并不等于跨字段尺度标准化，数值较大的字段仍可能主导统计反馈。
3. 仅覆盖一个条件 checkpoint、有限条件目标和数据域；不能保证跨模型/跨材料域可迁移。
4. MatterSim 与 CHGNet 都是代理，可能存在训练数据重叠。CHGNet 也参与属性护栏，不是完全未使用过的盲测模型。
5. 没有真实 DFT、声子、实验合成或动力学稳定性验证；几何 Validity 不是化学有效率或可合成性。
6. F0 的收益不意味着 equal-budget POST 更差；MatterSim 主指标上 POST 更好，独立势直接对比尚无明确胜者。
7. 有界修正必要性不受当前消融支持；简单组合未达到双收益保留要求；F1 未取得需要替换 F0 的收益。
8. 方向对照重放参考轨迹幅度，anti-force 非自身轨迹实时反梯度；未做 clean-x0 与 noisy-x_t 的严格等预算对照。
9. 不同 cohort/目标/运行资源不适合直接绝对排名，运行时间有共享 GPU 竞争混杂。
10. 原始 I1 部分字段和原 bootstrap RNG/次数未归档，E3 端到端运行时缺失；NA 和缺失元数据被明确保留。补充重算只扩大可审计性，不抹去历史来源局限。
11. 新增批次没有复现原 Formal256 的质量改善方向：完整 A0 的 E-hull 从 0.04337111739 增至 0.08587734637 eV/atom，差值 0.04250622899，补充未校正 bootstrap 95% CI [0.005317632327, 0.07863243756]。Stable 从 100.0000% 降至 62.5000%，NUS 均为 25.0000%；属性 MAE 从 0.03130602264 降至 0.019654905，其改善区间仍跨零。结果显示属性与结构质量的权衡，不能认定完整多字段/EMA 版本在本批全部主指标上优于简化版本。n=8 的 percentile bootstrap 和离散比例区间统计效能/覆盖率均有限，不将其单独升级为普遍显著性结论；同样不能因样本小而隐去负方向。创新点一继续保留为用户冻结的方法路线，但其效果证据必须标注为有限、跨批次不一致，而不是稳定复现的普遍提升。

本文所有数字由 `collect_results.py` 自动读取并按 seed 合并，不从提示词近似数抄写。完整精度见 [主结果 CSV](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/main_results.csv)、[逐种子数据](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/all_per_seed_metrics.csv)、[统计 CSV](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/paired_statistics.csv)。

历史判定、阈值和置信区间保存在 [原始判定归档](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/original_decisions.json)；原始来源、Git commit、SHA256、评价器和配置见 [复现清单](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/reproducibility_manifest.json)。创新点一原始统计来源：[Formal256 报告](/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/sources/a367f85efe8027bf8ef2db41cc48ec2fd3d60582/thesis_archive/reports/innovation1/formal_final_report.json)。

统一边界：SURROGATE_PROPERTY_EVAL=True；DFT_VERIFIED=False。Stable/E-hull/力均为代理评价，不是实验合成或 DFT 真值证明。不同 cohort 不作绝对指标直接排名。
