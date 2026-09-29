> 历史状态快照（已被取代）：此文记录 ALM 复现门槛通过之前的 NOT_RUN 判断，不能再作为当前实验状态引用。ALM 环境随后通过 seed 740000 复现门槛，32 个新配对 seed 的正式四臂实验已完成；请以 [最终分析](rcnfgd_timing_analysis_final.md) 和 [正式结果报告](../results/rcnfgd/timing_ablation/timing_ablation_report.md) 为准。原文保留以追踪决策过程。

# RC-NFGD Timing Ablation 分析状态

**结果状态：NOT_RUN。** 已重建可运行环境并实际执行历史 seed 740000 的 C0/F0 replay，但结构与 Property/MatterSim 指标复现门槛为 **FAIL**，所以没有注册/生成 32 个新 paired seeds。详见 `docs/rcnfgd_environment_rebuild_20260929.md`。不能计算 Early/Middle/Late 的 MaxF、mean force、Property MAE、Validity、Stable、E-hull、NUS 或均值/标准差/配对 95% CI。`results/rcnfgd/timing_ablation/timing_ablation_summary.csv` 的四臂保持空指标，不应被引用为实验结果。

冻结设置是：No Force C0；Early `t≤1.0`；Middle `t≤0.5`；Late 原始 `t≤0.02`。同一 1000 步 sampler 的 predictor 时间从 `1.0` 线性下降到 `0.001`，因此理论 eligible force-call 上限分别为 0/1000/500/20；实际调用、接受、fallback 仍需将来逐样本记录。这里只改原采样器 `guidance_t_max`，其他模型、force 公式和评价均冻结。因预算不同，未来结果是**起始时机与力调用预算同时变化**的比较，不能声称同预算的纯时机因果结论。上一轮各臂仅 20 次的互斥窗口设计已废止。

尚不能选择预设的任何解释分支：

- 若 Late 在完成的配对实验中折中最好：可说“在当前设置下后期反馈取得更优折中”，但需披露预算差异、代理势偏倚及无 DFT。
- 若阶段差异有限：只说“原 RC-NFGD 物理反馈有效，但未证实开始时机有稳定影响”。
- 若 Early/Middle 更好：如实保留并修改方法动机，不能隐藏负结果。

历史 `t≤0.02` clean-estimate 诊断是原方法的设计依据，不替代本次 Timing Ablation。环境恢复和历史 seed 复现 PASS 之前，论文第 4 章与摘要不得增加新的“后期最优”实证表述。
