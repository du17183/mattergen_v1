# RC-NFGD Timing Ablation 结论状态

**NOT_RUN；不能选择 Late / Similar / Early–Middle 三种结果分支。** 已重建可运行环境并实测历史 seed 740000，但结构与指标复现门槛 FAIL，因此没有启动任何新的 32 配对 seed。详情见 `docs/rcnfgd_environment_rebuild_20260929.md`。`results/rcnfgd/timing_ablation/timing_ablation_summary.csv` 保持四臂 `NOT_RUN`，指标为空；不存在可计算的 mean/std/paired bootstrap CI，也不存在可供论文使用的 early/middle/late 排名。

现有历史诊断只说明原作者为何选取 `t≤0.02` 的后期 clean estimate；它不是同 cohort、同评价的阶段消融。当前论文仍可准确表述“RC-NFGD 在末 20 个 predictor 步注入力反馈”，但不能表述“实验证明末期优于早期/中期”或“末期避免破坏生成过程”。

本轮已在查看任何新结果之前将“只修改开始时间”冻结为 Early `t≤1.0`、Middle `t≤0.5`、Late `t≤0.02`，三臂都持续到采样终点；旧每臂 20 次的互斥窗口设计已废止。恢复环境并通过 `docs/rcnfgd_reproducibility_check.md` 后，只能执行此方案，仍需在启动前核对完整历史 seed 碰撞。因为力调用预算分别最多 1000、500、20 次，必须报告 score calls、力调用、接受/回退、wall time，并把预算差异作为解释限制。只在全部新样本与评价完成后选择下列分支：

- Late 在预注册主要 MatterSim MaxF 及质量护栏上更好：可以写“本设置下后期注入较有利”，同时陈述预算差异、代理偏倚和未做 DFT；不能直接推出普遍机制。
- 阶段结果相近或 CI 不区分：只能写“原方法中的物理反馈有效，但本实验未证明注入阶段的稳定差异”。
- Early/Middle 更好或 Late 有害：完整保留负结果，修改方法动机与论文定位，不得再强调“晚期优选”。

本文件是边界声明，不是实验结果报告。详情见 `docs/rcnfgd_environment_rebuild_20260929.md` 与 `results/rcnfgd/timing_ablation/timing_ablation_report.md`。
