# RC-NFGD 注入时机代码审计与预注册方案

状态：代码审计完成；新 timing 实验尚未产生结果。原始 P0/Formal32/Formal256 文件只读，不改动。下述等调用窗口为上一轮设计；2026-09-29 最新指令只允许改变注入开始时间，故旧窗口方案被新的累计阈值方案取代，绝不可直接执行。见 `results/rcnfgd/timing_ablation/timing_ablation_report.md`。

## 冻结实现

- 采样器：`/mnt/datasets-livsyn/dxl/mattergen_v1_matersim_guidance/experiments/mattersim_late_force_guidance_p0/late_force_sampler.py` 的 `LateMatterSimForceGuidedSampler._evaluate_exact_score`。先调用冻结的 CFG score，再仅当 `sampling_context.phase == predictor` 且 `t <= guidance_t_max + 1e-8` 时，计算 clean estimate 上的 MatterSim 力。
- 力修正：去除整体平移分量，按冻结参考力 `0.07795149218357911 eV/Å` 与名义步长 `0.005 Å` 缩放、硬截断到 `0.01 Å`；利用 clean cell 转换为分数坐标，再除以 ancestral predictor 的精确 score 系数，只修改 position score。最小周期距离不得低于 `0.5 Å`；错误或不安全时该步回退至原 score。原子类别、晶胞、corrector 不受外力修改。
- 时间轴：`pc_sampler.py` 用 `torch.linspace(max_t, eps_t, N)`，冻结配置 `N=1000, eps_t=1/N=0.001`，模型最大扩散时间为 `1`。采样顺序是从噪声端 `t≈1` 到干净端 `t=0.001`，一共 1000 个 predictor 步及每步一个 corrector。历史 `t<=0.02` 即索引 980–999，共 20 次 eligible force calls。它是明确的后期窗口，不是 sigma 阈值或“20%采样进度”。
- 冻结环境：MatterGen `dft_mag_density` 官方 checkpoint SHA256 `01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e`；MatterSim-5M SHA256 `e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5`。目标 0.2，常量 CFG=2，batch=1，单 seed 重置 RNG，确定性 PyTorch。原采样入口见 `mattersim_late_force_guidance_formal32/generate_formal32_pair.py`。
- 已有结果：Formal256 为同一 256 种子配对的 C0/F0，不能重新标成新 timing cohort。A6 为独立 32 种子配对的 C0/F0/POST，可用于 Pareto；不同 cohort 的均值不可放在一条同质 Pareto 前沿上。

## Timing 新实验的冻结比较

仅在**新**实验目录添加薄包装采样器，调用上述冻结类的方法；原采样器、模型、力公式、阈值、checkpoint 与历史结果不修改。由于“仅提前开始，结束仍固定在最后一步”会令 Early/Middle/Late 分别多出大量力调用并混淆时间与预算，采用固定每臂 **20 个 predictor 步**的互斥窗口来隔离注入时机：

| Arm | predictor 索引（含） | 对应模型时间 t | 力调用预算 |
|---|---:|---:|---:|
| C0 | 无 | 无 | 0 |
| Early | 0–19 | 1.000–0.981 | 20 |
| Middle | 490–509 | 0.510–0.491 | 20 |
| Late | 980–999 | 0.020–0.001 | 20 |

窗口在结果出现前冻结。Late 必须精确调用原始 `t<=0.02` 路径，其对新种子上的结果是新实验而不是历史 Formal256 的重分析。Early/Middle 的 clean estimate 可能不可靠，甚至连续 20 步都触发安全回退；这些失败同样保留。这个设计检验的是**同等 force-call 数的注入窗口**，不是“更早开始并一直作用到终点”的累计策略。

预注册样本：32 个新配对 seed `750000–750031`，各臂使用完全相同的 seed、1000 步以及采样配置。`750000` 范围在当前历史实验文本和种子配置检索中未发现使用；运行前再次执行全工作树碰撞检查并记录。所有四臂同时完成，不因早期结果停样或改窗口。不得根据数据修改力权重/安全阈值/力模型/评价器。每种臂的 MatterGen score calls 理论上均为 2000；同时报告 eligible、accepted、fallback、实际 MatterSim 调用和 wall time，不能把“同 force-call 上限”写成“同实际耗时”。

主要读出：终态 MatterSim MaxF（eV/Å，越小越好）；次要为 MatterSim mean atomic force、Property MAE、Validity、Stable、E-hull、NUS。均报告均值、样本标准差及配对 C0−arm 的 20,000 次 bootstrap 95% percentile CI；Early/Middle/Late 间也给出配对差，保留全部 32 对，失败生成单列而不静默删除。`Stable/E-hull/Property` 是既有代理评估，不能称 DFT 验证。若任何臂的评价不完整，则论文结论状态为 incomplete，不能自行用完成子集作确认性结论。

## GPU/运行风险

主机为 h20-1。审计时 GPU0/1 被另一位用户的 JoyAI 服务占用；GPU2–7 显示 0 MiB，仅可在启动前再次确认。不得中断或抢占 GPU0/1。历史工作使用 `/mnt/datasets-livsyn/dxl/mattergen_v1/.venv`；尚需对新实验做非结果性环境/接口 smoke 检查。长任务必须在 tmux 及独立日志中运行。若环境、seed 独立性或 GPU 状态不满足，停止生成并说明，不得留下“已完成”报告。
