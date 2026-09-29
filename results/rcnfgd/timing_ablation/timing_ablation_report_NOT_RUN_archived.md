# RC-NFGD 力注入时机消融

**状态：NOT_RUN（seed 740000 复现门槛 FAIL）。** 已按授权重建环境并实测历史样本，但 C0/F0 的结构与指标未复现；因此没有启动 32 个新配对 seed。详情见 `docs/rcnfgd_environment_rebuild_20260929.md`。本文件是透明的实验登记与阻塞报告，不是结果报告；`timing_ablation_summary.csv` 中各指标留空，不得据此在论文中声称后期注入优于早/中期。

## 已冻结设计

按本轮“只改变开始时间”的最新约束，复用原 RC-NFGD 采样器的 MatterSim checkpoint、力修正公式、距离安全检查、CFG=2、MatterGen checkpoint、1000 个 predictor+corrector 步与代理评价器。冻结累计阈值为 Early `t≤1.0`（predictor 0–999，最多 1000 次力调用）、Middle `t≤0.5`（500–999，最多 500 次）、Late 原设置 `t≤0.02`（980–999，20 次）；No Force 为 C0。模型时间由原代码 `linspace(1,0.001,1000)` 推导，并非假设 sigma 或把采样进度误写为 t。新 cohort 预留 32 个配对 seed 750000–750031，运行前仍需完整碰撞扫描。只改变 `guidance_t_max`，其余全部冻结。**力调用预算随开始时间变化**，因此该消融不是同预算纯 timing 因果试验；必须逐臂报告 score calls、力调用、接受/回退次数和 wall time，并限制解释。旧 `docs/rcnfgd_timing_audit.md` 的等 20 次互斥窗口已被最新用户约束取代，不运行。

终态主要指标为 MatterSim MaxF；次要指标为 MatterSim mean force、Property MAE、Validity、Stable、E-hull、NUS。计划全 32 对报告均值、样本标准差及 20,000 次配对 bootstrap 95% CI。任何失败样本应保留、定位、报告，不静默删配对。Early/Middle 若全回退同样是有效负结果。

## 未执行的可核实原因

原基础环境 `/ebs/envs/liswam-py310-cu121` 仍缺失。经授权，在项目内重建了 Python 3.10.20 / PyTorch 2.4.1+cu121 的可运行环境，并在空闲 GPU 2 完成历史 seed 740000 C0/F0 隔离 replay。预注册门槛明确 FAIL：两臂元素组成、结构哈希/几何及 Property、MatterSim 力指标均不一致。故新 Timing 实验不运行；重建环境不能冒充原环境。

## 恢复条件

先恢复可执行的原冻结环境，核对 `torch`/MatterGen/MatterSim 版本与 checkpoint SHA256，做非结果性接口与单 seed 确定性检查；重新检查 GPU 和 seed 独立性；随后将新代码仅放到独立实验目录，tmux 记录生成→属性评价→MatterSim 松弛和质量评价→配对统计→图表与本报告。任何接口差异必须先审计，不能边看结果边改阈值、时间窗或评估定义。

当前问题“后期注入是否更合理”的答复是：**尚未验证**。历史 clean-estimate 诊断支持选择 late 作为原方法设置，但不等于与 early/middle 的新配对消融。即便未来差异明确，还需考虑各阶段不同 force-call 预算。
