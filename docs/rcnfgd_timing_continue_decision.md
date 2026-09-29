# RC-NFGD Timing Ablation 继续/停止决策

**seed 740000 的 ALM 复现门槛 = PASS；Timing Ablation = NOT_RUN。** 这解除此前“环境不可复现”的技术前置阻断，但不等于已经授权或完成 32 个新 paired seeds。用户本轮只要求试用 ALM 环境，因此本轮止于门槛检查。

## 门槛证据

- 初次 Python 3.10.20/Torch 2.4.1+cu121 环境复现为 `FAIL`，旧 `replay_740000/gate.json` 永久保留。其目标环境选自旧项目记录，并非 A1 Formal256 的最佳环境归属证据。
- 现存 ALM（本轮 Python 3.10.12/Torch 2.9.0+cu128）通过冻结 `generate_frozen_pair.py` 原入口，仅将输出根目录隔离到 `alm_replay_740000/`。C0/F0 extxyz SHA256 分别与历史完全一致；原子序数、晶胞和周期坐标差均为 0。
- 同原定绝对容差 `1e-6` 的 Property MAE、MatterSim MaxF、mean force 六项均 PASS，最大差 `1.47e-7`。F0 的 20 次引导事件在除运行耗时外的所有 trace 字段均与历史一致，且 20/20 accepted、0 fallback。机器结果：`alm_replay_740000/gate.json`；详见 [ALM 复现报告](rcnfgd_alm_replay_740000_report.md)。
- 历史没有保存当日完整 pip freeze，这一 PASS 证明该历史 seed 的结构/评价可恢复，不证明所有 seed 或所有第三方包逐位同一。

## 下一步边界

按既有约束，只有复现门槛 PASS 才**可以考虑**运行预先设计的 Timing Ablation。若用户明确继续，应在开跑前核对并固定 ALM 解释器、冻结源码/模型哈希、32 个未见过的 paired seeds、No Force/Early `t≤1.0`/Middle `t≤0.5`/Late `t≤0.02`、force call 预算及既定指标/CI；不得因为本次 replay 修改力权重、阈值、seed 或评价口径。若执行过程中环境变化或出现协议偏差，停止并报告，不能用趋势替代门槛。

**当前没有启动 Timing，也没有生成 750000–750031。** 创新点2当前正文仍只能基于已有历史 C0/F0、CHGNet 和 Pareto 证据表述；不能提前写“后期注入优于 Early/Middle”、DFT 稳定性或全面优于 POST。
