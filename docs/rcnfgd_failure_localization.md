# seed 740000 失败定位与最小化测试

**后续验证（2026-09-29）：ALM 环境经冻结 A1 原入口完整重放，同一历史 seed 的 C0/F0 extxyz SHA256 均与原结果完全一致，Property/MatterSim 原门槛 PASS。** 因而本报告以下“原环境尚未完整重放”之类语句是诊断阶段的时点记录；实际恢复结论以 [ALM 复现报告](rcnfgd_alm_replay_740000_report.md) 为准。旧 2.4.1 环境 FAIL 不删除。F0 20 条 force trace 除耗时字段外全部逐字段一致，强烈支持旧 replay 失败来自运行时错配；历史未保存前几步状态，无法指定旧环境分叉的**精确** timestep。

## 定位结论与证据等级

**已证实**：旧环境目标选错；失败 replay 的 Python 3.10.20/Torch 2.4.1+cu121 与 A1 历史协议和 worker 警告不匹配。**已实测**：现存 ALM（A1 强候选）与重建环境在相同历史源码入口下，seed 740000 的 condition、初始 prior、加载模型权重及 RNG 摘要相同，但**首次 denoiser score** 出现微小数值差异。**未证实**：旧 2.4.1 replay 与历史运行的首次实际分叉步号，因为历史没有保存中间 tensor。后续 ALM 独立复现已 PASS；原 2.4.1 记录仍是 FAIL，不可改写。

| 阶段 | 原 A1 历史保存内容 | 本轮两个现存环境的最小对照 | 判定 |
|---|---|---|---|
| 0. checkpoint/加载 | checkpoint SHA256 `01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e`；无当日加载后张量摘要 | 304 个 model state tensor 汇总 SHA256 均为 `2d505e7fc6d1ae3102e4d8881dc493852f8690b4db7dafaf86901250d2d99520` | 当前两环境加载权重相同 |
| 1. condition/RNG | 历史只有 target 0.2、最终 4 原子、seed；无 condition/prior 或 RNG 快照 | condition、初始 prior 三字段与 reseed/condition/prior/首次 score 后四类 RNG 摘要逐字节一致 | 当前两环境首次可见分叉不在初始化或 prior |
| 2. 首次 score，`t=1.0` | **没有历史 score** | 原子、位置、晶胞 score 哈希均不同；最大绝对差 `4.29e-6`、`8.38e-9`、`7.15e-7` | **当前两个运行时之间**最早观测到的分叉；同权重/同输入数值执行不同 |
| 3. 固定 noise 的单步 scheduler | 历史无 `x_t`、score/noise；本轮未执行 | 前一阶段已分叉，直接比较自由运行的单步只会混合 score 与 transition 差异 | 若后续需要，须保存共同 score/固定 noise 才能孤立 transition |
| 4. F0 第一次力反馈 | 历史 step 980、`t=0.020000001415610313`；修正前 MatterSim energy -45.071213 eV、MaxF 0.053565 eV/Å | 失败 replay 同 step/t，但修正前 energy -56.238422 eV、MaxF 0.005932 eV/Å | 原历史与失败 replay **最迟在首次力注入前**已不同 |
| 5. 最终结构与评价 | 历史 C0/F0 为 Fe2Gd2 | 失败 replay C0/F0 为 Gd4，lattice、coordinate、Property MAE、力均不符 | 已排除“只是导出格式或终点评估器差异” |

最小测试脚本为 `experiments/rcnfgd_timing_ablation_20260929/diagnose_start_state.py`；证据为同目录 `diagnostics/{rebuild_v2,alm_v2}_start_state.json` 与 `*_first_score.npz`。它只加载模型、抽历史 seed 的 condition/prior、计算一次 `t=1.0` score，没有 1000 步生成、力调用、新 seed 或 Timing。比较使用同一历史源码入口与已知 worker 进程变量，在空闲 H20 卡上串行执行。两环境 score 差异很小，是否在长链中被放大为成分变化仍属于合理机制推断，**非历史逐步证据**。

## 诊断阶段提出的后续测试顺序（第 1 项现已完成）

1. **已完成并 PASS。** 用历史原 worker 的完整入口/进程变量和 ALM 解释器，仅对**已有 seed 740000**重新跑原 C0/F0 门槛。先核对 `sys.executable`、当日可得依赖、GPU 映射，保留原 `FAIL` 和历史结果不可覆盖。只有结构/Property/MatterSim 全部按原阈值 `PASS`，才可说运行环境已恢复。
2. 若 ALM replay 仍失败，冻结相同 `x_t`、condition、t、权重，跨环境比较单个 denoiser（条件与无条件分支）及 PyG/scatter 子模块；随后用共同 score 和显式固定 noise 比较一次 predictor/corrector/scheduler 更新，定位算子或配置。不可把不同自由轨迹直接当成“同一步”比较。
3. 若数值轨迹吻合但最终结构不同，再在同一数值终态上检查 ASE/pymatgen 序列化；现有修正前 step 980 差异已排除“仅后处理”。

该顺序不会改变 seed、checkpoint、force 权重、threshold、统计阈值或原 A1 结论。历史中间状态缺失意味着即使确认 ALM 是正确环境，也无法仅凭已有日志精确指定 2026-09-13 那次运行的**第一**个分叉 timestep。
