# RC-NFGD A1 seed 740000：ALM 环境复现报告

**结论：ALM_REPLAY_GATE = PASS；Timing Ablation = NOT_RUN。** 2026-09-29 只重放历史 seed 740000 的 C0/F0。先前以 Python 3.10.20 / Torch 2.4.1+cu121 重建环境得到的 `FAIL` 仍保留在 `replay_740000/gate.json`，不是本次结果，不能合并或覆盖。

## 协议与隔离

- 使用现存 `/mnt/datasets-livsyn/dxl/alm/.venv/bin/python`（本轮现测 Python 3.10.12、Torch 2.9.0+cu128），物理 GPU 2；模型、MatterSim checkpoint 和冻结力引导源码仍须通过原哈希检查。
- 新包装器 `experiments/rcnfgd_timing_ablation_20260929/replay_740000_alm_exact.py` 只把冻结 A1 `generate_frozen_pair.py` 的输出根目录重定向至 `alm_replay_740000/`，并仅传递 `--seed 740000`；历史源文件、配置、参数、评价阈值未改。worker 已知环境变量与历史启动器一致，未设置未知的 `PYTHONHASHSEED`。
- 原 C0/F0 历史目录、旧失败 replay、其他 seed 和 Timing 实验均未触碰。日志分别在 `logs/alm_replay_740000.log`、`logs/alm_gate_740000.log`、`logs/alm_gate_740000_v2.log`；首次 gate 因 ALM 环境未直接安装 CHGNet 而导入失败，v2 按冻结历史评价代码追加同一项目包路径，未改指标公式或容差。

## 预定 PASS/FAIL 检查

原结构 extxyz 字节哈希必须一致，或同原子序数/周期边界且 cell/周期坐标绝对差均 ≤`1e-7 Å`；Property MAE、MatterSim MaxF、mean force 的历史绝对差均须 ≤`1e-6`。C0/F0 各为 1000 步、2000 MatterGen score calls；F0 要求 20 eligible / 20 accepted / 0 fallback。机器可读结果见 `alm_replay_740000/gate.json`。

| 臂 | 历史及本次结构 SHA256 | Property MAE 历史 → 本次（绝对差） | MaxF 历史 → 本次，eV/Å（绝对差） | Mean force 历史 → 本次，eV/Å（绝对差） | 判定 |
|---|---|---|---|---|---|
| C0 | `d4b2f381d3de5625e36af86e7355eb4e5e5079c4e1f0866a65b61405e1f56433` | 0.005590819281 → 0.005590814989 (`4.29e-9`) | 0.053438788830 → 0.053438808374 (`1.95e-8`) | 0.035072523737 → 0.035072463247 (`6.05e-8`) | PASS |
| F0 | `765e297599299fda973a6d6651f40b88f39805401fb5b84eb0e5a6206a7d1407` | 0.005857959624 → 0.005857959624 (`0`) | 0.038746394714 → 0.038746542121 (`1.47e-7`) | 0.025955640547 → 0.025955743613 (`1.03e-7`) | PASS |

两臂原子序数、晶胞、周期坐标逐项相同，几何最大差为 0。F0 的历史与重放 `late_force_trace.csv` 均有 20 条事件，除了 `physics_seconds` 耗时列之外，所有列每行文本值完全一致；trace 文件整体哈希不同仅因为耗时。F0 实际力调用计数文件也已归档。

## 解释边界与下一步

这证明**现存 ALM 环境能够在历史 seed 740000 上精确恢复 C0/F0 生成结构，并在原评价门槛内恢复属性与代理力指标**。它高度支持此前失败是选错运行环境，而不是 checkpoint、seed 或 RC-NFGD 公式本身错误。历史没有当日完整 `pip freeze`，因此不能据一个样本的 PASS 断言所有软件包/其他 seed 全部可逐位复现。

按此前约定，只有 `PASS` 才有资格进入 Timing；本轮用户只要求“ALM 环境试试”，故**未启动** 750000–750031 或 Early/Middle/Late。若随后决定继续，须沿用冻结的 32 paired seeds、注入时间和评价协议，并将该 PASS 证据与历史失败记录同时保存，不得基于本结果调参。
