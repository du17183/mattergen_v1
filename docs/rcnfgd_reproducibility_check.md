# RC-NFGD 历史单样本复现门槛

**当前 seed 740000 复现门槛：ALM 环境 PASS；Timing Ablation 仍为 NOT_RUN。** 初次 Python 3.10.20 / Torch 2.4.1+cu121 重建环境的 `FAIL` 保持原样，详见 [旧环境重建报告](rcnfgd_environment_rebuild_20260929.md)。在用户授权“ALM 环境试试”后，使用冻结 A1 入口和隔离输出重新验证，同历史 C0/F0 的 extxyz SHA256 均逐字节一致；Property MAE 与 MatterSim MaxF/mean force 的最大绝对差为 `1.47e-7`，低于原定 `1e-6` 容差。完整逐项证据见 [ALM 复现报告](rcnfgd_alm_replay_740000_report.md) 和 `alm_replay_740000/gate.json`。以下历史值仍是事先验收基准。

## 冻结复现对象

使用 Formal256 的既有 seed **740000**，目标 `dft_mag_density=0.2`、固定 CFG=2、1000 步、batch=1。只在新的临时输出目录运行 C0 和 F0 各一个样本；严禁覆盖 `mattersim_late_force_guidance_formal256/generation/{C0,F0}/740000/`，严禁把复现数据并入旧 Formal256 统计。启动前再次核对 MatterGen/MatterSim checkpoint 和冻结 sampler 源码 SHA256（见环境恢复报告），核对代码 commit/配置、GPU 类型、确定性开关及 `PYTHONPATH`。

| Method | 历史结构 | extxyz SHA256 | Property MAE | MatterSim MaxF (eV/Å) | Mean force (eV/Å) | Guidance |
|---|---|---|---:|---:|---:|---|
| C0 | Fe2Gd2, 4 atoms | `d4b2f381d3de5625e36af86e7355eb4e5e5079c4e1f0866a65b61405e1f56433` | 0.005590819280932802 | 0.05343878883037957 | 0.035072523737054107 | 0 calls |
| F0 | Fe2Gd2, 4 atoms | `765e297599299fda973a6d6651f40b88f39805401fb5b84eb0e5a6206a7d1407` | 0.0058579596239093645 | 0.038746394713961456 | 0.025955640547024556 | 20 eligible / 20 accepted / 0 fallback |

值来自原 `mattersim_late_force_guidance_formal256/per_structure_metrics.csv` 与对应 `generated_crystals.extxyz` 的当前 SHA256。两组历史 score calls 都是 2000；F0 只在末 20 个 predictor 步触发。

## 验收顺序

1. 新环境导入、包版本与 CUDA ABI 完整通过；配置和 checkpoint 哈希匹配。
2. 运行历史 seed 740000 的 C0/F0，各自独立重置相同 seed，输出到全新隔离目录；记录 extxyz 哈希、原子序数、cell、分数坐标和每步 guidance trace。
3. 首先要求结构公式/原子数/原子序数完全一致，并比较 extxyz SHA256。若字节哈希不同，仅在格式元数据差异可解释时继续解析结构，要求 cell 和分数坐标逐分量绝对误差 ≤1e-7，且无原子顺序变化；不能通过优化对齐掩盖差异。
4. 对复现结构使用相同冻结属性与 MatterSim 评价管线，要求 Property MAE、MaxF、mean force 的绝对差均 ≤1e-6；F0 guidance eligible/accepted/fallback 必须分别等于 20/20/0。原始结构哈希、质量评价脚本和评估日志同时归档。
5. 任一条件失败，则 `REPRODUCIBILITY_GATE=FAIL`，停止新 seed 注册及 Timing Ablation，报告差异和可能来源；不能在看结果后放宽容差或改变环境/算法以“做过门”。

本次 ALM 复现按以上门槛为 **PASS**，但只覆盖历史 seed 740000，不能等同全部 Formal256 或未来 Timing 的可复现性。旧 2.4.1 环境的 `FAIL` 及其制品均保留；两次结果不可互相覆盖或混入旧 Formal256。当前尚未启动任何新 seed 的 Timing 实验。
