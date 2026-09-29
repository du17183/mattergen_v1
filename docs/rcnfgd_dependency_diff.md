# RC-NFGD A1 Formal256 环境谱系与依赖差异

**关键更正：用于重建的 Python 3.10.20 / Torch 2.4.1+cu121 记录来自旧项目/旧实验，不足以代表 seed 740000 所属的 A1 Formal256。** A1 协议 `experiments/innovation2_finalization/protocol.py:17` 明列 ALM Python 路径；同目录 README 的状态命令也指定该解释器。真正启动的 `pipeline.py`、`run_generation.py` 通过 `sys.executable` 继承父解释器，所以仅凭协议常量无法证明当时进程一定使用 ALM。不过原 `pair_740000_gpu_5.log` 同时出现 `Hydra14MigrationWarning` 和“`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected`”警告，恰与现存 ALM 栈相符，而与 2.4.1 重建环境的警告形式不同。ALM 包文件先于 2026-09-13 的历史生成存在。这些是**强环境归属证据，不是当日完整 pip freeze**。

| 组件 | 旧项目/旧实验环境记录（错误重建目标） | 现存 ALM 环境（A1 强候选） | 本次失败 replay 重建环境 |
|---|---|---|---|
| Python | 3.10.20 | 当前 3.10.12 | 3.10.20 |
| PyTorch / CUDA runtime | 2.4.1+cu121 / 12.1 | 当前 2.9.0+cu128 / 12.8 | 2.4.1+cu121 / 12.1 |
| NumPy / SciPy | 1.26.4 / 1.15.3 | 2.2.6 / 1.15.3 | 1.26.4 / 1.15.3 |
| ASE / pymatgen | 3.25.0 / 2024.10.29 | 3.29.0 / 2025.6.14 | 3.25.0 / 2024.10.29 |
| MatterSim | 1.1.2 | 1.2.3 | 1.1.2 |
| Hydra / Lightning | 1.3.1 / 2.0.6 | 1.3.6 / 2.5.6 | 1.3.1 / 2.0.6 |
| PyG / torch_scatter | 2.8.0.post1 / 2.1.2+pt24cu121 | 2.8.0.post1 / 2.1.2+pt29cu128 | 2.8.0.post1 / 2.1.2+pt24cu121 |
| torch_sparse / torch_cluster / pyg_lib | pt24cu121 系列 | 0.6.18 / 1.6.3 / 0.6.0，均 pt29cu128 | 0.6.18 / 1.6.3 / 0.4.0，均 pt24cu121 |
| cuBLAS / cuDNN / nvJITLink | 原记录未逐包锁定 | 当前 12.8.4.1 / 9.10.2.21 / 12.8.93 | 当前 12.1.3.1 / 9.1.0.70 / 12.9.86 |
| spglib / monty / emmet-core | 原记录不完整 | 2.7.0 / 2025.3.3 / 0.85.1 | 2.7.0 / 2024.7.30 / 0.85.1 |

ALM 版本为本轮**现测**，不能把它无条件写为 2026-09-13 的历史精确版本；旧 2.4.1 版本也不能因曾用于其他实验而称为 A1 原环境。更严格的证据需历史启动命令或当日 `sys.executable`、`pip freeze`、wheel/hash 归档。现有 `implementation_lock.json` 锁 checkpoint 和部分源码，却没有冻结 Python/Torch/CUDA 全依赖。项目 `pyproject.toml` 的 Torch 声明也不是 A1 运行时锁定。

**最小数值试验**：以历史源码入口、同 seed 740000、同已知进程变量分别运行一次首次 score。两环境的 condition、prior、RNG 状态摘要完全相同；加载后 304 个模型 state tensor 汇总 SHA256 相同；首次 score 三字段哈希不同。最大绝对差分别为 atomic logits `4.29e-6`、position `8.38e-9`、cell `7.15e-7`。这直接表明同一权重/输入在两个软件栈下的数值执行不同；1000 步含随机采样的扩散可能放大小差异，但目前没有历史逐步张量可证明该差异**单独**造成终态 Fe2Gd2→Gd4。旧环境目标选择错误是目前最有力、可检验的解释，而非“checkpoint 不同”或“纯 RNG 初始化不同”。

证据位置：`experiments/innovation2_finalization/{protocol.py,README.md,run_generation.py,pipeline.py}`、`experiments/mattersim_late_force_guidance_formal256/logs/pair_740000_gpu_5.log`、`experiments/rcnfgd_timing_ablation_20260929/diagnostics/{rebuild_v2,alm_v2}_start_state.json`。

## 后续验证结果

用户授权后，已在上述现存 ALM 环境通过冻结 A1 原入口完整重放历史 seed 740000。C0/F0 的 extxyz SHA256 与历史逐字节相同，Property MAE、MatterSim MaxF/mean force 均在原定 `1e-6` 容差内；见 [ALM 复现报告](rcnfgd_alm_replay_740000_report.md)。这使“旧环境目标选错”成为对这次失败最有力的经验解释，但不能凭一个 seed 的 PASS 反推出历史当日完整依赖锁或所有新 seed 的可复现性。
