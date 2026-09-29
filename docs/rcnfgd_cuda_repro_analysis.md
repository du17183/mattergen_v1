# RC-NFGD CUDA、确定性与运行时审计

## 已核实的执行信息

A1 Formal256 历史 seed 740000 在物理 GPU 5（NVIDIA H20）运行；当前同机 driver 为 535.161.08，最小诊断使用空闲的物理 GPU 2（同型 H20）。**不能再把 Torch 2.4.1/cu121 称为 A1 已证实原运行时。** A1 协议显式指向 `/mnt/datasets-livsyn/dxl/alm/.venv/bin/python`，历史 worker log 的 Hydra 1.4 迁移警告和 Torch 强制 `weights_only=False` 警告与现存 ALM 环境相符。ALM 现测为 Python 3.10.12、Torch 2.9.0+cu128（CUDA runtime 12.8），cuDNN 9.10.2.21、cuBLAS 12.8.4.1；失败 replay 的重建栈则为 Python 3.10.20、Torch 2.4.1+cu121（runtime 12.1）、cuDNN 9.1.0.70、cuBLAS 12.1.3.1。历史并未保存这几项的当日二进制哈希，因此 ALM 是强候选而非形式证明。

冻结 `generate_formal32_pair.py` 与本轮单步诊断均启用：`torch.use_deterministic_algorithms(True)`、`cudnn.deterministic=True`、`cudnn.benchmark=False`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`，Torch intra-op=2、interop=1。历史 worker 还配置 `OMP_NUM_THREADS=2`、`MKL_NUM_THREADS=2`、`OPENBLAS_NUM_THREADS=2`、`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`；最小诊断在两环境复用这些已知变量并移除了此前 replay wrapper 额外设置的 `PYTHONHASHSEED=740000`。历史父进程是否另有 `PYTHONHASHSEED` 未归档。

确定性标志**只约束当前软硬件栈内受支持的运算**，不能保证 PyTorch/PyG/torch-scatter/cuBLAS/cuDNN 在不同版本间逐位一致，也不保证不同 GPU 物理卡上的完整扩散轨迹一致。两现存环境在相同历史源码入口、相同 seed、相同模型 state tensor、相同 condition/prior 和 RNG 指纹下，首次 denoiser score 三字段已经有微小数值差异：atomic logits 最大绝对差 `4.29e-6`，position `8.38e-9`，cell `7.15e-7`。forward 后 RNG 指纹仍一致。这支持“跨栈数值执行不同”，不是 RNG 初始化失败的证据；尚未做算子级定位，也不能断言单个 CUDA kernel 就是根因。

原历史与重建 replay 均执行 1000 步、2000 score calls，F0 在相同 step 980/`t=0.020000001415610313` 首次力调用；C0 终态已不同，且 F0 首次力修正**之前**的 MatterSim 能量/MaxF 已不同。因此后期力公式、MatterSim 终点评估或 extxyz 导出不是唯一分叉来源。未来若用户允许继续定位，应保留原 `FAIL` 门槛，先在 ALM 强候选环境按历史 worker 入口复验 seed 740000，再视结果比较 Torch build 配置、TF32、PyG 扩展 wheel、单算子输出；不得直接运行 Timing 或新 seed。

## ALM 完整重放后续结果

在同型号 H20 的物理 GPU 2，现存 ALM 栈经冻结 A1 原入口重放历史 seed 740000，C0/F0 最终结构字节哈希均与 GPU 5 上的历史结果相同；F0 除 `physics_seconds` 外 20 条引导记录逐字段一致。原定结构、Property 与 MatterSim 门槛已 `PASS`。因此跨物理 H20 卡并未阻止该 seed 复现；失败的 2.4.1 重建栈与 ALM 栈数值路径不同仍是事实，但不能把本轮差异归咎于“GPU 2 对比 GPU 5”本身。详见 [ALM 复现报告](rcnfgd_alm_replay_740000_report.md)。
