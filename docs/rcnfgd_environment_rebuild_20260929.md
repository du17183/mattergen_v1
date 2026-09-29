# RC-NFGD 等价环境重建与 seed 740000 复现门槛（2026-09-29）

**本报告仅记录初次 2.4.1/cu121 重建环境的失败尝试，不能作为当前总门槛状态。** 该尝试的 `FAIL` 记录原样保留。后续在 ALM 环境以原 A1 入口重放同一历史 seed 740000，C0/F0 结构哈希与原始结果一致，结构/Property/MatterSim 门槛均 `PASS`；详见 [ALM 复现报告](rcnfgd_alm_replay_740000_report.md)。Timing 仍为 `NOT_RUN`。

## 重建范围与核对

- 项目内新环境：`/mnt/datasets-livsyn/dxl/mattergen_v1_matersim_guidance/.envs/rcnfgd_equiv`。未覆盖原 `/mnt/datasets-livsyn/dxl/mattergen_v1/.venv`，未安装系统 PyTorch/CUDA，也未借用其他用户环境。
- 从 SHA256 校验过的 Python 3.10.20 源码构建项目内解释器；缺失的 SQLite/lzma 开发头文件仅解压到临时构建目录。验证 Python 3.10.20、NumPy 1.26.4、SciPy 1.15.3、PyTorch 2.4.1+cu121、CUDA runtime 12.1、torchvision 0.19.1+cu121、torchaudio 2.4.1+cu121，H20 CUDA 可用。
- 归档项目包的 MatterGen、MatterSim 1.1.2、PyG 2.8.0.post1 与 pt24cu121 扩展、Lightning 2.0.6、ASE 3.25.0、pymatgen 2024.10.29、CHGNet 0.3.0 均可导入。原 `pyproject.toml` 中 cu118 的声明与历史实验不符，未据此替换历史 cu121 版本。
- MatterGen checkpoint SHA256 `01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e`，MatterSim checkpoint SHA256 `e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5`，冻结力引导源码 SHA256 `7a1598338e305986f03f232b719cdbb90687ba3388b8827ace8aa191bb8d9523`，以及 MatterGen sampler 三个冻结文件哈希，均已核对。

原 Python/CUDA 基础层并未找回；本环境由同版本重装与归档包叠加而成，若干未在历史记录中锁定的转依赖和进程环境仍可能不同。版本一致不能替代样本复现。

## 门槛执行与结果

隔离目录 `experiments/rcnfgd_timing_ablation_20260929/replay_740000/` 中使用历史 seed 740000 重建 C0/F0。固定 target=0.2、CFG=2、1000 个 predictor 与 corrector 步、batch=1；F0 为原 `t≤0.02` 且 20 次 eligible/20 次 accepted/0 次 fallback，两臂各 2000 MatterGen score calls。旧 Formal256 生成、评价与统计文件未覆盖。

| 方法 | 结构 | 历史 / 重建 Property MAE | 历史 / 重建 MatterSim MaxF (eV/Å) | 历史 / 重建 mean force (eV/Å) | 结构门槛 |
|---|---|---:|---:|---:|---|
| C0 | 历史 Fe2Gd2；重建 Gd（4 原子） | 0.005590819 / 0.013646861 | 0.053438789 / 0.002374589 | 0.035072524 / 0.001840623 | FAIL |
| F0 | 历史 Fe2Gd2；重建 Gd（4 原子） | 0.005857960 / 0.013648414 | 0.038746395 / 0.002209704 | 0.025955641 / 0.001722112 | FAIL |

两臂 extxyz SHA256 均不一致。C0 最大晶胞分量差 2.0786659 Å、周期坐标差 2.3510317 Å；F0 分别为 2.0788635 Å 与 2.3462565 Å，且原子序数不一致，远超事先固定的 1e-7 Å 容差。三个指标的绝对差均超过 1e-6。逐项数值及原始/重建哈希以机器可读的 `replay_740000/gate.json` 为准；该文件状态为 `FAIL`，门槛命令退出码 1。

## 边界与可能原因

模型、checkpoint 和冻结 sampler 哈希一致，因此已排除这些文件被替换；但不能仅凭本次结果定位单一根因。历史基础层未保存完整 lock/snapshot，重建环境的部分转依赖（例如 CUDA nvJITLink 及通用 Python 包）和历史进程环境不能证实完全相同；冻结生成器使用 CUDA/PyG 数值路径，残余非确定性也需单独证明，不能先验归因。此次独立 replay 包装器并非原 `generate_frozen_pair.py` 的逐字运行，进程级环境亦未完全复制历史 worker；这也是须保留的复现风险。**对于本报告所述 2.4.1 环境尝试**，结构和评价指标明确不符合预注册 PASS 条件；不可把该次 FAIL 改写为 PASS。ALM 后续 PASS 是不同环境、同一历史 seed 的独立隔离复现。

在本次 2.4.1 环境尝试结束时，因 FAIL 停止；后续 ALM 重放已另行 PASS，但截至当前仍没有注册/生成 seed 750000–750031，也没有 Early/Middle/Late Timing 数据。旧 RC-NFGD 参数、阈值、评价指标和历史结论未改；本次 FAIL 不改写、不调容差。

日志：`experiments/rcnfgd_timing_ablation_20260929/logs/` 下 `pytorch_install_local_wheels.log`、`runtime_dependencies_install.log`、`replay_740000_retry.log`、`gate_740000.log`；所有路径相对于 RC-NFGD 代码项目根目录。
