# RC-NFGD 原实验环境恢复审计（h20-1）


**后续更新：** 原始 `/ebs` 基础层仍未找到，但经授权已在项目内重建可运行的同版本环境；历史 seed 740000 的实际结构和指标复现门槛 **FAIL**，Timing 未运行。当前结论和证据见 [rcnfgd_environment_rebuild_20260929.md](rcnfgd_environment_rebuild_20260929.md)。以下保留授权前的环境只读审计，不代表最新执行状态。

状态：**原运行环境未恢复；不得进行新的 Timing Ablation。** 审计日期：2026-09-29。本轮没有安装 PyTorch、创建替代环境、修改 checkpoint 或运行生成。

## 1. 原环境证据

历史记录 `mattergen_v1_matersim_guidance/experiments/corrector_residual_distillation_v1/environment.md` 与 `.../formal256/environment.md` 一致记录：Python **3.10.20**，PyTorch **2.4.1+cu121**（捆绑 CUDA runtime **12.1**），NVIDIA driver **535.161.08**，MatterSim **1.1.2**，PyTorch Lightning **2.0.6**，PyG **2.8.0.post1** 与 pt24cu121 扩展，NumPy **1.26.4**，ASE **3.25.0**，pymatgen **2024.10.29**，Hydra **1.3.1**。历史 `.venv` 使用 `include-system-site-packages=true`：PyTorch/CUDA 来自旧节点基础层，MatterGen/MatterSim/PyG 等来自项目虚拟环境。其激活脚本将根路径写死为旧挂载 `/mnt/lis-wam-data/dxl/mattergen_v1`。

原 MatterGen/MatterSim checkpoint SHA256 分别为 `01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e` 和 `e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5`；当前文件逐一复算匹配。冻结 `late_force_sampler.py` SHA256 为 `7a1598338e305986f03f232b719cdbb90687ba3388b8827ace8aa191bb8d9523`。项目 `pyproject.toml` 对 Linux 声明的是 `torch==2.2.1+cu118`，**与历史实际 2.4.1+cu121 不一致**，不能把 `pip install -e .` 当成等价环境恢复。未找到针对原 RC-NFGD 的完整 lockfile。

## 2. 当前机器状态

- 主机 h20-1，8×H20，当前 driver 535.161.08 与历史记录一致；审计时 GPU0/1 被其他服务占用，GPU2–7 显存显示空闲。GPU 空闲不等于实验环境可用。
- 归档 `/mnt/datasets-livsyn/dxl/mattergen_v1/.venv/bin/python` 是 39 字节的普通文本文件，内容为旧解释器绝对路径 `/ebs/envs/liswam-py310-cu121/bin/python`；`/ebs` 当前不存在。`.venv` 内有 MatterSim 1.1.2、PyG pt24cu121 等包元数据，但没有 `torch` 包；原设计依赖失联的系统基础层。
- `/usr/bin/python3` 和现有论文绘图环境均无法 `import torch`。当前 PATH 无 `conda`；Docker 只列出与本项目无关的 OpenLane 镜像且无在运行容器；项目范围未找到能直接启动的原 Docker/Conda/lock 环境。系统可见的其他用户服务环境不属于本实验，未借用或干预。
- 因缺失与历史一致的解释器/PyTorch 基础层，现阶段连历史 seed 的生成 sanity check 都不能合法启动。

## 3. 恢复顺序与验收门槛

首选由环境管理员重新挂载/提供旧 `/ebs/envs/liswam-py310-cu121` 只读基础层及旧 `/mnt/lis-wam-data` 路径映射，或提供经过版本、wheel 哈希、CUDA/PyG ABI 核验的原环境镜像。先在独立路径做 `python --version`、`import torch; print(torch.__version__, torch.version.cuda)`、`import mattersim, hydra, torch_geometric`、CUDA 可用性与项目导入检查，**不修改原归档 `.venv` 或全局环境**。

随后执行 `docs/rcnfgd_reproducibility_check.md` 规定的旧 seed 单样本门槛；只要结构、属性或力评估不一致，停止，不注册/生成新 timing seed。最新“只改变开始时间”的方案已在无新结果时冻结为 Early `t≤1.0`、Middle `t≤0.5`、Late 原设置 `t≤0.02`，各臂均持续到采样终点，预留 32 个配对 seed 750000–750031；实际执行前仍须完成 seed 碰撞扫描和 GPU 占用复核，并使用 tmux/日志。旧 `docs/rcnfgd_timing_audit.md` 的**等 20 次互斥窗口**设计改变了停止时间，已被取代。新方案 force-call 上限分别为 1000/500/20，必须作为预算混杂因素如实报告，不能写成同预算时机因果比较。

不采用的捷径：直接安装一套新版 PyTorch、改用 pyproject 所列 cu118 组合、借用他人服务环境、只因代码能跑就跳过历史样本复现、在看到新结果后调整注入阈值。这些均不能证明与历史 RC-NFGD 的可比性。
