# RC-NFGD 环境恢复状态（h20-1，2026-09-29）


**最新更新（2026-09-29）：** 初次 Python 3.10.20 / Torch 2.4.1+cu121 重建环境复现为 `FAIL`，但发现其环境记录属于旧实验；现存 ALM 环境经用户授权只重放历史 seed 740000，C0/F0 结构哈希与历史逐字节一致，原 Property/MatterSim 门槛全部 **PASS**。当前状态以 [ALM 复现报告](rcnfgd_alm_replay_740000_report.md) 为准；[旧重建报告](rcnfgd_environment_rebuild_20260929.md) 保留 FAIL 过程。Timing 未运行。下文是**已过时的授权前审计记录**，其中将 2.4.1 栈称为 A1 原环境的判断已被纠正。

## 旧授权前审计记录（不代表当前状态）

**结论：原实验环境未找到、未恢复；Reproducibility Gate = FAIL（环境前置条件）。** 本轮仅做只读检查，未安装 PyTorch、未创建替代环境、未使用其他用户环境、未更改 CUDA 或原实验文件。

## 检查结果

| 检查项 | 历史原环境 | 当前 h20-1 |
|---|---|---|
| Python | 3.10.20 | 系统 Python 3.10.12；归档 `.venv/bin/python` 不是可执行解释器，而是指向旧 `/ebs/envs/liswam-py310-cu121/bin/python` 的文本 |
| PyTorch/CUDA runtime | 2.4.1+cu121 / 12.1 | 系统和项目 `.venv` 均无可导入的 `torch`；缺失旧节点的基础层 |
| NVIDIA driver/GPU | 535.161.08 / 8×H20 | driver 535.161.08 / 8×H20；审计时 GPU0/1 为其他服务占用，2–7 显存空闲 |
| torchvision/torchaudio | 历史版本未在归档中独立记录，位于失联基础层 | 项目 `.venv` 没有可验证的原版本；不得按配套版本推测成事实 |
| 项目依赖 | MatterSim 1.1.2、PyG 2.8.0.post1 与 pt24cu121 扩展、Hydra 1.3.1、ASE 3.25.0、pymatgen 2024.10.29 等 | 项目 `.venv` 保留多数依赖及元数据，但依赖缺失的 PyTorch 基础层，不能运行原 sampler |
| Conda | 未发现原环境镜像 | `conda env list` 报 `conda: command not found`；常见 Conda 根路径不存在 |
| Docker | 未记录原 MatterGen 镜像 | `docker images`/`docker ps -a` 只有 OpenLane 镜像和容器，非本项目；未触碰运行容器 |
| 项目清单 | 历史 `environment.md` 与 `.venv/pyvenv.cfg` | 三个 MatterGen 项目根均有 `pyproject.toml`；未找到本实验的 `environment.yml`、requirements、conda-lock、uv.lock 或 Dockerfile |
| 历史路径 | `/ebs/envs/liswam-py310-cu121`、`/mnt/lis-wam-data/dxl/mattergen_v1` | `/ebs` 与 `/mnt/lis-wam-data` 均不存在，`findmnt /ebs` 无挂载；不能假造替代目录 |

特别注意：当前 `pyproject.toml` 对 Linux 声明 `torch==2.2.1+cu118`，与历史运行时的 **2.4.1+cu121** 不同；直接按项目声明安装会改变方法运行环境。项目区另有 PyTorch 2.9.0+cu128 或 2.5.1+cu121 的其他任务环境，也不是原环境，本轮未借用。MatterGen、MatterSim checkpoint 与冻结 sampler 的 SHA256 已在 `docs/rcnfgd_environment_recovery.md` 记录并核对。

## 恢复方案与风险

首选由管理员恢复旧 `/ebs` 基础环境的只读挂载和所需路径映射，或提供可证明来自同一环境的容器/快照。恢复后先核对 Python 3.10.20、PyTorch 2.4.1+cu121、`torch.version.cuda == 12.1`、torchvision/torchaudio 与 PyG 扩展 ABI、MatterSim/MatterGen 导入、checkpoint 与源码哈希、H20 CUDA 可用性，再做 seed 740000 的结构和指标复现。不能仅凭版本号相似就认定可比，尤其不能用当前 `pyproject.toml` 的 cu118 栈替代。

若原挂载/镜像无法找回，任何重新组装环境都属于新的待验证环境；本轮明确禁止直接安装新 PyTorch，因此不执行。需要外部提供原环境来源后才能继续。未满足复现门槛时，Timing Ablation 严格停在 `NOT_RUN`，不以空闲 GPU 为启动理由。
