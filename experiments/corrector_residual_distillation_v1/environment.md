# 实验环境

本实验只从项目目录中的虚拟环境运行：

```bash
cd /mnt/lis-wam-data/dxl/mattergen_v1
source experiments/corrector_residual_distillation_v1/activate.sh
```

激活脚本同时把 pip、Hugging Face、Torch 和临时文件缓存定向到项目内的 `.cache/`。本次曾为恢复 Git LFS 临时安装过系统包；完成权重恢复后已卸载。仓库当前使用项目内 `.tools/git-lfs/usr/bin/git-lfs`，没有任务新增的系统级软件包或模型权重。

## 硬件与软件

- OS/kernel: Linux 5.4.250-9-velinux1u2-amd64，glibc 2.35
- GPU: 8 × NVIDIA H20，每卡 97,871 MiB
- Driver: 535.161.08
- Python: 3.10.20
- PyTorch: 2.4.1+cu121，CUDA runtime 12.1
- PyTorch Lightning: 2.0.6
- PyG: 2.8.0.post1；`pyg-lib`/scatter/sparse/cluster 为 pt24cu121 wheel
- MatterSim: 1.1.2
- ASE: 3.25.0
- pymatgen: 2024.10.29
- lmdb: 2.3.0
- Hydra: 1.3.1

`.venv` 位于项目内，并只读复用服务器既有的 CUDA/PyTorch 基础层；MatterGen、MatterSim、PyG 扩展及本实验新增 Python 包安装在 `.venv`。所有实验命令均显式使用 `.venv/bin/python`。若需要重新下载 Python 包，可优先使用清华 PyPI 镜像；PyTorch CUDA wheel 可使用阿里云 PyTorch 镜像，缓存仍应指定为项目内 `.cache/pip`。

## 项目内模型与参考数据

| 资产 | 项目相对路径 | 字节数 | SHA-256 |
|---|---|---:|---|
| MatterGen `dft_mag_density` | `checkpoints/official/hf_mattergen/checkpoints/dft_mag_density/checkpoints/last.ckpt` | 511,777,278 | `01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e` |
| MatterSim 5M | `checkpoints/official/mattersim/mattersim-v1.0.0-5M.pth` | 91,176,875 | `e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5` |
| Alex-MP reference | `data-release/alex-mp/reference_MP2020correction.gz` | 873,410,170 | `c722f72c7d0cd398fc382870f0b731425b54bd4492b089ca36e3119e49f469b5` |
| Adapter | `experiments/corrector_residual_distillation_v1/adapter_final/residual_adapter.pt` | 本地生成 | 见 benchmark CSV |

参考数据已在项目内解压为 LMDB，并完成 845,997 条记录的全量顺序审计：首条含 `material_id`，`is_ordered=True`。权重、LMDB、teacher shard、生成结构和松弛轨迹均由 `.gitignore` 排除。
