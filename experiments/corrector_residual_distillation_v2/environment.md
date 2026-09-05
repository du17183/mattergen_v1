# V2 environment

- Host GPU: 8 × NVIDIA H20, 97,871 MiB each
- NVIDIA driver: 535.161.08
- Project root: `/mnt/lis-wam-data/dxl/mattergen_v1`
- Virtual environment: `/mnt/lis-wam-data/dxl/mattergen_v1/.venv`
- Python: 3.10.20
- PyTorch: 2.4.1+cu121
- CUDA runtime bundled with PyTorch: 12.1
- NumPy: 1.26.4
- ASE: 3.25.0
- pymatgen: 2024.10.29
- Hydra: 1.3.1
- MatterSim: 1.1.2
- MatterGen checkpoint SHA256:
  `01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e`
- MatterSim-5M checkpoint SHA256:
  `e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5`
- V1 Adapter checkpoint SHA256:
  `fae136412448be0b82267dd1a40d0168cf4847130d41cfa4295c8def715108c4`
- V2 frozen Adapter checkpoint SHA256:
  `a2fd0f17c27b760958b60c1d1386577883635a34fb2101eb08e38fd8472702c7`
- V2 frozen risk calibration file SHA256:
  `dff93d12d3637e260a3cfe0b9f418b39e809a39e13deb6d185ea0b4c31d3fd9f`
- Alex-MP MP2020-corrected reference SHA256:
  `c722f72c7d0cd398fc382870f0b731425b54bd4492b089ca36e3119e49f469b5`

质量指标来自 MatterSim-5M surrogate relaxation，不是 DFT。

Stage-C 质量弛豫在 GPU0 顺序运行，`EXPCELLFILTER`、`fmax=0.05 eV/Å`。
由于 PyTorch 2.4.1+cu121 在 H20 上对 MatterSim batched 3×3 cell 的
`torch.linalg.det` 间歇触发 `CUDA driver error: invalid argument`，正式 runner
使用可微解析 3×3 determinant。64 个随机 float64 矩阵对照 CPU 实现的最大绝对/
相对误差分别为 `8.88e-16`/`3.81e-15`，且 autograd 梯度全部有限。

网络安装如有需要优先使用清华 PyPI 镜像，但所有下载目录和 cache 必须显式指向
项目目录，禁止写入根目录。
