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

质量指标来自 MatterSim-5M surrogate relaxation，不是 DFT。

网络安装如有需要优先使用清华 PyPI 镜像，但所有下载目录和 cache 必须显式指向
项目目录，禁止写入根目录。
