# Formal256 environment

- Host: 8 × NVIDIA H20, 97,871 MiB each
- NVIDIA driver: 535.161.08
- Project root: `/mnt/lis-wam-data/dxl/mattergen_v1`
- Virtual environment: `/mnt/lis-wam-data/dxl/mattergen_v1/.venv`
- Python: 3.10.20
- PyTorch: 2.4.1+cu121
- CUDA runtime bundled with PyTorch: 12.1
- NumPy: 1.26.4
- SciPy: 1.15.3
- ASE: 3.25.0
- pymatgen: 2024.10.29
- Hydra: 1.3.1
- MatterSim: 1.1.2
- Precision: repository default, unchanged from frozen V2
- Batch size: 1
- Deterministic sampling: true
- Project-local caches: `.tmp`, `.cache`, `.cache/huggingface`, `.cache/torch`

No environment, checkpoint, model weight, or cache is stored under `/root`.
Dependency downloads, if required, use a China-accessible mirror such as the
Tsinghua PyPI mirror and still write only under the project root. No dependency
download was required for this run.

MatterSim-5M is a surrogate evaluator. `DFT_VERIFIED=False`.
