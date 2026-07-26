# CrystalREPA unconditional MatterGen reproduction

This package runs the isolated `U0` versus `R1` experiment on the official
unconditional MP-20 MatterGen checkpoint. It intentionally excludes Adaptive
CFG, conditional fields, Corrector Gating, and the earlier conditional FN-PRA
design.

## Frozen interpretation

The implementation follows the paper's four-block MatterGen setup, aligns
GemNet block 2 with symmetric element-aware NCE, uses temperature `0.1` and
alignment weight `1.0`, and fine-tunes the full backbone. The local frozen
Teacher is CHGNet 0.3.0. CHGNet is not one of CrystalREPA's ten reported
Teachers, so this is a controlled CrystalREPA-like reproduction diagnostic,
not a bit-for-bit reproduction of the paper.

## Run and resume

```bash
source /data/dxl/env.sh
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate /data/dxl/envs/mattergen_py310
cd /data/dxl/mattergen_v1
bash research/crystalrepa_repro/ops/run_repro.sh
```

The fixed tmux session is `mattergen_crystalrepa_repro`. Use:

```bash
bash research/crystalrepa_repro/ops/status_repro.sh
bash research/crystalrepa_repro/ops/resume_repro.sh
bash research/crystalrepa_repro/ops/stop_repro.sh
```

Successful atomic tasks and exact training milestones are resume-safe. The
decision-training cap is 10,000 optimizer steps. Evaluation uses paired seeds
17000–17063, the official 1,000-step full predictor/corrector sampler, and
MatterSim-5M relaxation.

## Tests

```bash
python -m pytest mattergen/tests/test_crystalrepa.py -q
python -m research.crystalrepa_repro.validate_integration
python -m research.crystalrepa_repro.validate_ddp_ea_nce
git diff --check
```

## Publishable artifacts

`collect_publishable_artifacts.py` copies reports, statistics, training curves,
generated structures, relaxed structures, hashes, and progress manifests into
`artifacts/`. It rejects model/checkpoint files, Teacher or dataset caches,
NumPy cache arrays, environments, and large logs. Every copied file is listed
with its SHA-256 digest in `artifact_manifest.json`.
