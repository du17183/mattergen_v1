# Corrector residual distillation formal256

This is the confirmatory, no-tuning evaluation of the already frozen
`V2_Frozen_Atomic75_Late30` method against the original MatterGen `C0` sampler.
It starts from V2 HEAD `1b8b62284f694b08d778895da345c708204ec2db` and uses
256 strictly paired seeds 67000–67255. No V1, Skip, Reuse, A0, or alternative V2
candidate is part of the formal experiment.

The V2 runtime is derived by reading the exact immutable file
`experiments/corrector_residual_distillation_v2/v2_frozen_config.yaml`; it is not
restated as an independently editable sampler configuration. The formal protocol
validates all model/config hashes before generation and quality analysis.

The primary efficiency result is a separate same-GPU, batch-size-1 rerun on
physical GPU0 for paired seeds 67000–67015. The 8-H20 run uses fixed 32-seed
shards and alternating C0/V2 waves; its throughput is a system measure only.

Quality is evaluated for all 512 structures with the frozen MatterSim-5M
surrogate, `EXPCELLFILTER`, and `fmax=0.05 eV/Å`. It is not DFT:
`DFT_VERIFIED=False`.

The seed interval, 20,000 paired-bootstrap analysis, non-inferiority margins,
force thresholds, tail guardrails, and PASS/BORDERLINE/FAIL rule were frozen and
committed before any formal generation result was read.

## Reproduction

```bash
cd /mnt/lis-wam-data/dxl/mattergen_v1
source .venv/bin/activate
export TMPDIR=/mnt/lis-wam-data/dxl/mattergen_v1/.tmp
export XDG_CACHE_HOME=/mnt/lis-wam-data/dxl/mattergen_v1/.cache
export HF_HOME=/mnt/lis-wam-data/dxl/mattergen_v1/.cache/huggingface
export TORCH_HOME=/mnt/lis-wam-data/dxl/mattergen_v1/.cache/torch

python -m pytest -s \
  mattergen/diffusion/tests/test_residual_distillation.py \
  mattergen/diffusion/tests/test_guidance_schedule.py \
  research/corrector_distillation/test_formal256_protocol.py -q

python -m research.corrector_distillation.run_formal256_generation --mode main
python -m research.corrector_distillation.aggregate_formal256_generation --mode main
python -m research.corrector_distillation.run_formal256_generation --mode single-h20
python -m research.corrector_distillation.aggregate_formal256_generation --mode single-h20
python -m research.corrector_distillation.formal256_speed_statistics
```

MatterSim precomputation and CPU metric commands are recorded verbatim in
`final_report.md` after completion. Large structures, trajectories, logs, caches,
and weights remain project-local and are excluded from Git.
