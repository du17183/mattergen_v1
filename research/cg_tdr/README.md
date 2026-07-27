# CG-TDR Phase 0

`Confidence-Gated Terminal Denoising Refiner` performs one deterministic,
bounded correction after the complete MatterGen predictor/corrector trajectory.
It freezes MatterGen and Adaptive CFG, never changes atomic numbers, and does
not import CHGNet or MatterSim at inference.

## Serial authorization

Phase A completed before this branch was created:

- same-method, same-seed three-repeat determinism passed;
- A0 and RP-QTFG used identical per-call random tapes;
- different seeds produced distinct priors and structures;
- the RP-QTFG post-hoc 16-seed check violated the frozen E-hull safety gate;
- `PHASE_A_GATE_FOR_CG_TDR=True`.

The full random-tape evidence is kept on `experiment/seed-random-tape-audit`.
The compact gate summary is archived in `artifacts/seed_audit_final_summary.json`.

## Module

- `model.py`: invariant scalar messages multiplied by periodic edge directions,
  yielding a rotation-equivariant Cartesian position residual; a weaker
  lattice-basis symmetric strain head; independent confidence gates; strict
  position/cell trust bounds.
- `sampler.py`: mirrors the frozen A0 PC loop, retains only the final three
  mean states, captures one conditional GemNet terminal feature, and applies
  CG-TDR exactly once.
- `run_teacher_sample.py`: one deterministic A0 structure and terminal feature
  record.
- `launch_teacher_generation.py`: resume-safe 8-GPU generation for seeds
  30000–30511.
- `build_teacher_labels.py`: offline CHGNet 0.3.0 candidate selection. MatterSim
  is deliberately absent.
- `train.py`: seed 3100, 100-step overfit, 1000-step smoke, at most 5000 steps;
  only CG-TDR parameters are optimized.

## Frozen split

| split | seeds | structures |
|---|---:|---:|
| train | 30000–30383 | 384 |
| validation | 30384–30447 | 64 |
| test | 30448–30511 | 64 |

Evaluation seeds 23000–23031 are disjoint. The workflow stops after the
32-seed decision and never starts 64- or 256-seed validation automatically.

## Commands

```bash
/data/dxl/reports/cg_tdr/phase0/run.sh
/data/dxl/reports/cg_tdr/phase0/status.sh
/data/dxl/reports/cg_tdr/phase0/stop.sh
```

After feature generation:

```bash
CUDA_VISIBLE_DEVICES=0 \
  /data/dxl/envs/fn_pra_teacher/bin/python \
  research/cg_tdr/build_teacher_labels.py

CUDA_VISIBLE_DEVICES=0 \
  /data/dxl/envs/mattergen_py310/bin/python \
  research/cg_tdr/train.py
```
