# Calibrated Safe-Selection — Initial Audit

Audit date: 2026-09-15 (Asia/Shanghai)

## Repository isolation

- Main repository: `/mnt/datasets-livsyn/dxl/mattergen_v1`
- Main HEAD: `1af301c5e534ff6b9b9f96ea5dab48036d9c46af`
- Main branch: `experiment/raab-sc-p0`
- Main contains user-owned untracked research outputs; none are modified here.
- New worktree: `/mnt/datasets-livsyn/dxl/mattergen_v1_safe_cfg`
- New branch: `experiment/calibrated-safe-selection-cfg`
- Parent code commit: `a4fda8a573ee0e372d3b53064e235ae87f0e670c`
- Old Risk CFG and Oracle worktrees are untouched.
- All five stashes remain present (`stash@{0}` through `stash@{4}`).

## Resource audit

- GPU 0/3/4/6: occupied by an unrelated four-GPU video diffusion task.
- GPU 2: occupied by the longstanding Python task; prohibited for this experiment.
- GPU 7: occupied by a VLLM engine.
- GPU 1 and GPU 5: 3 MiB used, 0% utilization at the initial audit; eligible.
- Host tmux: no server/session at the initial audit.
- The pipeline will use only explicitly supplied free GPUs and will never use GPU 2.

## Seed and split audit

- Historical unique seeds detected: 2,954 across 466 seed-bearing files.
- Scan scope: `experiments/`, `diagnostics/`, and `research/` in all `mattergen_v1*` worktrees.
- New registered seeds: 128 dataset + 32 P0 + 256 Formal256 = 416 unique.
- New-vs-historical overlap: 0.
- Dataset split: Train 64, Calibration 26, Validation 19, Test 19.
- Split unit: complete trajectory seed; all four splits are disjoint.

## Verification before freeze

- Python source compilation: passed.
- Required dependencies: scikit-learn 1.7.2, SciPy 1.15.3, PyTorch 2.9.0+cu128.
- Inherited full-outcome sampler tests: 4 passed.
