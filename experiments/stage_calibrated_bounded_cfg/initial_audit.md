# Initial audit — Stage-Calibrated Bounded Adaptive CFG

- Audit date: 2026-09-15 (Asia/Shanghai)
- Worktree: `/mnt/datasets-livsyn/dxl/mattergen_v1_stage_cfg`
- Branch: `experiment/stage-calibrated-bounded-cfg`
- Base commit: `a4fda8a573ee0e372d3b53064e235ae87f0e670c`
- Main repository was not modified.
- Existing worktrees, checkpoints, historical outputs, and five stashes were preserved.

## Preserved main-repository stashes

```text
stash@{0}: On raab-sc-p0: preserve-raab-sc-p0
stash@{1}: On mattergen-rl-p0: preserve-mattergen-rl-p0
stash@{2}: On alm-gen-p0: preserve-alm-gen-p0
stash@{3}: On mattergen-lora-p0: preserve-mattergen-lora-p0
stash@{4}: On composable-property-adapter-p0: preserve-gcpa-p0
```

## Compute audit immediately before freezing

```text
GPU 0: 62450 MiB, 100% — occupied by sgl_diffusion::scheduler_U0
GPU 1: 62509 MiB, 100% — occupied by sgl_diffusion::scheduler_U1
GPU 2: 83319 MiB — occupied by another user's Python task
GPU 3: 62713 MiB, 100% — occupied by sgl_diffusion::scheduler_U2
GPU 4: 3 MiB, 0% — idle and eligible
GPU 5: 62509 MiB, 100% — occupied by sgl_diffusion::scheduler_U3
GPU 6: 39439 MiB — occupied by VLLM::EngineCore
GPU 7: 86921 MiB — occupied by VLLM::EngineCore
```

Only GPU 4 is eligible at this audit point. The pipeline will not use or preempt GPUs 0/1/2/3/5/6/7 while they are occupied.

Existing tmux session:

```text
phizero (attached; unrelated and preserved)
```

The new run must use a separate tmux session and retained per-stage logs.

## Pre-freeze implementation smoke test

A non-study historical seed (`42`) was used only for an implementation test. The sampler completed all 1000 reverse steps for C0, constant, permanent, and bounded policies (7600 score calls). The C0 structure returned by the sampler and the explicitly recorded C0 structure were exactly equal in atomic numbers, fractional positions, and cell. No smoke-test outcome was written into calibration, P0, or Formal256.
