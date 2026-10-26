# Frozen Linear-K2 confirmation preflight

Audit date: 2026-09-17 (Asia/Shanghai)

## Git

```text
source worktree: /mnt/datasets-livsyn/dxl/mattergen_v1
source HEAD: 1af301c5e534ff6b9b9f96ea5dab48036d9c46af
source branch: experiment/raab-sc-p0
confirmation worktree: /mnt/datasets-livsyn/dxl/mattergen_v1_linear_k2_confirm
confirmation branch: experiment/frozen-linear-k2-confirmation
confirmation base HEAD: 1af301c5e534ff6b9b9f96ea5dab48036d9c46af
```

The source worktree contains untracked thesis experiment outputs.  They were
not modified or moved.  Existing stashes were left untouched.

## Runtime resources at preflight

```text
GPU 0: busy, sgl_diffusion, approximately 64 GB, 100% utilization
GPU 1: busy, sgl_diffusion, approximately 64 GB, 100% utilization
GPU 2: idle
GPU 3: busy, sgl_diffusion, approximately 64 GB, 100% utilization
GPU 4: idle
GPU 5: busy, sgl_diffusion, approximately 64 GB, 100% utilization
GPU 6: busy, VLLM, approximately 39 GB
GPU 7: busy, VLLM, approximately 87 GB
tmux: only pre-existing `phizero`
```

Only GPUs that remain completely idle at the actual launch audit may be used.
No other process may be terminated or preempted.
