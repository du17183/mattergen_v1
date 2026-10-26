# Initial isolation and resource audit

- Main project: `/mnt/datasets-livsyn/dxl/mattergen_v1`
- Isolated worktree: `/mnt/datasets-livsyn/dxl/mattergen_v1_robust_adaptive_cfg`
- Branch: `experiment/robust-adaptive-cfg-v2`
- Base HEAD: `1af301c5e534ff6b9b9f96ea5dab48036d9c46af`
- Worktree status immediately after checkout: clean
- Existing tmux at initial sandbox check: inaccessible inside sandbox; escalated read-only check returned `no server running`
- Authorized experiment GPUs: 0, 1, 3, 4, 5, 6, 7
- Excluded occupied GPU: 2

The main worktree's untracked experiment/thesis results were preserved and were not copied, moved, staged, or edited. All five pre-existing stashes remained untouched:

```text
stash@{0}: On raab-sc-p0: preserve-raab-sc-p0
stash@{1}: On mattergen-rl-p0: preserve-mattergen-rl-p0
stash@{2}: On alm-gen-p0: preserve-alm-gen-p0
stash@{3}: On mattergen-lora-p0: preserve-mattergen-lora-p0
stash@{4}: On composable-property-adapter-p0: preserve-gcpa-p0
```

No process belonging to another user is terminated or preempted. Cohort launch checks select only explicitly authorized GPUs with less than 5000 MiB memory in use; otherwise the pipeline records `WAITING_RESOURCE`.
