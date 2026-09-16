# Initial audit — Field-Decoupled Adaptive CFG

- Worktree: `/mnt/datasets-livsyn/dxl/mattergen_v1_field_cfg`
- Branch: `experiment/field-decoupled-adaptive-cfg`
- Base commit: `f281462d8f690493f3bc3b78faffaea6abdeb2ae`
- Main repository, historical worktrees, checkpoints, experiments, and five stashes were preserved.

## Preserved stashes

```text
stash@{0}: On raab-sc-p0: preserve-raab-sc-p0
stash@{1}: On mattergen-rl-p0: preserve-mattergen-rl-p0
stash@{2}: On alm-gen-p0: preserve-alm-gen-p0
stash@{3}: On mattergen-lora-p0: preserve-mattergen-lora-p0
stash@{4}: On composable-property-adapter-p0: preserve-gcpa-p0
```

## Implementation feasibility

Historical seed 42 was used only for a non-study implementation test. A standard shared CFG2 trajectory and field-wise `(g_atomic,g_pos,g_cell)=(2,2,2)` trajectory used identical initial state and random stream for 1000 reverse steps.

```text
atomic_numbers_identical = true
positions_max_abs_error = 0.0
cell_max_abs_error = 0.0
status = PASS
```

No implementation-test structure enters calibration, P0, or Formal256.

## Resource audit

At the pre-freeze snapshot, GPUs 1/2/4/6/7 were occupied by other users. GPUs 0/3/5 were idle. An attempted test on GPU 4 was terminated during Python import immediately after its newly occupied state was observed; no model was loaded there. The successful implementation check ran only on idle GPU 5.

Existing unrelated tmux session `phizero` is preserved. The formal pipeline must use a separate tmux session and re-audit selected GPUs immediately at launch.
