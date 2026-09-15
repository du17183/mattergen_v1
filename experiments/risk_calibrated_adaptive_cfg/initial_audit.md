# Initial audit

- Timestamp: `2026-09-15T12:04:42+08:00`
- Worktree: `/mnt/datasets-livsyn/dxl/mattergen_v1_risk_cfg`
- Branch: `experiment/risk-calibrated-adaptive-cfg`
- Base HEAD: `1af301c5e534ff6b9b9f96ea5dab48036d9c46af`
- Initial new-worktree status: clean immediately after checkout completed
- Main-repository stash count: `5`; all five retained
- Initial tmux status: no server running
- Storage: JuiceFS 1.0P total, 5.6T used, approximately 1019T available
- Checkpoint policy: use the historical official checkpoint by absolute read-only path; do not copy or modify it
- GPU 0: H20, 72 MiB / 97871 MiB, 0% utilization
- GPU 1: H20, 3 MiB / 97871 MiB, 0% utilization
- GPU 2: H20, 83319 MiB / 97871 MiB; **excluded because occupied**
- GPU 3: H20, 208 MiB / 97871 MiB, 0% utilization
- GPU 4: H20, 3 MiB / 97871 MiB, 0% utilization
- GPU 5: H20, 3 MiB / 97871 MiB, 0% utilization
- GPU 6: H20, 3 MiB / 97871 MiB, 0% utilization
- GPU 7: H20, 3 MiB / 97871 MiB, 0% utilization

The transient all-deleted status seen during `git worktree add` disappeared when
the initial checkout completed. No reset, checkout overwrite, deletion, or stash
operation was used.
