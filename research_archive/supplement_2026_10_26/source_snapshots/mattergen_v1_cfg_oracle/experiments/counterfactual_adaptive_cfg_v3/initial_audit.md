# Counterfactual Adaptive CFG V3 — initial audit

- Source worktree: `/mnt/datasets-livsyn/dxl/mattergen_v1`
- Isolated worktree: `/mnt/datasets-livsyn/dxl/mattergen_v1_cfg_oracle`
- Branch: `experiment/counterfactual-adaptive-cfg-v3`
- Base commit: `1af301c5e534ff6b9b9f96ea5dab48036d9c46af`
- Existing source-worktree untracked thesis/experiment outputs were preserved.
- Existing stash count: 5; no stash was applied, dropped, or modified.
- GPU 2 was occupied at audit time and is excluded.
- Oracle GPU pool at audit time: 0, 1, 3, 4, 5, 6, 7.
- Seed audit: 12 worktrees and 41,734 files; zero semantic seed overlap.
- Two digit sequences inside explicitly named SHA256 values were retained as audited false positives.
- Oracle seeds: 816000–816031, calibration-only.
- Contingent fresh P0 seeds: 817000–817031.
- Contingent fresh Formal256 seeds: 818000–818255.
- Formal256 directory is not created before V3 P0 GO.

The frozen scientific order is Oracle generation → surrogate outcome evaluation →
Headroom Gate → Predictability Gate → fresh P0 → fresh Formal256. Any scientific
gate failure stops the downstream stages without rescue tuning.

Frozen-preflight test note: all 13 existing guidance-schedule tests pass. Four
pre-existing cases in `test_sampling.py` call the current Langevin corrector
without its required `dt` keyword and fail before touching Oracle code; the
corrector and those historical tests are intentionally not changed here.
