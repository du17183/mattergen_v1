This release supplements the thesis-facing and historical exploration branches with the non-weight research files that were previously only on h20-1.

It includes full overlay snapshots for the recorded research worktrees: RC-NFGD Timing execution and seed-740000 replay/gate code, historical exploratory implementations and configurations, source modifications, per-seed statistics, generated structures, traces, logs, diffusion/RNG snapshots, and generated teacher/DAgger data. It does not change the frozen experimental results or scientific conclusions.

The archive branch contains browseable code and reports at `research_archive/supplement_2026_10_26/`. `files.jsonl` and `worktrees.json` record exact paths, commits and SHA256 values. Data bundles deduplicate identical file contents. The restore kit provides the manifests and `restore_snapshot.py`; verify the compressed assets and their internal blobs before applying an overlay to a clean checkout of the recorded commit.

Trained network weights, environments, runtime/package caches, external reference databases and private configuration are excluded and documented. Named non-weight `.pt` diffusion states and generated datasets are retained. All original server files remain in place.

Use `SHA256SUMS.txt` for download validation. Extract the restore kit, then run `python3 restore-kit/restore_snapshot.py --assets-dir /absolute/path/to/downloads --verify-only`. Detailed restore steps are in the kit's README. Restoring files does not establish a new GPU runtime's reproducibility; the historical seed-740000 gate still applies before fresh RC-NFGD experiments.
