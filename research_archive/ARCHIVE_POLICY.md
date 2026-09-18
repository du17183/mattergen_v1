# Exploration Archive Policy

## Purpose

This archive preserves scientifically meaningful process evidence without turning the thesis-facing release into an uncurated dump. It is additive: source worktrees and branches remain untouched, and no server data is deleted.

## Classification

### FULL_ARCHIVE

Used when a route changed research direction, produced a key positive or negative result, implemented a distinct method, supplied an important ablation, or is cited by the thesis/defense. The landing directory contains a README, source locator, and standard `code/`, `config/`, `results/`, `figures/`, and `seeds/` areas. Small high-value artifacts are copied with SHA-256 provenance.

### SUMMARY_ONLY

Used for small, overlapping, already tracked or non-confirmatory studies. The landing directory records the question, hypothesis, change, evidence, result, stop reason, next direction, source branch/commit/path and reproducibility level. Already tracked `experiments/` data is linked rather than duplicated.

### EXTERNAL_DATA_ONLY

Used where only raw structures, weights, logs or bytecode remnants survived and no defensible decision package could be reconstructed. The server artifact is hashed and held for manual review; it is not presented as scientific support.

### IGNORE

Used for duplicate generated packages and non-scientific artifacts. Global ignored patterns include caches, `__pycache__`, editor files, environment directories, temporary shell scratch, duplicate tracked outputs and downloaded third-party weights. The rules are recorded here; cache files are not individually enumerated.

## External-data policy

The following do not enter ordinary Git: raw generated structures, trajectories, relaxation caches, LMDBs, checkpoints, large tensors, bulk per-step traces, large raw logs, and third-party weights. [external_data_manifest.csv](external_data_manifest.csv) records size, file count, deterministic SHA-256/tree hash, original path, reason, recommended storage and reproduction need.

The source server files are retained. A later cleanup requires a separate authorization after remote archive review.

## Large-file policy

- Prefer archive files below 10 MB.
- Review files above 10 MB.
- Avoid files above 50 MB.
- Never add a new ordinary Git blob above 100 MB.

Upstream Git LFS objects inherited from the base branch are not duplicated.

## Historical-result policy

Historical source reports are immutable. Archive prose distinguishes:

1. the conclusion at the time of the experiment; and
2. the final thesis interpretation after later independent evidence.

Negative, mixed and stopped routes remain visible. The archive cannot be used to retroactively promote a historical `FAIL`/No-Go or conceal an adverse sample.

## Secrets and logs

Only selected small logs may be archived, and none are currently copied as raw bulk logs. The secret audit scans text-like archive files for common credential families and reports only pattern labels and file paths, never the suspected value. A detected credential blocks push.

## Final-claim lock

Archive edits must not change these conclusions:

- Fixed-K2 reference-preserved budgeted branching is the final Innovation 1 method.
- Learned Linear-K2 is `NOT_SUPPORTED`.
- RC-NFGD is `SUPPORTED` as Innovation 2.
- `DFT_VERIFIED=false`.

## Source preservation

No archive command may automatically remove a source worktree, branch, experiment directory, checkpoint or dataset. Inventory and hashing are read-only with respect to source worktrees.
