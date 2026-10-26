# Non-weight research archive supplement — 2026-10-26

This additive snapshot closes the gap between the earlier selected-evidence archive and the source worktrees. It preserves uncommitted code/configuration, source modifications, per-seed outputs, generated structures, logs, diagnostics and generated training examples. Historical conclusions, source files, seeds, checkpoints and thesis statistics are not changed. No experiment is run by the archival helpers.

The snapshot is an **overlay on the recorded Git commit of each worktree**, not a duplicate of every already-published tracked file. `worktrees.json` records the branch, base commit and dirty status; `files.jsonl` records each overlay file's original path, size, mode and SHA256. Browseable source and compact reports are in `source_snapshots/`. Every included overlay file, including raw outputs, is also stored as a SHA256-addressed blob in the Release bundles listed by `assets.json`.

Release: https://github.com/du17183/mattergen_v1/releases/tag/mattergen-nonweight-20261026

## Included versus excluded

- Included `.pt` files are specifically named diffusion/RNG prefix states, diagnostic states or generated teacher/DAgger examples; they are data, not pretrained network weights.
- Trained neural weights/checkpoints, environments, package/bytecode caches, temporary reference databases and private configuration are excluded. `excluded.jsonl` records exclusions and available weight fingerprints. Existing small frozen scikit-learn allocator artifacts follow the original thesis-release policy.
- Symlinked files within the project are materialized as ordinary files for portability, with the original target recorded. Safe same-worktree directory aliases are preserved as relative links. External symlinks are not followed.
- The original 2026-09-18 manifests remain historical records. This supplement supplies a separate current inventory and does not rewrite old scientific reports or reinterpret negative results.
- Original project code and generated research outputs retain their applicable repository licenses. This supplement does not redistribute pretrained checkpoints or external reference databases. Runtime credentials must pass the path-only preflight scan before any upload.

## Download and verify

The verification/restore helper needs Python 3.10 or newer and the `zstd`
command. It uses only Python's standard library; model environments are not
needed to inspect or restore files.

Download the Release `.tar.zst` parts and `SHA256SUMS.txt` to an assets directory. With GitHub CLI, use `gh release download mattergen-nonweight-20261026 --repo du17183/mattergen_v1 --dir /absolute/path/to/assets`.

Individual public download URLs are also recorded in `assets.json`; GitHub
CLI is optional when files are downloaded through those URLs or the browser.

Run:

```bash
python3 research_archive/supplement_2026_10_26/restore_snapshot.py \
  --assets-dir /absolute/path/to/assets --verify-only
```

This checks compressed-asset hashes and every internal blob against its SHA256 filename. It does not run model inference.

## Restore one original worktree

1. Use a separate clone/checkout of the repository and check out the desired worktree's exact `commit` from `worktrees.json`. Source branches that previously existed only locally are published alongside this supplement.
2. Keep a copy of this supplement directory available and run its `restore_snapshot.py` with `--worktree` and `--destination`. For example, choose `mattergen_v1_matersim_guidance` to restore RC-NFGD and the Timing/seed-740000 execution evidence.
3. The helper refuses a destination with the wrong base commit or conflicting files; an explicit `--overwrite` option exists for deliberate restoration into a disposable checkout. It never deletes source files.
4. Restore licensed pretrained weights separately. For new GPU experiments, retain the original historical seed-740000 reproducibility gate and its unchanged tolerances. Package snapshots do not establish bitwise equivalence on another device.

The Timing scripts and gate checks are under `source_snapshots/mattergen_v1_matersim_guidance/experiments/rcnfgd_timing_ablation_20260929/`. Some historical scripts contain original absolute server paths. Restore dependencies and configure paths in a separate reproduction checkout before execution; archiving a script does not validate a new runtime.
