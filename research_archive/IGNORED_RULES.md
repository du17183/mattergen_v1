# Ignored Artifact Rules

These rules apply across the scanned `mattergen_v1*` trees. They are recorded once rather than repeated for every cache file.

- Python bytecode and `__pycache__/`.
- Conda, virtual environments and package caches.
- `.matplotlib/`, W&B caches, editor swap/backup files and temporary shell scratch.
- Duplicate outputs already represented by an immutable tracked source or the frozen final release.
- Bulk raw logs unless a compact log summary is the only surviving decision evidence.
- Downloaded third-party weights and official checkpoints; required instances are externally hashed.
- Raw generated structures, relaxation trajectories, per-step traces, LMDBs and large tensors; these are externally manifested rather than silently ignored.

`X-05` in the experiment inventory records the duplicate `final_thesis_results` build package as an aggregate `IGNORE` decision. This does not authorize deleting it from the server.
