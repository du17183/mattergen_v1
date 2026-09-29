# Observed software environment and PDF reference

Snapshot taken on 2026-09-29 on `h20-1` (Linux 5.15.0-91-generic x86_64; NVIDIA H20, driver 535.161.08).

| Purpose | Interpreter/runtime | Evidence |
|---|---|---|
| RC-NFGD historical replay and Timing environment | Python 3.10.12; PyTorch 2.9.0+cu128; CUDA runtime 12.8 | `alm_pip_versions_2026-09-29.txt` (291 installed packages) |
| Figure rendering | Python 3.10.12; Matplotlib 3.8.4; PyMuPDF 1.27.2.3; NumPy 2.2.6 | `figure_pip_versions_2026-09-29.txt` (14 installed packages) |
| Thesis typesetting | XeTeX 3.141592653-2.6-0.999998; TeX Live 2026; kpathsea 6.4.2 | project-local ignored `.TinyTeX/` |

These are `pip list --format=freeze` snapshots, **not** a solver lockfile or proof that installing these versions elsewhere reproduces CUDA trajectories. Binary PyG/CUDA packages require compatible builds. No API tokens, private indexes or local wheel URLs are included.

The 2026-09-29 integrated PDF built here has 137 pages, 30 figures and 29 tables. The latest build's SHA256 is `7372e6eb386c3401038f7dc536c21f5e90380092a27e135a09fec6e77b9e882e`. PDF binary identity may differ on another TeX distribution or build time even if the scientific content and rendered pages match. The reproducible build entry point is `thesis/latex/build_ustb_final_integrated.sh`.

Checkpoint fingerprints, gate details and the failed alternative environment are in `docs/rcnfgd_alm_replay_740000_report.md` and `docs/rcnfgd_environment_rebuild_20260929.md`. The ALM historical C0/F0 seed-740000 gate passed; the Python 3.10.20 / PyTorch 2.4.1+cu121 rebuilt environment failed. The later 32-seed timing result is reported separately under `results/rcnfgd/timing_ablation/`.

For a new machine, start with `docs/REPRODUCE_ON_NEW_MACHINE.md`. The package lists support environment reconstruction; full generation still needs excluded licensed checkpoints and a new PASS on the historical replay gate.
