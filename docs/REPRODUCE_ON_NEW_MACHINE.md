# Cross-machine reproduction guide

This repository archives the thesis source, figure sources and compact evidence; it does not contain model weights or the large raw generation directories. A clean checkout can rebuild the thesis PDF. Re-running MatterGen/RC-NFGD generation is a separate, GPU-backed reproduction and requires the excluded checkpoints and raw-data manifests.

## Repository and environment

- Source branch: `writing/thesis-draft-2026`; thesis-facing branch: `release/thesis-final-2026`. Use the exact commit recorded by `git rev-parse HEAD` after cloning, not the branch name alone.
- Original host: h20-1, Linux 5.15.0-91-generic x86_64, NVIDIA H20, driver 535.161.08 (8 GPUs available on host; no claim that 8 GPUs are required to build the PDF).
- Validated RC-NFGD replay environment: Python 3.10.12, PyTorch 2.9.0+cu128, CUDA runtime 12.8, located locally at `/mnt/datasets-livsyn/dxl/alm/.venv`. This path is **not portable**. Exact package versions are in `reproducibility/env/alm_pip_versions_2026-09-29.txt`.
- Figure environment: Python 3.10.12, Matplotlib 3.8.4, PyMuPDF 1.27.2.3, NumPy 2.2.6; package list in `reproducibility/env/figure_pip_versions_2026-09-29.txt`.
- LaTeX: XeTeX 3.141592653-2.6-0.999998, TeX Live 2026 (kpathsea 6.4.2). Local `.TinyTeX/` is deliberately excluded. The USTB class currently resolves Fandol Song via `../../../.TinyTeX/texmf-dist/fonts/opentype/public/fandol/FandolSong-Regular.otf` relative to the thesis directory. Reproduce that local layout with a licensed TeX Live 2026/TinyTeX installation before building; setting `TEXROOT` alone will not fix a missing font file.

These lists record the **observed environment**, not a proven portable lockfile. In particular, packages with compiled CUDA extensions must be installed for the target GPU, driver and PyTorch ABI. Do not silently substitute PyTorch 2.4.1+cu121: that rebuilt environment failed the historical seed-740000 replay despite matching the nominal older version. The failure report is retained in `docs/rcnfgd_environment_rebuild_20260929.md`.

## Rebuild the thesis PDF (no GPU or weights)

1. Clone this repository and check out the intended commit. Install XeLaTeX/TeX Live 2026 and the required fonts/packages; restore the ignored `.TinyTeX` layout described above.
2. Run `./thesis/latex/build_ustb_final_integrated.sh` from the repository root. The script makes three XeLaTeX passes, checks unresolved references/glyphs, and writes `thesis/latex/ustb_2026_polished_preview/ustb_mattergen_final_integrated.pdf`.
3. Compare page/figure/table counts and PDF SHA256 to the environment-specific reference in `reproducibility/env/README.md`. Font installations, timestamps and TeX distribution revisions can change PDF bytes without changing the scientific content; inspect the rendered pages and log as well as the hash.

The submitted manuscript still has cover placeholders. It is a pre-advisor-review build, not a completed personal-information submission.

## Rebuild figures and compact result interpretation

- `scripts/draw_thesis_integration_figures.py` redraws two mechanism diagrams from frozen drawing definitions and the Timing contrast plot from the committed `results/rcnfgd/timing_ablation/timing_paired_bootstrap.csv`. Run it in the figure environment from the repository root; preserve the stored figures as the reference artwork. The search-framework plots have separate provenance scripts and compact tables in `results/trajectory_search/`.
- Figure and table labels should be checked against the committed reports, particularly `results/rcnfgd/timing_ablation/timing_ablation_report.md` and the Best-of-2/Budget-Scaling CSVs. Those reports are the source of the rounded values in the thesis.
- Some older figure scripts and trajectory-search finalization code refer to `/root/.codex/skills/nature-figure` or `/mnt/datasets-livsyn/dxl/mattergen_v1` and to larger raw experiment directories. They are archival provenance, **not** a turnkey clean-checkout regeneration path. Update only local path configuration in a separate reproducibility run, while keeping frozen metrics/protocols unchanged.

## Re-run generation only after a gate

The weight/checkpoint files and raw sample trees are not committed. Restore them from their documented sources and verify their SHA256 against the hashes in `docs/rcnfgd_environment_rebuild_20260929.md` and the experiment-specific protocols. The raw-data inventory is on `archive/thesis-exploration-2026` at `research_archive/external_data_manifest.csv`; it indexes but does not embed those large files. Obtain the original pretrained MatterGen and MatterSim assets under their applicable licenses. Do not infer that having a matching package version is sufficient for deterministic replay.

Before any fresh Timing Ablation, replay historical seed `740000` for both C0 and F0 using the frozen sampler and checkpoint. Compare extxyz structure SHA256 (or the predeclared geometry tolerance), property MAE, MatterSim MaxF and mean force against `docs/rcnfgd_alm_replay_740000_report.md`; require the original `1e-6` metric tolerance and trace-count check. Stop if the gate fails. Only the ALM environment passed this gate in the archived investigation. The later 32-paired-seed Timing Ablation is already complete and its compact per-seed/statistical tables are committed; no rerun is needed to build the thesis.

Best-of-2 and Budget Scaling have their own frozen cohorts and evidence under `experiments/trajectory_search/` and `results/trajectory_search/`. Do not reuse thesis confirmation seeds or modify selection rules, verifier thresholds or evaluation metrics when reproducing them.

## Scope and limitations

The repository includes scientific source, protocols, negative outcomes, compact results and manuscript assets. Excluded weights, full raw CIF/EXTXYZ outputs and heavy logs require separate licensed archival transfer. Exact bitwise replication on a different driver/GPU/TeX installation is not guaranteed by a version list; validate the historical replay and report any deviation. No DFT validation or Fixed-K2/RC-NFGD joint gain is claimed.
