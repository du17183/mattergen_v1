# 2026-09-29 thesis integration upload scope

The working integration branch contains the latest Chapter 1/3/4/5/6 and abstract revisions, the 137-page USTB-template PDF, final vector figure assets and source scripts, the 32-paired-seed RC-NFGD Timing summary/paired bootstrap tables, Pareto tables, Best-of-2 and Budget-Scaling compact evidence, frozen trajectory-search protocols and negative-result audits.

The companion `reproducibility/env/` snapshot records the *observed* ALM and plotting package versions plus TeX/GPU details. `docs/REPRODUCE_ON_NEW_MACHINE.md` separates a clean-checkout PDF build from the checkpoint-backed GPU replay. The source-to-PDF command is `./thesis/latex/build_ustb_final_integrated.sh`.

The exploratory artwork revisions under `thesis_release/figure_pilots/` and the older `ustb_mattergen_final_revised.pdf` are intermediate artifacts, not final thesis claims. They may be retained on the writing branch, but the thesis-facing release should point readers to `ustb_mattergen_final_integrated.pdf` and its current LaTeX source.

No model weight, virtual environment, TinyTeX distribution, full raw generation tree or credential is included. Existing large raw data remain in place on the source machine and are indexed by `archive/thesis-exploration-2026:research_archive/external_data_manifest.csv`; this normal Git upload does not erase or replace them. Any separate raw-data publication must respect the existing `thesis_release/ARCHIVE_AND_UPLOAD.md` licensing, hashes and size policy.

Timing chronology matters: `docs/rcnfgd_alm_replay_740000_report.md` correctly says `NOT_RUN` as of the gate-only run. A later independent authorized run completed all 32 paired seeds; its final report is `results/rcnfgd/timing_ablation/timing_ablation_report.md`. The earlier failed Python 3.10.20/Torch 2.4.1 replay and interim `NOT_RUN` placeholders are preserved rather than rewritten.
