# Budget-Aware Convergence-Guided Corrector Scheduling

This package contains the implementation support, frozen configs, resume-safe runners, refined 8/32-seed evidence, final No-Go report, figures, and reproducibility manifests for the second-generation Corrector Gating study.

## Branch lineage

`main` (innovation 1) → `feature/convergence-aware-corrector-gating` (original innovation 2 / G3) → `feature/budget-aware-corrector-gating` (this budget-aware study).

The branch is intentionally based on `feature/convergence-aware-corrector-gating`; it must not be merged into `main` without an explicit user decision.

## Outcome

Both G1 and G2 passed the 8-seed smoke test. At the frozen 32-seed screen, neither passed every predeclared quality-speed gate, so the run stopped without retuning and without starting seeds 14032–14063 or formal seeds 30000–30255. See `reports/final/budget_aware_final_report.md`.

## Data policy

This Git branch contains refined, reviewable artifacts. Complete raw non-weight artifacts are archived in [GitHub Release budget-aware-gating-data-20260724](https://github.com/du17183/mattergen_v1/releases/tag/budget-aware-gating-data-20260724) with SHA256 manifests. Checkpoints, MatterSim weights, Conda environments, caches, datasets, and large per-task logs are excluded from Git history.
