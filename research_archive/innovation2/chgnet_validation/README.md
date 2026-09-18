# Independent CHGNet Validation

## Research Question

Does the force benefit generalize beyond the guiding MatterSim evaluator?

## Motivation

An independent CHGNet force evaluation will preserve the effect direction.

## Change from Previous Stage

Re-evaluated all C0/F0 structures with frozen CHGNet 0.3.0.

## Implementation

Archive category: `FULL_ARCHIVE`. Reproducibility level: `FULL`.

Original server path `/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_independent_mlip_validation`. Portable reproduction must use repository-relative paths or an explicit user-supplied data root; the original server path is provenance only.

## Dataset / Seeds

512 structures and two recomputed force endpoints. Seed manifests are copied when small; otherwise their immutable source path and external-data hash are recorded.

## Main Results

Cross-MLIP generalization was supported, within the stated independence limits.

## Decision

Archive status: **SUPPORTED**.

Historical conclusion at the time: See the immutable source report for the exact historical GO/FAIL wording. This archive does not overwrite or retroactively relabel that source.

Final thesis interpretation after later confirmation: Cross-MLIP generalization was supported, within the stated independence limits.

## Why This Route Was Stopped

The planned independent surrogate check was complete.

## What It Motivated Next

Report as surrogate cross-validation, not DFT.

## Provenance

- Source worktree: `/mnt/datasets-livsyn/dxl/mattergen_v1`
- Source branch: `experiment/raab-sc-p0`
- Source commit: `1af301c5e534ff6b9b9f96ea5dab48036d9c46af`
- Original path: `experiments/final_independent_mlip_validation`
- Archive date: `2026-09-18`
- Covered by final release: `yes/core`
- Used in thesis: `core`
- Notes: `none`
