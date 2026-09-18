# Clean-x0 Force Evaluation

## Research Question

Should force be evaluated on noisy x_t or the denoised crystal estimate?

## Motivation

Clean-x0 evaluation reduces surrogate uncertainty.

## Change from Previous Stage

Evaluated the denoised structure for force feedback.

## Implementation

Archive category: `SUMMARY_ONLY`. Reproducibility level: `PARTIAL`.

Original server path `/mnt/datasets-livsyn/dxl/mattergen_v1_matersim_guidance/experiments/innovation2_finalization`. Portable reproduction must use repository-relative paths or an explicit user-supplied data root; the original server path is provenance only.

## Dataset / Seeds

Reliability diagnostics and frozen method definition. Seed manifests are copied when small; otherwise their immutable source path and external-data hash are recorded.

## Main Results

Clean-x0 was retained as a supported implementation choice, with explicit causal limitations.

## Decision

Archive status: **SUPPORTED**.

Historical conclusion at the time: See the immutable source report for the exact historical GO/FAIL wording. This archive does not overwrite or retroactively relabel that source.

Final thesis interpretation after later confirmation: Clean-x0 was retained as a supported implementation choice, with explicit causal limitations.

## Why This Route Was Stopped

No separate noisy-x_t confirmatory ablation was available.

## What It Motivated Next

Carry the choice into the frozen P0 protocol.

## Provenance

- Source worktree: `/mnt/datasets-livsyn/dxl/mattergen_v1_matersim_guidance`
- Source branch: `experiment/mattersim-late-force-guidance-p0`
- Source commit: `1af301c5e534ff6b9b9f96ea5dab48036d9c46af`
- Original path: `experiments/innovation2_finalization`
- Archive date: `2026-09-18`
- Covered by final release: `yes/summary`
- Used in thesis: `supporting`
- Notes: `none`
