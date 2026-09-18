# Stage Reliability Study

## Research Question

At which denoising stages is surrogate force feedback reliable enough to use?

## Motivation

Late clean estimates are more reliable than strongly noisy states.

## Change from Previous Stage

Added stage-wise reliability diagnostics.

## Implementation

Archive category: `SUMMARY_ONLY`. Reproducibility level: `PARTIAL`.

Original server path `/mnt/datasets-livsyn/dxl/mattergen_v1_matersim_guidance/experiments/innovation2_finalization`. Portable reproduction must use repository-relative paths or an explicit user-supplied data root; the original server path is provenance only.

## Dataset / Seeds

Frozen diagnostic summaries used by the finalization package. Seed manifests are copied when small; otherwise their immutable source path and external-data hash are recorded.

## Main Results

Evidence supported restricting guidance to reliable late stages.

## Decision

Archive status: **SUPPORTED**.

Historical conclusion at the time: See the immutable source report for the exact historical GO/FAIL wording. This archive does not overwrite or retroactively relabel that source.

Final thesis interpretation after later confirmation: Evidence supported restricting guidance to reliable late stages.

## Why This Route Was Stopped

Diagnostic purpose was complete.

## What It Motivated Next

Use clean-x0 estimates and late bounded intervention.

## Provenance

- Source worktree: `/mnt/datasets-livsyn/dxl/mattergen_v1_matersim_guidance`
- Source branch: `experiment/mattersim-late-force-guidance-p0`
- Source commit: `1af301c5e534ff6b9b9f96ea5dab48036d9c46af`
- Original path: `experiments/innovation2_finalization`
- Archive date: `2026-09-18`
- Covered by final release: `yes/summary`
- Used in thesis: `supporting`
- Notes: `none`
