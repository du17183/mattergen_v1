# Equal-Budget Post-Generation Comparison

## Research Question

Does during-sampling guidance retain value against equal-budget post-processing?

## Motivation

Online RC-NFGD differs from a compute-matched endpoint corrector.

## Change from Previous Stage

Added a post-generation equal-budget comparator.

## Implementation

Archive category: `FULL_ARCHIVE`. Reproducibility level: `PARTIAL`.

Original server path `/mnt/datasets-livsyn/dxl/mattergen_v1_matersim_guidance/experiments/mattersim_equal_budget_post_generation`. Portable reproduction must use repository-relative paths or an explicit user-supplied data root; the original server path is provenance only.

## Dataset / Seeds

Matched-budget results and final comparative audit. Seed manifests are copied when small; otherwise their immutable source path and external-data hash are recorded.

## Main Results

The registered comparison supported retaining the RC-NFGD interpretation.

## Decision

Archive status: **SUPPORTED**.

Historical conclusion at the time: See the immutable source report for the exact historical GO/FAIL wording. This archive does not overwrite or retroactively relabel that source.

Final thesis interpretation after later confirmation: The registered comparison supported retaining the RC-NFGD interpretation.

## Why This Route Was Stopped

Comparator question was resolved.

## What It Motivated Next

Use the frozen online method in the thesis.

## Provenance

- Source worktree: `/mnt/datasets-livsyn/dxl/mattergen_v1_matersim_guidance`
- Source branch: `experiment/mattersim-late-force-guidance-p0`
- Source commit: `1af301c5e534ff6b9b9f96ea5dab48036d9c46af`
- Original path: `experiments/mattersim_equal_budget_post_generation`
- Archive date: `2026-09-18`
- Covered by final release: `yes/core`
- Used in thesis: `core`
- Notes: `none`
