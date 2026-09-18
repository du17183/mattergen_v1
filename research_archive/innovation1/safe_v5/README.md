# Safe-Selection CFG V5

## Research Question

Can conservative candidate selection deliver safe adaptive gains?

## Motivation

A calibrated selection rule can avoid harmful candidates without discarding all gains.

## Change from Previous Stage

Replaced continuous adaptation with frozen safe candidate selection.

## Implementation

Archive category: `FULL_ARCHIVE`. Reproducibility level: `PARTIAL`.

Original server path `/mnt/datasets-livsyn/dxl/mattergen_v1_safe_cfg/experiments/calibrated_safe_selection_cfg`. Portable reproduction must use repository-relative paths or an explicit user-supplied data root; the original server path is provenance only.

## Dataset / Seeds

Validation gate, calibration reports and stop decision. Seed manifests are copied when small; otherwise their immutable source path and external-data hash are recorded.

## Main Results

The validation gate failed; downstream P0/Formal stages were not run.

## Decision

Archive status: **FAIL**.

Historical conclusion at the time: See the immutable source report for the exact historical GO/FAIL wording. This archive does not overwrite or retroactively relabel that source.

Final thesis interpretation after later confirmation: The validation gate failed; downstream P0/Formal stages were not run.

## Why This Route Was Stopped

The frozen validation rule rejected the policy.

## What It Motivated Next

Investigate whether effects are stage- or field-specific.

## Provenance

- Source worktree: `/mnt/datasets-livsyn/dxl/mattergen_v1_safe_cfg`
- Source branch: `experiment/calibrated-safe-selection-cfg`
- Source commit: `7c9d9261d5064a37b8c4cd03a0ecf022621145a7`
- Original path: `experiments/calibrated_safe_selection_cfg`
- Archive date: `2026-09-18`
- Covered by final release: `yes/key-negative`
- Used in thesis: `supporting`
- Notes: `none`
