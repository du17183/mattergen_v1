# Robust Adaptive CFG V2

## Research Question

Can calibrated residual normalization reduce V1 harmful adaptation?

## Motivation

Timestep normalization, robust aggregation, slew limiting and fallback will make V1 reproducibly safer.

## Change from Previous Stage

Added deterministic replay calibration and bounded robust adaptive control.

## Implementation

Archive category: `FULL_ARCHIVE`. Reproducibility level: `PARTIAL`.

Original server path `/mnt/datasets-livsyn/dxl/mattergen_v1_robust_adaptive_cfg/experiments/robust_adaptive_cfg_v2`. Portable reproduction must use repository-relative paths or an explicit user-supplied data root; the original server path is provenance only.

## Dataset / Seeds

Calibration, fresh P0, paired statistics and harm/tail analysis. Seed manifests are copied when small; otherwise their immutable source path and external-data hash are recorded.

## Main Results

Fresh P0 failed the preregistered gate; Formal256 was correctly not run.

## Decision

Archive status: **FAIL**.

Historical conclusion at the time: See the immutable source report for the exact historical GO/FAIL wording. This archive does not overwrite or retroactively relabel that source.

Final thesis interpretation after later confirmation: Fresh P0 failed the preregistered gate; Formal256 was correctly not run.

## Why This Route Was Stopped

The frozen P0 rule failed and rescue tuning was forbidden.

## What It Motivated Next

Test whether oracle headroom exists before learning another controller.

## Provenance

- Source worktree: `/mnt/datasets-livsyn/dxl/mattergen_v1_robust_adaptive_cfg`
- Source branch: `experiment/robust-adaptive-cfg-v2`
- Source commit: `1af301c5e534ff6b9b9f96ea5dab48036d9c46af`
- Original path: `experiments/robust_adaptive_cfg_v2`
- Archive date: `2026-09-18`
- Covered by final release: `yes/key-negative`
- Used in thesis: `supporting`
- Notes: `none`
