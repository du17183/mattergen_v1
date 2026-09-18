# Phase B Learned Allocator

## Research Question

Does a learned allocator beat fixed branching at matched compute?

## Motivation

Features can predict the best two branches better than a fixed choice.

## Change from Previous Stage

Trained and frozen the Phase B allocator at C=1.0.

## Implementation

Archive category: `FULL_ARCHIVE`. Reproducibility level: `FULL`.

Original server path `/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/reference_preserved_budgeted_cfg/feature_study`. Portable reproduction must use repository-relative paths or an explicit user-supplied data root; the original server path is provenance only.

## Dataset / Seeds

Held-out selections, validation/test results, reconstruction audit and 20k bootstrap. Seed manifests are copied when small; otherwise their immutable source path and external-data hash are recorded.

## Main Results

Adaptive allocation value was not supported.

## Decision

Archive status: **NOT_SUPPORTED**.

Historical conclusion at the time: See the immutable source report for the exact historical GO/FAIL wording. This archive does not overwrite or retroactively relabel that source.

Final thesis interpretation after later confirmation: Adaptive allocation value was not supported.

## Why This Route Was Stopped

The registered allocator-advantage gate failed.

## What It Motivated Next

Prefer the simpler Fixed-K2 policy.

## Provenance

- Source worktree: `/mnt/datasets-livsyn/dxl/mattergen_v1`
- Source branch: `experiment/raab-sc-p0`
- Source commit: `1af301c5e534ff6b9b9f96ea5dab48036d9c46af`
- Original path: `experiments/reference_preserved_budgeted_cfg/feature_study`
- Archive date: `2026-09-18`
- Covered by final release: `yes/core-negative`
- Used in thesis: `core-negative`
- Notes: `none`
