# Frozen C1 Confirmation

## Research Question

On wholly new paired seeds, do Fixed-K2 and Linear-K2 improve over C0?

## Motivation

Both frozen K2 policies will preserve gains under confirmatory evaluation.

## Change from Previous Stage

Ran a strictly frozen confirmatory cohort with new seeds.

## Implementation

Archive category: `FULL_ARCHIVE`. Reproducibility level: `PARTIAL`.

Original server path `/mnt/datasets-livsyn/dxl/mattergen_v1_linear_k2_confirm/experiments/frozen_linear_k2_confirmation`. Portable reproduction must use repository-relative paths or an explicit user-supplied data root; the original server path is provenance only.

## Dataset / Seeds

Generation, paired metrics, 20k bootstrap, quality guardrails and compute accounting. Seed manifests are copied when small; otherwise their immutable source path and external-data hash are recorded.

## Main Results

Fixed-K2 was supported; learned Linear-K2 was NOT_SUPPORTED.

## Decision

Archive status: **MIXED**.

Historical conclusion at the time: See the immutable source report for the exact historical GO/FAIL wording. This archive does not overwrite or retroactively relabel that source.

Final thesis interpretation after later confirmation: Fixed-K2 was supported; learned Linear-K2 was NOT_SUPPORTED.

## Why This Route Was Stopped

The learned policy failed its GO threshold; no rescue tuning was allowed.

## What It Motivated Next

Freeze Fixed-K2 as the thesis method.

## Provenance

- Source worktree: `/mnt/datasets-livsyn/dxl/mattergen_v1_linear_k2_confirm`
- Source branch: `experiment/frozen-linear-k2-confirmation`
- Source commit: `1af301c5e534ff6b9b9f96ea5dab48036d9c46af`
- Original path: `experiments/frozen_linear_k2_confirmation`
- Archive date: `2026-09-18`
- Covered by final release: `yes/core`
- Used in thesis: `core`
- Notes: `none`
