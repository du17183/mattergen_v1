# Adaptive CFG V1

## Research Question

Can online multi-field residual adaptation improve conditional generation?

## Motivation

Residual-driven CFG updates will improve property error without harming structure quality.

## Change from Previous Stage

Introduced the original adaptive multi-field CFG controller and EMA/clipping choices.

## Implementation

Archive category: `FULL_ARCHIVE`. Reproducibility level: `PARTIAL`.

Original server path `/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_innovation1_audit`. Portable reproduction must use repository-relative paths or an explicit user-supplied data root; the original server path is provenance only.

## Dataset / Seeds

Original Formal256 evidence plus a later eight-seed minimal ablation audit. Seed manifests are copied when small; otherwise their immutable source path and external-data hash are recorded.

## Main Results

Historical positive evidence was not stable across the later batch; benefits and quality trade-offs coexist.

## Decision

Archive status: **MIXED**.

Historical conclusion at the time: See the immutable source report for the exact historical GO/FAIL wording. This archive does not overwrite or retroactively relabel that source.

Final thesis interpretation after later confirmation: Historical positive evidence was not stable across the later batch; benefits and quality trade-offs coexist.

## Why This Route Was Stopped

Cross-batch inconsistency prevented a general stability claim.

## What It Motivated Next

Audit harmful adaptation and preserve a reference branch.

## Provenance

- Source worktree: `/mnt/datasets-livsyn/dxl/mattergen_v1`
- Source branch: `archive/thesis-analysis-package-v1`
- Source commit: `a367f85efe8027bf8ef2db41cc48ec2fd3d60582`
- Original path: `experiments/final_innovation1_audit`
- Archive date: `2026-09-18`
- Covered by final release: `yes/summary`
- Used in thesis: `supporting`
- Notes: `Historical conclusion and later thesis interpretation are deliberately separated.`
