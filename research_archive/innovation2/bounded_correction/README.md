# Bounded-Correction Ablation

## Research Question

Is norm bounding the necessary mechanism behind the force-guidance benefit?

## Motivation

Bounded correction outperforms an otherwise matched unbounded update.

## Change from Previous Stage

Removed norm-dependent saturation and hard caps while retaining other safeguards.

## Implementation

Archive category: `FULL_ARCHIVE`. Reproducibility level: `FULL`.

Original server path `/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_bounded_correction_audit`. Portable reproduction must use repository-relative paths or an explicit user-supplied data root; the original server path is provenance only.

## Dataset / Seeds

Registered three-arm audit. Seed manifests are copied when small; otherwise their immutable source path and external-data hash are recorded.

## Main Results

Bounding was not supported as the core mechanism.

## Decision

Archive status: **NOT_SUPPORTED**.

Historical conclusion at the time: See the immutable source report for the exact historical GO/FAIL wording. This archive does not overwrite or retroactively relabel that source.

Final thesis interpretation after later confirmation: Bounding was not supported as the core mechanism.

## Why This Route Was Stopped

The unbounded comparator was not worse under the registered test.

## What It Motivated Next

Narrow the thesis claim to reliable constrained guidance, not bounding necessity.

## Provenance

- Source worktree: `/mnt/datasets-livsyn/dxl/mattergen_v1`
- Source branch: `experiment/raab-sc-p0`
- Source commit: `1af301c5e534ff6b9b9f96ea5dab48036d9c46af`
- Original path: `experiments/final_bounded_correction_audit`
- Archive date: `2026-09-18`
- Covered by final release: `yes/key-negative`
- Used in thesis: `supporting`
- Notes: `none`
