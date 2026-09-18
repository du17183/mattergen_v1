# Corrector Residual Distillation V3 Anchor

## Research Question

Can anchoring repair residual reuse tails?

## Motivation

An anchored adapter prevents V2 geometry/force tails.

## Change from Previous Stage

Added teacher anchoring and frozen independent64.

## Implementation

Archive category: `SUMMARY_ONLY`. Reproducibility level: `SUMMARY_ONLY`.

Original server path `/mnt/datasets-livsyn/dxl/mattergen_v1_thesis_final/experiments/corrector_residual_distillation_v3_anchor`. Portable reproduction must use repository-relative paths or an explicit user-supplied data root; the original server path is provenance only.

## Dataset / Seeds

Independent64 and tail validation. Seed manifests are copied when small; otherwise their immutable source path and external-data hash are recorded.

## Main Results

The registered Stage-C criteria failed.

## Decision

Archive status: **FAIL**.

Historical conclusion at the time: See the immutable source report for the exact historical GO/FAIL wording. This archive does not overwrite or retroactively relabel that source.

Final thesis interpretation after later confirmation: The registered Stage-C criteria failed.

## Why This Route Was Stopped

Tail robustness was not confirmed.

## What It Motivated Next

Stop the family without retuning K/thresholds.

## Provenance

- Source worktree: `/mnt/datasets-livsyn/dxl/mattergen_v1_thesis_final`
- Source branch: `experiment/corrector-residual-distillation-v3-anchor`
- Source commit: `f983e1285de0913d5bb2c2cdef4b5488057f58ad`
- Original path: `experiments/corrector_residual_distillation_v3_anchor`
- Archive date: `2026-09-18`
- Covered by final release: `no`
- Used in thesis: `no`
- Notes: `none`
