# Multifield Self-Conditioning P0

## Research Question

Can multi-field self-conditioning stabilize generation?

## Motivation

A learned self-conditioning state improves coherence across fields.

## Change from Previous Stage

Trained and evaluated a multi-field self-conditioning checkpoint.

## Implementation

Archive category: `SUMMARY_ONLY`. Reproducibility level: `SUMMARY_ONLY`.

Original server path `/mnt/datasets-livsyn/dxl/mattergen_v1_self_condition/experiments/multifield_self_conditioning_p0`. Portable reproduction must use repository-relative paths or an explicit user-supplied data root; the original server path is provenance only.

## Dataset / Seeds

P0 paired outcomes and quality guardrails. Seed manifests are copied when small; otherwise their immutable source path and external-data hash are recorded.

## Main Results

P0 failed with severe E-hull degradation.

## Decision

Archive status: **FAIL**.

Historical conclusion at the time: See the immutable source report for the exact historical GO/FAIL wording. This archive does not overwrite or retroactively relabel that source.

Final thesis interpretation after later confirmation: P0 failed with severe E-hull degradation.

## Why This Route Was Stopped

Quality guardrails failed.

## What It Motivated Next

Do not scale the checkpoint; preserve its hash externally.

## Provenance

- Source worktree: `/mnt/datasets-livsyn/dxl/mattergen_v1_self_condition`
- Source branch: `experiment/multifield-self-conditioning-p0`
- Source commit: `1af301c5e534ff6b9b9f96ea5dab48036d9c46af`
- Original path: `experiments/multifield_self_conditioning_p0`
- Archive date: `2026-09-18`
- Covered by final release: `no`
- Used in thesis: `no`
- Notes: `none`
