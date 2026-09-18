# Counterfactual Oracle V3

## Research Question

Does per-sample counterfactual selection reveal usable adaptive-CFG headroom?

## Motivation

If harmful V1 decisions are avoidable, an oracle should expose a consistent selection advantage.

## Change from Previous Stage

Evaluated counterfactual candidates instead of committing to one online controller.

## Implementation

Archive category: `FULL_ARCHIVE`. Reproducibility level: `PARTIAL`.

Original server path `/mnt/datasets-livsyn/dxl/mattergen_v1_cfg_oracle/experiments/counterfactual_adaptive_cfg_v3`. Portable reproduction must use repository-relative paths or an explicit user-supplied data root; the original server path is provenance only.

## Dataset / Seeds

Oracle audit, candidate metrics and paired counterfactual outcomes. Seed manifests are copied when small; otherwise their immutable source path and external-data hash are recorded.

## Main Results

The original oracle promotion gate failed, while later analysis retained evidence of avoidable harmful decisions.

## Decision

Archive status: **MIXED**.

Historical conclusion at the time: See the immutable source report for the exact historical GO/FAIL wording. This archive does not overwrite or retroactively relabel that source.

Final thesis interpretation after later confirmation: The original oracle promotion gate failed, while later analysis retained evidence of avoidable harmful decisions.

## Why This Route Was Stopped

No deployable predictor followed from the oracle evidence.

## What It Motivated Next

Calibrate risk and safe-selection rules, then move toward branching.

## Provenance

- Source worktree: `/mnt/datasets-livsyn/dxl/mattergen_v1_cfg_oracle`
- Source branch: `experiment/counterfactual-adaptive-cfg-v3`
- Source commit: `1af301c5e534ff6b9b9f96ea5dab48036d9c46af`
- Original path: `experiments/counterfactual_adaptive_cfg_v3`
- Archive date: `2026-09-18`
- Covered by final release: `yes/key-negative`
- Used in thesis: `supporting`
- Notes: `none`
