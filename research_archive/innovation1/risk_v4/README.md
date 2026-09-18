# Risk-Calibrated CFG V4

## Research Question

Can a risk model selectively suppress harmful adaptive decisions?

## Motivation

Calibrated selective prediction will retain useful adaptations while rejecting risky ones.

## Change from Previous Stage

Added a frozen risk score and selective gate.

## Implementation

Archive category: `FULL_ARCHIVE`. Reproducibility level: `PARTIAL`.

Original server path `/mnt/datasets-livsyn/dxl/mattergen_v1_risk_cfg/experiments/risk_calibrated_adaptive_cfg`. Portable reproduction must use repository-relative paths or an explicit user-supplied data root; the original server path is provenance only.

## Dataset / Seeds

Development calibration, selective gate metrics, decisions and seed-level tables. Seed manifests are copied when small; otherwise their immutable source path and external-data hash are recorded.

## Main Results

The selective gate failed; no fresh P0 or Formal256 was permitted.

## Decision

Archive status: **FAIL**.

Historical conclusion at the time: See the immutable source report for the exact historical GO/FAIL wording. This archive does not overwrite or retroactively relabel that source.

Final thesis interpretation after later confirmation: The selective gate failed; no fresh P0 or Formal256 was permitted.

## Why This Route Was Stopped

The preregistered calibration gate did not pass.

## What It Motivated Next

Try explicit safe selection without changing the evaluated gate post hoc.

## Provenance

- Source worktree: `/mnt/datasets-livsyn/dxl/mattergen_v1_risk_cfg`
- Source branch: `experiment/risk-calibrated-adaptive-cfg`
- Source commit: `a4fda8a573ee0e372d3b53064e235ae87f0e670c`
- Original path: `experiments/risk_calibrated_adaptive_cfg`
- Archive date: `2026-09-18`
- Covered by final release: `yes/key-negative`
- Used in thesis: `supporting`
- Notes: `none`
