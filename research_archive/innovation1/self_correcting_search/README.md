# Self-Correcting Search P0

## Research Question

Can a learned probe identify timesteps that benefit from correction?

## Motivation

Probe confidence can select reliable correction steps.

## Change from Previous Stage

Added a probe-driven search controller.

## Implementation

Archive category: `SUMMARY_ONLY`. Reproducibility level: `SUMMARY_ONLY`.

Original server path `/mnt/datasets-livsyn/dxl/mattergen_v1_self_correcting_search/experiments/self_correcting_search_p0`. Portable reproduction must use repository-relative paths or an explicit user-supplied data root; the original server path is provenance only.

## Dataset / Seeds

Probe reliability audit and small diagnostic checkpoint. Seed manifests are copied when small; otherwise their immutable source path and external-data hash are recorded.

## Main Results

No reliable timesteps were found and P0 was not run.

## Decision

Archive status: **ABANDONED_BEFORE_EVALUATION**.

Historical conclusion at the time: See the immutable source report for the exact historical GO/FAIL wording. This archive does not overwrite or retroactively relabel that source.

Final thesis interpretation after later confirmation: No reliable timesteps were found and P0 was not run.

## Why This Route Was Stopped

The probe gate failed.

## What It Motivated Next

Avoid search controllers without an observable reliable signal.

## Provenance

- Source worktree: `/mnt/datasets-livsyn/dxl/mattergen_v1_self_correcting_search`
- Source branch: `experiment/self-correcting-search-p0`
- Source commit: `1af301c5e534ff6b9b9f96ea5dab48036d9c46af`
- Original path: `experiments/self_correcting_search_p0`
- Archive date: `2026-09-18`
- Covered by final release: `no`
- Used in thesis: `no`
- Notes: `none`
