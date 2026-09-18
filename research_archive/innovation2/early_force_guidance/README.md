# Early Closed-Loop Force Guidance

## Research Question

Does a closed-loop force controller improve generation quality?

## Motivation

Repeated feedback will outperform the frozen F0 reference.

## Change from Previous Stage

Added online feedback beyond F0.

## Implementation

Archive category: `FULL_ARCHIVE`. Reproducibility level: `PARTIAL`.

Original server path `/mnt/datasets-livsyn/dxl/mattergen_v1_closed_loop_guidance/experiments/closed_loop_force_guidance_p0`. Portable reproduction must use repository-relative paths or an explicit user-supplied data root; the original server path is provenance only.

## Dataset / Seeds

Sixteen-seed P0 with force and quality endpoints. Seed manifests are copied when small; otherwise their immutable source path and external-data hash are recorded.

## Main Results

F1 provided no gain over F0 and the route failed.

## Decision

Archive status: **FAIL**.

Historical conclusion at the time: See the immutable source report for the exact historical GO/FAIL wording. This archive does not overwrite or retroactively relabel that source.

Final thesis interpretation after later confirmation: F1 provided no gain over F0 and the route failed.

## Why This Route Was Stopped

The preregistered P0 comparison failed.

## What It Motivated Next

Keep the simpler bounded F0 design; do not tune the failed loop.

## Provenance

- Source worktree: `/mnt/datasets-livsyn/dxl/mattergen_v1_closed_loop_guidance`
- Source branch: `experiment/closed-loop-force-guidance-p0`
- Source commit: `1af301c5e534ff6b9b9f96ea5dab48036d9c46af`
- Original path: `experiments/closed_loop_force_guidance_p0`
- Archive date: `2026-09-18`
- Covered by final release: `no`
- Used in thesis: `supporting`
- Notes: `none`
