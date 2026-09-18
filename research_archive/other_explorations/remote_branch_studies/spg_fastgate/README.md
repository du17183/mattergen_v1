# SPG MatterGen Fastgate

## Research Question

Can static periodic-graph execution accelerate MatterGen safely?

## Motivation

Static graph handling enables faster generation.

## Change from Previous Stage

Added SPG prototypes and quality checks.

## Implementation

Archive category: `SUMMARY_ONLY`. Reproducibility level: `SUMMARY_ONLY`.

Git object path `origin/feature/spg-mattergen-fastgate:research/spg_fastgate`. Portable reproduction must use repository-relative paths or an explicit user-supplied data root; the original server path is provenance only.

## Dataset / Seeds

Fastgate report. Seed manifests are copied when small; otherwise their immutable source path and external-data hash are recorded.

## Main Results

The overall fastgate advanced, while native batching and field-safe BF16 did not.

## Decision

Archive status: **MIXED**.

Historical conclusion at the time: See the immutable source report for the exact historical GO/FAIL wording. This archive does not overwrite or retroactively relabel that source.

Final thesis interpretation after later confirmation: The overall fastgate advanced, while native batching and field-safe BF16 did not.

## Why This Route Was Stopped

Subcomponents failed and required an MVP study.

## What It Motivated Next

Evaluate a narrowly scoped static-graph MVP.

## Provenance

- Source worktree: `remote-only`
- Source branch: `origin/feature/spg-mattergen-fastgate`
- Source commit: `a6718d2c0a0ba362b3c7cb8d03c1da6fd5258123`
- Original path: `git:research/spg_fastgate`
- Archive date: `2026-09-18`
- Covered by final release: `no`
- Used in thesis: `no`
- Notes: `none`
