# Q3 E3-PCR Frozen64

## Research Question

Does Q3 reproduce and is the learned gate necessary?

## Motivation

The refiner and learned gate both outperform controls.

## Change from Previous Stage

Ran frozen64 with always-on/random controls.

## Implementation

Archive category: `SUMMARY_ONLY`. Reproducibility level: `SUMMARY_ONLY`.

Git object path `origin/feature/q3-e3-pcr-frozen64:reports/q3_e3_pcr/frozen64`. Portable reproduction must use repository-relative paths or an explicit user-supplied data root; the original server path is provenance only.

## Dataset / Seeds

Remote final report and ablation. Seed manifests are copied when small; otherwise their immutable source path and external-data hash are recorded.

## Main Results

Force benefit reproduced, but the learned-gate mechanism was not supported at this stage.

## Decision

Archive status: **MIXED**.

Historical conclusion at the time: See the immutable source report for the exact historical GO/FAIL wording. This archive does not overwrite or retroactively relabel that source.

Final thesis interpretation after later confirmation: Force benefit reproduced, but the learned-gate mechanism was not supported at this stage.

## Why This Route Was Stopped

Formal confirmation was still required.

## What It Motivated Next

Run frozen Formal256 and report mechanism separately.

## Provenance

- Source worktree: `remote-only`
- Source branch: `origin/feature/q3-e3-pcr-frozen64`
- Source commit: `87853b030ff7d9350ef41574bf0c3f24f6093dc0`
- Original path: `git:reports/q3_e3_pcr/frozen64`
- Archive date: `2026-09-18`
- Covered by final release: `no`
- Used in thesis: `no`
- Notes: `none`
