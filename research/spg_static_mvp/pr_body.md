## What changed

- Adds an exact fixed-capacity periodic graph builder for the frozen 8–12 atom,
  repetition ≤2 bucket.
- Shares periodic graph construction across conditional/unconditional joint CFG
  geometry.
- Adds strict 10,000-state graph equivalence, 64-state GemNet/joint-CFG
  numerical equivalence, capacity analysis, profiler-guided builder
  optimization, and 300×3 builder/full-forward benchmarks.
- Archives compact JSON/CSV/Markdown evidence and reproducible benchmark
  scripts; no weights, environments, trajectories, caches, or raw profiler
  traces are included.

## Why

This evaluates whether exact static periodic graph construction can reduce
dynamic graph launch/allocation overhead without changing MatterGen generation
semantics or quality.

## Result and impact

Correctness passes: 10,000/10,000 strict graph states use the static path with
zero fallback and 100% ordered edge/offset/triplet agreement; 64/64 joint-CFG
states are bitwise identical through all GemNet blocks and final
atomic/position/cell outputs.

Performance does not meet the frozen gates after the single permitted
profiler-supported optimization:

- Builder: 4.683872 ms dynamic vs 2.666656 ms static, 1.756459× (<2.25× gate).
- Full joint-CFG forward: 26.952528 ms dynamic vs 27.972800 ms static,
  0.963526× (<1.08× gate; about 3.79% slower).

Therefore the evidence-backed outcome is `SINGLE_BUCKET_NO_GO`. The protocol
forbids starting 8-seed generation when these gates fail, so no quality run was
launched.

## Root cause

Joint-CFG graph sharing reduces kernels, allocations, and synchronizations, but
exact fixed-workspace indexing/scatter/topology packing costs erase the launch
savings in the complete GemNet forward. GemNet receives compact valid graphs,
so the regression is not caused by padded edges/triplets entering message
passing.

## Checks

- SPG specialized + GemNet regressions: 16/16 passed.
- Strict graph equivalence after optimization: 10,000/10,000 passed, zero
  fallback, max distance/vector error 0.
- Joint-CFG/GemNet numerical equivalence after optimization: 64/64 passed,
  bitwise rate 100%, max error 0.
- Full repository suite with the project PyTorch compatibility variable:
  165/170 passed. The remaining 5 failures are in files unchanged from `main`:
  four legacy Corrector tests omit the now-required `dt`, and one training test
  uses a removed `ReduceLROnPlateau(verbose=...)` argument.
- `git diff --check`: passed.

This PR is intentionally draft and must not be merged before human review of
the No-Go evidence.
