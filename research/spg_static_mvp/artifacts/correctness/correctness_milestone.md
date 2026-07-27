# SPG static periodic graph correctness milestone

- Branch: `feature/spg-static-periodic-graph-mvp`
- Base: `main@9bc6747a3ddfd26db6d931bcdb6df5d299844544`
- Frozen bucket: 8–12 atoms, periodic repetitions at most 2
- Recorded states: 64,000 across 32 C0 trajectories
- Strict equivalence manifest: 10,000 states
- Static-path states: 10,000/10,000
- Fallback states: 0
- Ordered edge/offset/triplet match rates: 100%
- Maximum edge-distance and direction-vector error: 0
- Joint CFG: duplicate geometry reproduces the original batch-two CUDA
  operation shape before sharing top-50, symmetric-edge, and triplet work
- Specialized + GemNet regressions: 16 passed
- Isolated upstream randomized RDF regression: passed on rerun
- `git diff --check`: passed

The large raw trajectory states remain external under
`/data/dxl/results/spg_static_mvp`. Compact configuration, distribution,
manifest metadata, and equivalence summaries are archived with this source.
