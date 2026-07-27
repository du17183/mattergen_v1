# SPG single-bucket performance summary

## Frozen protocol

- Bucket: 8–12 atoms, periodic repetitions at most 2
- State coverage: 33.6578125%
- Representative real states: 9, spanning 148–626 edges and
  2,592–32,156 triplets
- Warmup: 50 calls
- Measurement: 300 calls × 3 rounds, CUDA Events and wall clock
- Precision: FP32
- Disallowed features: BF16, `torch.compile`, CUDA Graph, larger batches,
  approximate neighbors, graph truncation

## Exactly one profiler-supported optimization

The initial profiler identified full-capacity candidate selection as the
largest remaining builder hotspot: scatter/index-put and top-k operated over
18,000 candidate slots after joint-CFG graph sharing had already reduced the
number of graph builds. The one allowed optimization replaced this stage with
ordered compact packing of only the selected candidates (at most 600) and
ordered representative-edge packing.

The optimization does not change periodic images, cutoff, top-50 selection,
edge order, offsets, symmetric-edge construction, or triplet construction.
After the change:

- specialized and GemNet tests: 16/16 passed;
- strict graph equivalence: 10,000/10,000 static, 0 fallback, all ordered
  edge/offset/triplet match rates 100%, distance/vector max error 0;
- 64-state joint-CFG/GemNet validation: bitwise rate 100%, max error 0.

## Builder result

| Measurement | Dynamic joint CFG | Static persistent | Speedup | Gate |
|---|---:|---:|---:|---:|
| Initial | 4.910064 ms | 2.720464 ms | 1.804863× | fail |
| After the one optimization | 4.683872 ms | 2.666656 ms | 1.756459× | fail |

The optimized static latency improved by about 1.98% relative to its initial
measurement, but the within-run frozen speedup is only 1.756459×, below the
required 2.25×. Therefore `STATIC_BUILDER_PERFORMANCE_GO=False`.

## Complete joint-CFG score forward result

| Metric | Dynamic | Static |
|---|---:|---:|
| CUDA median | 26.952528 ms | 27.972800 ms |
| CUDA P95 | 27.070646 ms | 28.750679 ms |
| Wall time/call | 26.949342 ms | 28.128742 ms |
| Kernel count/call | 3,011.0 | 2,626.7 |
| Allocator calls/call | 800.0 | 625.0 |
| Synchronization calls/call | 250.1 | 144.1 |
| Scatter self CUDA/call | 1.767397 ms | 2.010395 ms |

The complete static forward is 0.963526× the dynamic baseline (about 3.79%
slower), below the required 1.08×. Kernel launches, allocations, and
synchronizations decrease, but the fixed-workspace builder adds enough
indexing/scatter work to erase the launch savings. GemNet receives compact
valid graphs, so this regression is not caused by padded edges or padded
triplets entering message passing.

At the measured per-bucket forward ratio, Amdahl estimates are also negative:

- current 33.6578% coverage: 0.987419×;
- two equal non-overlapping buckets: 0.975151×;
- three equal non-overlapping buckets: 0.963526×.

These multi-bucket numbers are theoretical projections, not measured results.

## Gate decision

Both performance gates fail after the one permitted optimization:

```text
STATIC_BUILDER_SPEEDUP=1.756459 < 2.25
BUCKET_FULL_FORWARD_SPEEDUP=0.963526 < 1.08
STATIC_BUILDER_PERFORMANCE_GO=False
BUCKET_FULL_FORWARD_PERFORMANCE_GO=False
```

The frozen protocol therefore forbids the 8-seed stage. The evidence-backed
termination state is `SINGLE_BUCKET_NO_GO`; this is a performance No-Go, not a
correctness failure.
