# SPG complete joint-CFG forward microbenchmark

- Representative states: `9`
- Warmup: `50`
- Timed calls: `300 x 3` per configuration
- Dynamic CUDA median: `26.952528 ms`
- Static CUDA median: `27.972800 ms`
- BUCKET_FULL_FORWARD_SPEEDUP: `0.963526x`
- Gate >=1.08x: `False`
- Static calls observed: `1015`
- Static fallbacks observed: `0`

| Metric | Dynamic | Static |
|---|---:|---:|
| CUDA median ms | 26.952528 | 27.972800 |
| CUDA P95 ms | 27.070646 | 28.750679 |
| Wall ms/call | 26.949342 | 28.128742 |
| Kernel count/call | 3011.0 | 2626.7 |
| Allocator calls/call | 800.0 | 625.0 |
| Sync calls/call | 250.1 | 144.1 |
| Scatter self CUDA ms/call | 1.767397 | 2.010395 |
| GPU active proxy | 99.99% | 99.93% |
| Peak incremental VRAM bytes | 86663680 | 86663680 |

## Amdahl estimates

- Current bucket (33.6578% coverage): `0.987419x`
- Two equal non-overlapping buckets: `0.975151x`
- Three equal non-overlapping buckets: `0.963526x`

The multi-bucket estimates are theoretical upper bounds under identical
per-bucket speedup and non-overlapping coverage assumptions.
