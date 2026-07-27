# Persistent multi-trajectory B1 runtime

| Workers/GPU | 8-GPU samples/hour | Median latency (s) | GPU util mean | Peak memory MiB | Speedup | Bitwise |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 452.973 | 60.340 | 39.99 | 1420 | 1.0000x | True |
| 2 | 526.820 | 104.421 | 87.10 | 2491 | 1.1630x | True |
| 4 | 535.090 | 195.303 | 81.85 | 4856 | 1.1813x | True |

Selected workers/GPU: `4`
Runtime speedup: `1.1813x`
Final state: `GPU_ACCELERATION_NO_GO`

Each task remains batch_size=1, FP32, full Predictor/Corrector, original dynamic graph,
with an independently seeded process/CUDA context. Only residency and scheduling change.
