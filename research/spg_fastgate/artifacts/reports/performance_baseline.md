# SPG-MatterGen Fast Gate performance baseline

| Method | Batch | Fixed-8 throughput (samples/h) | Speedup vs B1 | Median sample latency (s) | GPU util (%) |
|---|---:|---:|---:|---:|---:|
| C0 | 1 | 427.513 | 1.000× | 64.290 | 25.51 |
| C0 | 4 | 1284.138 | 3.004× | 21.458 | 36.49 |
| C0 | 8 | 1833.581 | 4.289× | 15.194 | 44.06 |
| A0 | 1 | 422.881 | 1.000× | 64.511 | 25.58 |
| A0 | 4 | 1242.076 | 2.937× | 22.110 | 36.12 |
| A0 | 8 | 1781.077 | 4.212× | 15.419 | 43.57 |
