# SPG single-bucket capacity utilization

- States: `21541`
- Candidate/edge/triplet capacities: `18000` / `626` / `32156`
- GemNet receives compact valid edge/triplet tensors; fixed-capacity padding is not passed into message passing.
- Invalid edge compute share in GemNet: `0%`
- Invalid triplet compute share in GemNet: `0%`

| Percentile | Candidate util. | Edge util. | Triplet util. |
|---:|---:|---:|---:|
| P10 | 41.667% | 63.578% | 59.740% |
| P25 | 44.444% | 71.885% | 67.987% |
| P50 | 60.000% | 79.872% | 76.228% |
| P75 | 69.444% | 87.859% | 83.282% |
| P90 | 84.028% | 95.527% | 90.994% |
| P95 | 100.000% | 96.486% | 92.679% |
| P99 | 100.000% | 97.444% | 94.595% |
| P100 | 100.000% | 100.000% | 100.000% |

The main risk is builder-side fixed candidate and triplet-mask work,
not padded GemNet message passing. The microbenchmark must determine
whether that workspace overhead erases the allocation/launch savings.
