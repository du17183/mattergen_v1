# SPG Fast Gate profiler breakdown

| Method | Periodic | Triplet | Graph total | Scatter | GemNet | Blocks | CFG | Predictor | Corrector |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 11.1592% | 2.2935% | 6.1508% | 7.5639% | 80.5311% | 18.4416% | 89.7317% | 3.5616% | 4.0907% |
| A0 | 7.6273% | 3.2188% | 8.0876% | 10.5257% | 88.9082% | 26.0668% | 97.6210% | 4.0423% | 2.5563% |

- C0-B1 nvidia-smi mean-utilization proxy: 25.5109%
- Nsight full-process GPU activity coverage: 3.3313%
- Nsight CPU launch gap share: 59.6621%
- SM active / memory bandwidth hardware counters: unavailable (NVGPUCTRPERM)
- CUDA operator call count estimate in profiled run: 242083
