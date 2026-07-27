# SPG builder microbenchmark (optimized)

- Representative states: `9`
- Warmup: `50`
- Timed calls: `300 x 3` per configuration
- Dynamic median: `4.683872 ms`
- Static persistent median: `2.666656 ms`
- STATIC_BUILDER_SPEEDUP: `1.756459x`
- Gate >=2.25x: `False`

| Configuration | CUDA median ms | CUDA P95 ms | Wall ms/call | Kernels | Allocator calls | Sync calls |
|---|---:|---:|---:|---:|---:|---:|
| D0_dynamic_joint | 4.683872 | 4.736205 | 4.642950 | 811.0 | 294.0 | 143.1 |
| D1_static_no_share | 4.977520 | 6.533929 | 5.497970 | 815.0 | 236.0 | 58.1 |
| D2_static_joint_share | 2.669584 | 3.450626 | 2.847308 | 426.9 | 119.0 | 37.1 |
| D3_static_persistent | 2.666656 | 3.448178 | 2.845181 | 426.9 | 119.0 | 37.1 |
