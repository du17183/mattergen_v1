# SPG builder microbenchmark (initial)

- Representative states: `9`
- Warmup: `50`
- Timed calls: `300 x 3` per configuration
- Dynamic median: `4.910064 ms`
- Static persistent median: `2.720464 ms`
- STATIC_BUILDER_SPEEDUP: `1.804863x`
- Gate >=2.25x: `False`

| Configuration | CUDA median ms | CUDA P95 ms | Wall ms/call | Kernels | Allocator calls | Sync calls |
|---|---:|---:|---:|---:|---:|---:|
| D0_dynamic_joint | 4.910064 | 4.963043 | 4.859952 | 811.0 | 294.0 | 143.1 |
| D1_static_no_share | 5.078400 | 6.655536 | 5.610340 | 803.0 | 236.0 | 58.1 |
| D2_static_joint_share | 2.720384 | 3.508546 | 2.911832 | 420.9 | 119.0 | 37.1 |
| D3_static_persistent | 2.720464 | 3.511691 | 2.900434 | 420.9 | 119.0 | 37.1 |
