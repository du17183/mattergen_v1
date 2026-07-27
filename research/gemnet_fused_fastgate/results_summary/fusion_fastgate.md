# GemNet K2 local-compile fast gate

- Validation states: `100`
- Numerical equivalent: `False`
- Maximum absolute error: `0.0009765625`
- Maximum aggregate relative L2: `1.50667709e-06`
- Minimum cosine: `1`
- K2 chain CUDA speedup: `1.1240x`
- Full joint-CFG forward CUDA speedup: `1.0092x`
- Pre-E2E gate: `False`

The implementation locally compiles only the nine profiled AtomUpdate/OutputBlock
forwards. The full GemNet and sampler remain eager; edge ordering and native
scatter-add semantics are unchanged.
