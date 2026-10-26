# Phase B final report

This is a fresh-seed, surrogate-only study. No DFT verification is claimed.

## Decision

ADAPTIVE_ALLOCATION_VALUE = NOT_SUPPORTED
Validation-selected allocator = Linear_K2
Best fixed K2 = GPulse + PPulse
Held-out fixed K2 MAE = 0.02459741
Held-out adaptive K2 MAE = 0.02363429
Fixed recovered headroom = 0.7975
Adaptive recovered headroom = 0.8984
Adaptive Top-2 safe-oracle recall = 0.8571

## Frozen gates

B1_allocator_advantage = FAIL
B2_quality = PASS
B3_random = PASS
B4_recall = PASS

## Held-out test metrics

| Method | Property MAE | Recovered headroom | Top2 recall | Stable | NUS | Validity |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 0.03221501 | 0.0000 | 0.0000 | 0.6250 | 0.3750 | 1.0000 |
| Fixed_K2 | 0.02459741 | 0.7975 | 0.7143 | 0.6250 | 0.4375 | 1.0000 |
| Random_K2 | 0.02465528 | 0.7915 | 0.7143 | 0.6250 | 0.3750 | 1.0000 |
| Residual_K2 | 0.02647439 | 0.6010 | 0.7143 | 0.6250 | 0.4375 | 1.0000 |
| Linear_K2 | 0.02363429 | 0.8984 | 0.8571 | 0.6875 | 0.5000 | 1.0000 |
| GBDT_K2 | 0.02529944 | 0.7240 | 0.5714 | 0.6875 | 0.5000 | 1.0000 |
| Oracle_K2 | 0.02266367 | 1.0000 | 1.0000 | 0.6875 | 0.5000 | 1.0000 |
| Oracle_All | 0.02266367 | 1.0000 | 1.0000 | 0.6875 | 0.5000 | 1.0000 |

Unique and NUS above were recomputed on each method's final mixed held-out batch.

## Bootstrap

All paired intervals use 20,000 deterministic resamples.

Phase C is prohibited by the preregistered stop rule. Innovation 1 remains MIXED.
