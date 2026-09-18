# Phase A — Branch-Compatible Oracle Audit

This audit generated no new structures and launched no GPU sampling. It retained the existing 24/12/12 split and restricted candidates to `GPulse`, `APulse`, `PPulse`, and `CPulse` plus exact C0.

## Offline-test oracle

| Selector | C0 MAE | Branchable Oracle-All MAE | Relative improvement | Stable | Novel | Unique | NUS | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SAFE-A | 0.02879873 | 0.02004107 | 30.41% | 75.00% | 58.33% | 100.00% | 50.00% | 50.00% |
| SAFE-B | 0.02879873 | 0.02132689 | 25.95% | 66.67% | 75.00% | 100.00% | 50.00% | 50.00% |

## Budget comparison

| Method | Policies / definition | Property MAE | Recovered headroom | Compute ×C0 |
|---|---|---:|---:|---:|
| Oracle-K2 | per-seed hindsight | 0.02004107 | 100.00% | 2.20× |
| Fixed-K2 | `CPulse;GPulse` selected on train | 0.02244455 | 72.56% | 2.20× |
| Random-K2 | 20k deterministic allocations | 0.02269349 mean | 69.71% mean | 2.20× |

The Oracle-K result is an existence upper bound because the registered definition gives hindsight access to the best policy identity for each seed. Random-K confidence intervals are randomization intervals over candidate allocations on the fixed 12-seed test cohort. Mixed Unique/NUS were recomputed; cross-seed matcher edges in the complete test pool: 0.

```text
BRANCHABLE_ORACLE_IMPROVEMENT = 0.304099
BRANCHABLE_ORACLE_K2_RECOVERY = 1.000000
BEST_FIXED_BRANCHABLE_K2 = CPulse;GPulse
BEST_FIXED_BRANCHABLE_K2_RECOVERY = 0.725557
RANDOM_BRANCHABLE_K2_RECOVERY = 0.697131
K2_SHARED_PREFIX_COMPUTE = 2.20
BRANCHABLE_BUDGETED_HEADROOM = GO
ADAPTIVE_SELECTION_NEEDED = YES
```

Phase A next action: `START_PHASE_B`. All physical-quality and property values remain surrogate evaluations; `DFT_VERIFIED=False`.
