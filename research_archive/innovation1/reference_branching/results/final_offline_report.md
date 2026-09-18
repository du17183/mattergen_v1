# Reference-Preserved Budgeted Adaptive CFG — offline report

No new structures or GPU samples were generated. The analysis used the completed 48-seed × 12-policy Field-Decoupled matrix.

## Frozen split and data audit

- Train / validation / test: 24 / 12 / 12, assigned by `SHA256(rpb-a-cfg-offline-v1:seed)`.
- Candidate rows: 576; policies: 12; seeds: 48.
- Exact-structure duplicate groups: 159.
- Maximum property-error spread within exact duplicates: 4.9e-08.
- Frozen meaningful-improvement tolerance: 1e-06.
- Mixed-batch Unique and NUS were recomputed after every reported terminal selection.

## Primary offline-test results (SAFE-A)

| Method | Property MAE | Recovered Oracle headroom | Estimated compute ×C0 |
|---|---:|---:|---:|
| C0 | 0.02879873 | 0.00% | 1.00× |
| Oracle-All | 0.01642374 | 100.00% | 10.40× |
| Oracle-K2 | 0.01642374 | 100.00% | 2.33× |
| Fixed-K2 `AC19;G19` | 0.02332932 | 44.20% | 3.00× |

`Oracle-K2` equals Oracle-All by the registered definition because hindsight may choose the policy identity separately for every seed; one adaptive slot already contains the best safe policy. It is a theoretical budget oracle, not evidence that an allocator can identify that policy.

## Allocator audit and stop decision

The Field sampler retained final outcomes and `prefix_state_sha256`, but did not serialize atomic/position/cell residual statistics, conditional-unconditional score norms, alignment, EMA, or stage trace summaries. The hash is an integrity identifier and was not converted into a predictor. Candidate final property and MatterSim outcomes were prohibited as allocator inputs.

Consequently, residual, logistic/linear, and shallow-tree sample-adaptive allocators are `NOT_EVALUABLE`. A model using only policy identity produces the same ranking for every seed and collapses to Fixed-K. The random allocator is reported as a baseline but does not establish adaptive value.

```text
BUDGETED_HEADROOM = GO
ADAPTIVE_ALLOCATION_VALUE = NOT_SUPPORTED
BEST_FIXED_K2 = AC19;G19
BEST_ADAPTIVE_K2 = NOT_EVALUABLE_NO_PREBRANCH_FEATURES
ORACLE_K2 = 0.01642374
ORACLE_ALL = 0.01642374
METHOD_COLLAPSED_TO_BEST_OF_K = TRUE
INNOVATION1_FINAL_STATUS = MIXED
```

Gate A passes as a retrospective existence bound. Gate B fails because no deployable sample-adaptive allocator can be evaluated from the registered offline inputs. Under the frozen stop rule, branch-sampler implementation, reference reproduction, P0, and Formal256 are not run.

The train-selected Fixed-K2 requires an estimated 3.00× C0 generation compute because both selected policies are full-trajectory constants. Even the hindsight Oracle-K2 averages 2.33× under the source sampler's cost model, above the preferred 2.2× target. The existing constant-policy outcomes cannot be reinterpreted as step-400 shared-prefix outcomes without generating a different experiment.

All results use CHGNet/MatterSim surrogate evaluations; `DFT_VERIFIED=False`.
