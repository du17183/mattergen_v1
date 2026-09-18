# Late-stage MatterSim position-force guidance P0

## Decision

**LATE_PHYSICS_SIGNAL_RELIABLE=YES**
**MATTERSIM_FORCE_GUIDANCE_P0=GO**
**NEXT=FORMAL32**

Adaptive CFG used? **NO**. `SURROGATE_PROPERTY_EVAL=True` and
`DFT_VERIFIED=False`. Formal32 and Formal256 were not run.

## Diagnostic

The frozen 32-seed clean-estimate diagnostic passed: the best late point was
`t_norm=0.02`, where clean-estimate MaxF predicted final MaxF with Spearman
rho=0.925587 and 100% finite/valid MatterSim evaluations. The final row marked
`requested_t_norm=0.00` is actually the state after the epsilon predictor at
`actual_model_t=0.001`, not a mathematical t=0 model evaluation.

## Frozen guidance mechanism

Only the position predictor is guided for `t<=0.02`. MatterSim force is
evaluated on complete clean `x0_hat(t)`, centered to remove translation, and
converted using the row-vector identity `delta_frac = delta_cart @ inv(cell)`.
The single frozen lambda is 1.0; there was no sweep. Nominal per-atom Cartesian
contribution is 0.005 A, hard-capped at 0.01 A (below the protocol's absolute
0.02 A limit). A candidate with minimum periodic distance below 0.5 A falls
back to the untouched official score. Cell, atomic-type, and stress scores are
never guided.

Across 320 eligible calls, 320 were accepted and
0 fell back. The largest observed correction was
0.005 A.

## Paired P0 effect (16 fresh seeds)

| Metric | B0 | F0 | F0-B0 / improvement |
|---|---:|---:|---:|
| Mean MaxF (eV/A) | 0.304028 | 0.253392 | 16.655% improvement |
| Mean atomic force (eV/A) | 0.148469 | 0.123007 | -0.0254619 |
| Mean RMSD (A) | 0.116299 | 0.114453 | -0.00184585 |
| Mean E-hull (eV/atom) | 0.092021 | 0.0928301 | +0.000809155 |
| Stable | 75.00% | 75.00% | +0.00 pp |
| NUS | 31.25% | 31.25% | +0.00 pp |
| Mag MAE (A^-3) | 0.00994048 | 0.00988885 | -0.519% |
| Validity | 100.00% | 100.00% | -0.00 pp |
| End-to-end generation | 150.284s | 150.252s | 0.9998x |

MaxF paired wins/ties/losses for F0 are **15/0/1**. The paired
20,000-resample bootstrap 95% CI for relative mean MaxF improvement is
[10.171%,
 31.241%]. Full per-seed,
quality, efficiency, and bootstrap results are in the required CSV artifacts.

Seeds 710033 (GPU0) and 710034 (GPU4) completed under logged external GPU
contention from PIDs 3760852-3760855. After those pairs completed, no further
A work was dispatched to GPU0/4; the remaining unique seeds used GPUs5/6/7.
The runtime gate uses every preregistered pair, including the contended runs.

## Gate application

GO requires mean MaxF improvement >=15%, >=11/16 paired wins, and all five
guardrails. The frozen gate produced `GO`. This stage performs no rescue
sweep and does not automatically start Formal32.

## Execution audit

Two technical, pre-scientific failures are retained under `failed_attempts/`:
the first diagnostic smoke exposed a NumPy-2 `np.math` compatibility issue,
and the first relaxation launcher used a checkpoint path relative to the
independent worktree. The fixes were respectively a compatibility alias and
the already frozen main-project absolute checkpoint path with SHA256
verification. Neither changed a seed, structure, model, threshold, lambda, or
scientific protocol; the relaxation path failure occurred before any structure
was relaxed. All 32 diagnostic trajectories and all 16 P0 pairs completed.
