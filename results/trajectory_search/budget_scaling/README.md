# Budget Scaling K=0–4: final result

Status: **complete**, 128 new paired primary seeds (`2100010000–2100010127`), disjoint from the Best-of-2 cohort and historical C1-128. K arms are nested on the **same** fresh cohort; treating them as independent seed groups would invalidate paired statistics. The branch order `GPulse → PPulse → APulse → CPulse`, SAFE-A verifier/acceptance and budgets were fixed before outcomes. All 640 candidate branches and five selected-method cohorts received property/quality evaluation. No DFT verification was performed.

| Width | Score calls/sample | Property MAE ↓ | C0−K gain, paired 95% CI | W/T/L vs K0 | E-hull ↓ | Stable ↑ | NUS ↑ | Validity ↑ |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| K0 | 2,000 (1.0×) | 0.035141 | reference | 0/128/0 | 0.094254 | 0.625000 | 0.367188 | 1.000000 |
| K1 | 3,200 (1.6×) | 0.031636 | `[0.002108, 0.005062]` | 39/89/0 | 0.081928 | 0.679688 | 0.375000 | 1.000000 |
| K2 | 4,400 (2.2×) | 0.028454 | `[0.004537, 0.009102]` | 59/69/0 | 0.080551 | 0.687500 | 0.382812 | 1.000000 |
| K3 | 5,600 (2.8×) | 0.028419 | `[0.004538, 0.009140]` | 59/69/0 | 0.080787 | 0.687500 | 0.390625 | 1.000000 |
| K4 | 6,800 (3.4×) | **0.024669** | `[0.007774, 0.013398]` | 72/56/0 | **0.078960** | **0.710938** | **0.406250** | 1.000000 |

All four K>0 arms pass the prespecified cohort-level proxy quality guardrails versus K0. The marginal paired improvements (previous width minus new width; **post hoc descriptive**, 20,000 paired bootstrap resamples) are K1 `0.003505` (CI `[0.002142, 0.005132]`, W/T/L `39/89/0`), K2 `0.003182` (CI `[0.001546, 0.005187]`, `29/99/0`), K3 `0.000035` (CI `[0, 0.000105]`, `1/127/0`) and K4 `0.003750` (CI `[0.001979, 0.005748]`, `26/102/0`). The K3 step is effectively a plateau; K4 improves further, so K2 is **not** the empirically best width in this finite experiment.

Because the candidate sets are nested and SAFE-A only accepts improvements over C0, nonincreasing per-seed selected property error as K grows is **guaranteed by the selection rule**. The descending average curve alone is not independent evidence that more search generalizes or that shared-prefix branching beats independent sampling. The irregular marginal gains depend on the frozen action order. No post-result K selection or change to the historical C1 method is made.

Artifacts: `budget_scaling_summary.csv`, `incremental_paired_gains.csv`, and `mean_mae_bootstrap_20k.csv` are derived outputs. Immutable source outputs remain in `results/budget_scaling/` (paired rows, frozen 20k primary comparisons, complete candidate and MatterSim outcomes, score-call accounting). Reproduce derived files with `experiments/trajectory_search/finalize_evidence.py`.
