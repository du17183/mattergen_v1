# Independent Best-of-2 versus Fixed-K2: final result

Status: **complete**, 128 paired fresh primary seeds (`2100000000–2100000127`), target `dft_mag_density=0.1`. The independent second sample uses the separately registered child seeds `2100001000–2100001127`. This cohort is separate from the historical C1-128 and from Budget Scaling. All 512 branch candidates received property and MatterSim quality evaluation; selected-method quality was evaluated as three complete 128-structure cohorts. No DFT verification was performed.

| Method | MatterGen score calls/sample | Property MAE ↓ | E-hull (eV/atom) ↓ | Stable ↑ | NUS ↑ | Validity ↑ | Novel ↑ | Unique ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 2,000 (1.0×) | 0.037623 | 0.123872 | 0.515625 | 0.312500 | 1.000000 | 0.757812 | 1.000000 |
| Independent Best-of-2 | 4,000 (2.0×) | **0.028097** | **0.096393** | **0.625000** | **0.351562** | 1.000000 | 0.679688 | 1.000000 |
| Fixed-K2 | 4,400 (2.2×) | 0.029506 | 0.100629 | 0.617188 | 0.328125 | 1.000000 | 0.687500 | 1.000000 |

For the primary paired CHGNet absolute-error endpoint, C0 minus Independent Best-of-2 is `+0.009527` (20,000-resample paired percentile 95% CI `[+0.006439, +0.012891]`; W/T/L = `36/92/0`). C0 minus Fixed-K2 is `+0.008117` (CI `[+0.005533, +0.011054]`; W/T/L = `53/75/0`). Both pass the prespecified cohort-level proxy quality guardrails versus C0.

The **direct** paired comparison is Independent Best-of-2 minus Fixed-K2 = `−0.001410` (95% CI `[−0.005039, +0.002167]`; W/T/L = `41/59/28`, where W means Fixed-K2 has lower error). The mean favors Best-of-2, but the interval crosses zero. There is **no evidence that Fixed-K2 outperforms independent Best-of-2**, let alone that a trajectory-specific advantage has been established. The two methods also have unequal score-call budgets and unequal terminal-verifier counts; see `docs/best_of_two_fairness_audit.md`.

The property metric and terminal selection both use the same CHGNet proxy. The E-hull/stability guardrail and quality outcomes use MatterSim. They are surrogate outcomes, susceptible to verifier selection optimism; they are not DFT-confirmed material properties. Novelty, uniqueness and NUS are outcomes, not selector inputs.

Artifacts: `best_of_2_summary.csv`, `mean_mae_bootstrap_20k.csv`, `compute_normalized_comparison.json`, and `generation_wall_time_accounting.csv` are reproducible derived outputs. Immutable source outputs remain in `results/search_baseline/`, especially `paired_results.csv`, `selected_outcomes.csv`, `table_best_of_n.csv`, `bootstrap_20k.json`, `compute_accounting.json`, all branch and mixed-quality directories, and per-seed run summaries. Reproduce derived files with `experiments/trajectory_search/finalize_evidence.py`.
