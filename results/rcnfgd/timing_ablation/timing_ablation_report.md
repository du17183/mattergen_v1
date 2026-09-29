# RC-NFGD Force Injection Timing Ablation

**Status: COMPLETE; 32 fresh paired seeds (750000–750031), four arms, no dropped pairs.**

Only force-feedback start time changed: C0 none; Early t≤1.0 (1000 predictor opportunities); Middle t≤0.5 (500); Late t≤0.02 (20, original RC-NFGD). The original MatterGen/MatterSim checkpoints, frozen sampler, force formula and safety fallback were unchanged. Each method uses 2000 MatterGen score calls per sample. Results are **not equal force-compute comparisons**; changing the start time also changes the cumulative force-call budget.

Execution provenance: two earlier wrapper attempts stopped on environment-compatibility/configuration-key errors before any guided-arm structures were produced. Twelve fully completed C0 records from the second attempt were retained and deterministically reused; the third attempt completed every remaining C0 and all guided arms. No seed, checkpoint, force formula, threshold, acceptance rule, or evaluator was tuned. Failed-attempt logs and status files are preserved under `experiments/rcnfgd_timing_ablation_20260929/` in the experiment workspace.

The pre-relaxation MatterSim MaxF is primary; mean force, CHGNet magnetic-density property MAE, validity, Stable, E-hull and NUS are secondary surrogate/official pipeline metrics. No DFT calculation was performed. Means use n=32; std is sample standard deviation. Paired intervals use 20,000 percentile bootstrap resamples with seed 20260929, candidate−baseline sign; reported intervals are exploratory and not multiplicity-adjusted.

| Method | Force calls/sample mean±std | End-to-end wall s mean±std | CUDA-event interval s mean±std | MaxF mean±std (eV/Å) | Property MAE mean±std | Stable | NUS |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.0±0.0 | 86.7±1.5 | 86.7±1.5 | 0.18986±0.25256 | 0.01515±0.02150 | 0.781 | 0.312 |
| Early | 931.5±26.0 | 258.5±192.0 | 258.2±192.0 | 0.16860±0.27755 | 0.00838±0.00894 | 0.750 | 0.156 |
| Middle | 500.0±0.2 | 106.5±3.0 | 106.3±3.0 | 0.14072±0.17955 | 0.01120±0.01044 | 0.781 | 0.406 |
| Late | 20.0±0.0 | 88.3±2.6 | 88.1±2.6 | 0.13864±0.21623 | 0.01522±0.02151 | 0.781 | 0.312 |

The other required metrics (mean±sample std) are:

| Method | Mean force (eV/Å) | Validity | E-hull (eV/atom) |
|---|---:|---:|---:|
| C0 | 0.08450±0.09846 | 1.000±0.000 | 0.06549±0.06896 |
| Early | 0.07294±0.09224 | 1.000±0.000 | 0.07125±0.06907 |
| Middle | 0.05816±0.05968 | 1.000±0.000 | 0.07429±0.06711 |
| Late | 0.06051±0.08720 | 1.000±0.000 | 0.06548±0.06895 |

Relative to the C0 wall-time mean, Early costs 2.98×, Middle 1.23× and Late 1.02×. This is a descriptive throughput comparison, not a controlled active-GPU-time measurement. The CSV tables preserve unrounded mean, standard deviation and bootstrap mean CI for every metric, and paired bootstrap CI for all seven metrics in all six arm contrasts.

CUDA events bracket the sampling call and may include CPU waiting; they are **not** a measure of active GPU-SM utilization. Potential-loading time is included in end-to-end wall time but not in the CUDA-event interval.

## Paired MaxF contrasts

| Candidate − baseline | Mean delta (eV/Å) | Paired 95% CI | Better/tie/worse pairs |
|---|---:|---:|---:|
| Early − C0 | -0.02127 | [-0.12918, 0.08591] | 20/0/12 |
| Middle − C0 | -0.04915 | [-0.14368, 0.02438] | 21/0/11 |
| Late − C0 | -0.05123 | [-0.06834, -0.03770] | 32/0/0 |
| Middle − Early | -0.02788 | [-0.12117, 0.05256] | 12/0/20 |
| Late − Early | -0.02996 | [-0.13136, 0.06633] | 20/0/12 |
| Late − Middle | -0.00208 | [-0.06685, 0.07727] | 18/0/14 |

## Stage interpretation

The force ranking among Early, Middle and Late is not resolved by both unadjusted paired intervals; do not claim a unique best start stage.

Late versus C0 lowers MaxF by 0.05123 eV/Å (paired 95% CI [-0.06834, -0.03770]; 32/32 favorable pairs) and mean force by 0.02399 eV/Å ([-0.02983, -0.01879]). Its property MAE is slightly higher by 0.000066 Å⁻³ ([0.000010, 0.000149]); Stable, NUS and validity are unchanged on this cohort. Early has lower mean property MAE than C0, but its MaxF interval crosses zero and its point-estimate NUS is lower. Middle lowers mean force versus C0 with an unadjusted paired interval excluding zero, while its MaxF interval crosses zero. These are exploratory, unadjusted contrasts and do not establish a unique best arm.

This is a joint change of injection timing and cumulative force-computation budget, so it cannot isolate a pure timing effect at fixed compute. Force opportunities, actual calls, accepted/fallback events, wall time and CUDA-event intervals are preserved per seed. Unfavorable outcomes and guidance fallbacks remain included.

On the cohort mean (Property MAE, MaxF) plane, nondominated methods are: Early, Middle, Late. This descriptive Pareto check does not include POST from another seed cohort and does not establish statistical dominance.

Full 128-row per-seed table: `timing_ablation_summary.csv`; per-method mean/std/mean bootstrap interval: `timing_method_summary.csv`; every paired 20k interval: `timing_paired_bootstrap.csv`. Existing NOT_RUN placeholders were preserved as `*_NOT_RUN_archived.*`.

No claim of DFT-level stability, universal late optimality, or superiority over all post-processing methods is supported.
