# MatterSim late clean-estimate diagnostic

**A_DIAGNOSTIC=GO**; **LATE_PHYSICS_SIGNAL_RELIABLE=True**; **NEXT=RUN_16_PAIRED_P0**.

The primary gate excludes the actual final structure (`t_norm=0.00`). That row is the state after the final predictor at the sampler epsilon `model_t=0.001`, not a model evaluation at mathematical t=0.

## Primary evidence

Best non-final `t<=0.05` MaxF-to-final-MaxF Spearman: **0.925587** at t=0.02 (n=32, p=3.422e-14).
Finite/valid fraction at that time: **100.00%**.
Energy/atom-to-final-E-hull Spearman at that time: **0.269428**.

## Q1: late-signal stability

|   t_norm |   n |   finite_valid_fraction |   obvious_ood_fraction |   median_maxF |   median_abs_maxF_delta_to_final |   median_abs_log10_maxF_ratio_to_final |   maxF_final_spearman |   maxF_final_spearman_pvalue |   correlation_n |
|---------:|----:|------------------------:|-----------------------:|--------------:|---------------------------------:|---------------------------------------:|----------------------:|-----------------------------:|----------------:|
|     0.1  |  32 |                       1 |                      0 |     0.130215  |                        0.037641  |                              0.195515  |              0.812683 |                  1.6091e-08  |              32 |
|     0.05 |  32 |                       1 |                      0 |     0.0869594 |                        0.0302772 |                              0.145758  |              0.707845 |                  5.86284e-06 |              32 |
|     0.02 |  32 |                       1 |                      0 |     0.0784385 |                        0.0120487 |                              0.089164  |              0.925587 |                  3.42233e-14 |              32 |
|     0.01 |  32 |                       1 |                      0 |     0.0825303 |                        0.0125316 |                              0.0831223 |              0.912757 |                  3.40887e-13 |              32 |
|     0    |  32 |                       1 |                      0 |     0.0917883 |                        0         |                              0         |              1        |                  0           |              32 |

## Q2: predictive correlations

All requested Spearman and Pearson combinations, including calibration/validation splits, are in `diagnostic_correlations.csv`. The decision uses the preregistered full 32-seed MaxF Spearman only.

## Q3: finite/OOD audit

Across 160 evaluations, finite rate=100.00%; obvious-OOD rate=0.00%. OOD was frozen as nonfinite, |energy/atom|>20 eV, MaxF>100 eV/A, minimum distance<0.5 A, or V/N outside [1,200] A^3.

All energy, force, stress, relaxation, and E-hull quantities are MatterSim-5M surrogate results. `DFT_VERIFIED=False`.
