# GBSA-TCL P1 FINAL

Final decision: **FAIL**. Frozen GBSA-TCL reproduced the late-score-drift mechanism, improved RMSD and relaxation cost, and preserved quality/diversity, but it did not reproduce the required atomic-force and maximum-force repair. Seed `85014` produced a severe force outlier (`atomic-force mean = 3.046884 eV/A`, `MaxF = 6.342232 eV/A`), so the frozen method does not pass the P1 force-robustness gate. No P2 or new training was started.

All energy, stability, force, and relaxation results below are MatterSim-5M surrogate results. They are not DFT results.

## Protocol and execution

1. Branch: `experiment/gbsa-tcl-p1`.
2. Final HEAD: recorded in the final handoff after committing this report. Starting/frozen P0 HEAD: `5a72cfde92bec7145105b4deef38f5fa7f2da8b5`.
3. P1 seed range: `85000-85031` (32 paired seeds).
4. Historical overlap: `none`. The only repository search hit resembling `850xx` was a decimal measurement in a nanoindentation CSV, not an experimental seed.
5. P1 training steps: `0`. No training, retraining, checkpoint/config switch, or P1-driven tuning occurred.
6. Frozen GBSA-TCL checkpoint SHA256: `1edc1c0c7848e9edaff9d73b135642fcfaa36e23591e4e2f336c39b3df3dcf0f`, identical to P0.
7. C0 generation success: `32/32` (100%), technical reruns `0`.
8. FT0 generation success: `32/32` (100%), technical reruns `0`.
9. Frozen TCL generation success: `32/32` (100%), technical reruns `0`.
10. Frozen GBSA-TCL generation success: `32/32` (100%), technical reruns `0`.
11. MatterSim-5M relaxation/evaluation success: `128/128` (32/32 per method).
12. `DFT_VERIFIED=False`. Necessary checks found no NaN/Inf, corrupt result, traceback, OOM, killed job, missing sample, or residual P1 process.

Common generation settings were target magnetic density `0.1`, constant `CFG=2`, 1000 diffusion steps, one corrector step per diffusion time, batch size 1, and 8-GPU seed-level parallelism. A sample never spanned multiple GPUs.

## Complete four-method results

13. The following tables are the complete four-method quality and geometry summary (`n=32` each).

14. E-hull in eV/atom:

| Method | Mean | Median | P95 | Max |
|---|---:|---:|---:|---:|
| C0 | 0.139579 | 0.116273 | 0.378462 | 0.702695 |
| FT0 | 0.148154 | 0.157720 | 0.259841 | 0.313541 |
| TCL | 0.103043 | 0.096274 | 0.212754 | 0.538298 |
| GBSA-TCL | 0.118655 | 0.080281 | 0.346181 | 0.580920 |

15-18. Stable, NUS, Novel, and Unique:

| Method | Stable | NUS | Novel | Unique |
|---|---:|---:|---:|---:|
| C0 | 14/32 (43.75%) | 7/32 (21.875%) | 24/32 (75.00%) | 32/32 (100%) |
| FT0 | 12/32 (37.50%) | 6/32 (18.75%) | 25/32 (78.125%) | 32/32 (100%) |
| TCL | 16/32 (50.00%) | 10/32 (31.25%) | 25/32 (78.125%) | 32/32 (100%) |
| GBSA-TCL | 19/32 (59.375%) | 15/32 (46.875%) | 27/32 (84.375%) | 32/32 (100%) |

19. RMSD in A:

| Method | Mean | Median | P95 | Max | RMSD > 0.5 |
|---|---:|---:|---:|---:|---:|
| C0 | 0.087994 | 0.026248 | 0.474451 | 0.831157 | 2 |
| FT0 | 0.098930 | 0.037353 | 0.413855 | 0.516041 | 1 |
| TCL | 0.066632 | 0.025302 | 0.181343 | 0.628265 | 1 |
| GBSA-TCL | 0.044114 | 0.033079 | 0.105243 | 0.335560 | 0 |

20. Mean atomic-force magnitude per structure in eV/A:

| Method | Mean | Median | P95 | Max |
|---|---:|---:|---:|---:|
| C0 | 0.127401 | 0.110726 | 0.276175 | 0.491502 |
| FT0 | 0.268413 | 0.236373 | 0.562784 | 0.710376 |
| TCL | 0.165544 | 0.098430 | 0.502538 | 0.621735 |
| GBSA-TCL | 0.179521 | 0.074903 | 0.306442 | 3.046884 |

21. Maximum force per structure in eV/A:

| Method | Mean | Median | P95 | P99 | Max | >1 | >2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.246207 | 0.182426 | 0.671498 | 0.994339 | 1.089802 | 1 | 0 |
| FT0 | 0.521636 | 0.437125 | 1.087043 | 1.341152 | 1.428661 | 5 | 0 |
| TCL | 0.354640 | 0.171174 | 1.080231 | 1.663700 | 1.888907 | 3 | 0 |
| GBSA-TCL | 0.391117 | 0.146603 | 0.800739 | 4.732783 | 6.342232 | 2 | 1 |

22. Relaxation steps:

| Method | Mean | Median | P95 | Max | >200 | >400 |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 58.3438 | 26.5 | 244.85 | 443 | 3 | 1 |
| FT0 | 86.7500 | 43.0 | 278.90 | 706 | 3 | 1 |
| TCL | 58.5938 | 42.5 | 183.90 | 338 | 2 | 0 |
| GBSA-TCL | 36.8750 | 27.0 | 79.70 | 214 | 1 | 0 |

## Paired comparisons

For all tables below, `delta = GBSA-TCL - baseline`. Negative is favorable for E-hull/RMSD/forces/steps; positive is favorable for Stable/NUS/Novel/Unique. Stable/NUS/Novel/Unique deltas and CIs are percentage points. W/T/L counts use the preferred direction and exact ties. Every CI uses 20,000 paired bootstrap resamples with seed `20260920`.

23. GBSA-TCL minus TCL:

| Metric | Mean delta | Paired 95% CI | W/T/L |
|---|---:|---:|---:|
| E-hull | +0.015613 | [-0.019134, +0.055169] | 17/0/15 |
| Stable | +9.375 pp | [-15.625, +31.250] | 9/17/6 |
| NUS | +15.625 pp | [-3.125, +34.375] | 8/21/3 |
| Novel | +6.250 pp | [-12.500, +25.000] | 6/22/4 |
| Unique | 0.000 pp | [0.000, 0.000] | 0/32/0 |
| RMSD | -0.022518 A | [-0.066777, +0.015883] | 17/0/15 |
| Atomic-force mean | +0.013977 eV/A | [-0.110942, +0.207864] | 18/0/14 |
| Structure MaxF | +0.036477 eV/A | [-0.232777, +0.434788] | 15/0/17 |
| Relaxation steps | -21.718750 | [-46.968750, +2.125000] | 22/0/10 |

Only RMSD and relaxation have favorable means among the four geometry/relaxation endpoints, and no core geometry CI excludes zero. Atomic-force and MaxF means are adverse because the method retains a rare but severe force tail.

24. GBSA-TCL minus C0:

| Metric | Mean delta | Paired 95% CI | W/T/L |
|---|---:|---:|---:|
| E-hull | -0.020924 | [-0.080819, +0.037636] | 20/0/12 |
| Stable | +15.625 pp | [-6.250, +37.500] | 9/19/4 |
| NUS | +25.000 pp | [+6.250, +43.750] | 10/20/2 |
| Novel | +9.375 pp | [-9.375, +28.125] | 6/23/3 |
| Unique | 0.000 pp | [0.000, 0.000] | 0/32/0 |
| RMSD | -0.043880 A | [-0.114561, +0.013897] | 14/0/18 |
| Atomic-force mean | +0.052120 eV/A | [-0.056278, +0.233566] | 18/0/14 |
| Structure MaxF | +0.144909 eV/A | [-0.087926, +0.523248] | 18/0/14 |
| Relaxation steps | -21.468750 | [-57.062500, +7.032031] | 15/0/17 |

This is not a broad TCL-Formal-like degradation across every endpoint: RMSD, relaxation, E-hull, and quality point estimates are favorable, and GBSA has no RMSD>0.5 or Steps>400 cases. However, C0 force protection is not secure: atomic-force/MaxF means are worse, MaxF>1 is 2 versus 1, and MaxF>2 is 1 versus 0.

25. GBSA-TCL minus FT0:

| Metric | Mean delta | Paired 95% CI | W/T/L |
|---|---:|---:|---:|
| E-hull | -0.029499 | [-0.082907, +0.030958] | 22/0/10 |
| Stable | +21.875 pp | [-3.125, +46.875] | 13/13/6 |
| NUS | +28.125 pp | [+9.375, +46.875] | 11/19/2 |
| Novel | +6.250 pp | [-12.500, +25.000] | 6/22/4 |
| Unique | 0.000 pp | [0.000, 0.000] | 0/32/0 |
| RMSD | -0.054816 A | [-0.108378, -0.005362] | 19/0/13 |
| Atomic-force mean | -0.088892 eV/A | [-0.219348, +0.116435] | 27/0/5 |
| Structure MaxF | -0.130519 eV/A | [-0.389162, +0.284651] | 25/0/7 |
| Relaxation steps | -49.875000 | [-98.375781, -11.593750] | 23/1/8 |

GBSA-TCL did not fall back to ordinary FT0: NUS, RMSD, and relaxation have favorable CIs excluding zero, while every reported point estimate is at least as good as FT0. Nevertheless, `FT0 < TCL < GBSA-TCL` is not a uniform chain: it holds for RMSD and main quality rates, but GBSA force means are worse than TCL.

## Mechanism, tails, and replication

26. TCL late-position relative L2 score drift versus C0: `0.202297`.
27. GBSA-TCL late-position relative L2 score drift versus C0: `0.051065`.
28. Late-position drift reduction: `74.757%` relative to TCL.
29. Late-cell relative drift: TCL `0.110302`, GBSA-TCL `0.037113`, a `66.353%` reduction.
30. Late-atomic relative drift: TCL `0.147981`, GBSA-TCL `0.052750`, a `64.354%` reduction.

The score probe used all 32 C0 plus all 32 TCL-P1 initial structures (64 states), identical `x_t`, time, condition, corruption, `CFG=2`, ten progress bins, and fixed noise seed `20260921`.

31. All severe GBSA-TCL seeds (none removed):

| Seed | Formula | E-hull | RMSD | Atomic-force mean | MaxF | Steps | Flags |
|---:|---|---:|---:|---:|---:|---:|---|
| 85008 | EuPt2 | 0.191977 | 0.335560 | 0.087210 | 0.133860 | 214 | Steps>200 |
| 85014 | Gd2Mn7Al2 | 0.580920 | 0.105228 | 3.046884 | 6.342232 | 63 | MaxF>1; MaxF>2 |
| 85023 | Eu6GdB2O | 0.471621 | 0.095769 | 0.377641 | 1.150462 | 61 | MaxF>1 |

Tail-count comparison `(RMSD>0.5, MaxF>1, MaxF>2, Steps>200, Steps>400)` is C0 `(2,1,0,3,1)`, FT0 `(1,5,0,3,1)`, TCL `(1,3,0,2,0)`, and GBSA-TCL `(0,2,1,1,0)`.

32. P0 to P1 direction replication (`delta = GBSA-TCL - TCL`):

| Metric | Preferred | P0 direction | P1 direction | P1 delta | P1 95% CI | Replicated? |
|---|---|---|---|---:|---:|---|
| E-hull | lower | favorable | unfavorable | +0.015613 | [-0.019134, +0.055169] | No |
| Stable | higher | favorable | favorable | +9.375 pp | [-15.625, +31.250] | Yes |
| NUS | higher | favorable | favorable | +15.625 pp | [-3.125, +34.375] | Yes |
| RMSD | lower | favorable | favorable | -0.022518 | [-0.066777, +0.015883] | Yes |
| Atomic-force mean | lower | favorable | unfavorable | +0.013977 | [-0.110942, +0.207864] | No |
| Structure MaxF | lower | favorable | unfavorable | +0.036477 | [-0.232777, +0.434788] | No |
| Relaxation steps | lower | favorable | favorable | -21.718750 | [-46.968750, +2.125000] | Yes |
| Novel | higher | tie | favorable | +6.250 pp | [-12.500, +25.000] | No (direction changed from tie) |
| Unique | higher | tie | tie | 0.000 pp | [0.000, 0.000] | Yes |
| Late-position drift | lower | favorable | favorable | -0.151232 | [-0.174322, -0.128840] | Yes |

P0 had only eight seeds and is used only as a directional reference.

## Gate answers

33. Mechanism/phenotype replication:
   - Gradient repair: **fixed checkpoint evidence retained, not independently re-trained or re-estimated in P1**. Checkpoint identity is exact; P0 evidence was clipping `0/1000` and maximum balanced auxiliary/base gradient ratio `0.877780 < 1`. P1 correctly ran zero training steps.
   - Score mechanism: **YES**. Late position/cell/atomic drift reductions are 74.757%/66.353%/64.354%; late-position paired CI excludes zero.
   - Geometry repair: **PARTIAL, overall NO for the required geometry/force package**. RMSD and relaxation repair reproduced, but atomic-force and MaxF mean directions did not.
   - Quality preservation: **YES**. Stable/NUS/Novel are higher than TCL and FT0 at the point-estimate level; Unique remains 100%.
34. Is RMSD clearly better than TCL? **Point estimate and practical magnitude: yes** (`-0.022518 A`, exceeding the 0.01 A target; P95 0.105243 vs 0.181343; zero RMSD>0.5 vs one). **Statistical clarity: no**, because the paired CI `[-0.066777, +0.015883]` crosses zero.
35. Is atomic force clearly better than TCL? **No**. Mean delta is adverse (`+0.013977 eV/A`) and the CI crosses zero, although the median and P95 are favorable. Seed 85014 creates a catastrophic maximum.
36. Is the maximum-force tail safe? **No**. P95 improves (0.800739 vs 1.080231) and MaxF>1 count improves (2 vs 3), but GBSA alone has a MaxF>2 event and its max/P99 are 6.342232/4.732783 versus TCL 1.888907/1.663700.
37. Did systematic geometry degradation reappear relative to C0? **Not as a broad systematic pattern, but C0 force protection failed**. RMSD and relaxation means/tails improve, while atomic-force and MaxF means/tails worsen due to a severe outlier. Therefore the method cannot be said to have safely regained the C0 geometry prior.
38. Did E-hull/Stable/NUS hold? **YES for preservation, not for superiority over TCL on every endpoint**. Versus FT0, all three point estimates improve; versus TCL, E-hull worsens slightly by 0.015613 eV/atom while Stable and NUS improve by 9.375 and 15.625 pp. No clear quality collapse occurred.
39. Did Novel/Unique collapse? **No**. GBSA-TCL is 84.375% Novel and 100% Unique, versus TCL 78.125% and 100%.
40. Final decision: **FAIL** under the predeclared P1 rules. The hard atomic-force direction and MaxF-tail gates fail; only one of three core geometry mean estimates is favorable, and none of their paired CIs excludes zero.
41. Proceed to frozen 64-seed P2? **NO**. P2 and Formal256 were not started.
42. If FAIL, discuss Strict Atomic-Only TCL Residual Adapter? **YES, as the preferred next-route discussion, but do not implement or tune it in this P1 round.** The P1 seeds must not be reused for adapter selection/tuning; any later method should use separate development seeds before a new independent validation.

## Core scientific answer

The frozen GBSA-TCL checkpoint independently reproduced the score-anchoring mechanism and preserved the promising RMSD/quality improvements, but it did **not** reproduce robust force repair. The P1 data therefore break the full proposed chain at the real force-robustness stage. The appropriate conclusion is FAIL, stop before P2, retain all adverse samples, and discuss a Strict Atomic-Only TCL Residual Adapter as the next distinct hypothesis rather than tuning GBSA-TCL on these validation seeds.

Primary result files: `quality_results.csv`, `paired_statistics.csv`, `tail_analysis.csv`, `score_drift_results.csv`, `score_drift_per_sample.csv`, `replication_table.csv`, `per_seed_results.csv`, and `generation_results.csv` in this directory.
