# GBSA-TCL P0 FINAL

Final decision: **GO**. This is an eight-seed directional P0 result, not a formal statistical conclusion. All quality and relaxation metrics use MatterSim-5M/project evaluation; **DFT_VERIFIED=False**.

1. Branch: `experiment/gbsa-tcl-p0`.
2. Result commit before this generated report: `868f27b`. The exact final handoff HEAD is reported after committing this report.
3. P0 seeds: `84000–84007` (8 paired seeds).
4. Historical overlap: none; `81000–81031`, `82000–82063`, and `83000–83255` were excluded.

5. Total/trainable parameters: `48,760,443` / `4,250,213`.
6. Trainable ratio: `8.7165%`.
7. Frozen modules: input embedding, noise/time encoder, GemNet blocks 1–4, position/output heads, cell/lattice heads, property embeddings, and every other pretrained parameter.
8. Trainable modules: condition adapter/mixin `4,198,400` parameters; atomic head `51,813` parameters.
9. Old cell TCL was genuinely removed: `True`. No clean-cell `1/alpha_t` objective is present.
10. Final loss: two-view original MatterGen base loss + warm-started atomic TCL + normalized periodic position TCL + piecewise C0 position score anchor + piecewise C0 cell score anchor. A sampled exact-gradient cap targets each auxiliary at no more than 1× base before global clipping; no atomic C0 anchor was added in P0.
11. Reverse-progress anchor schedule (`p=1-t`): position `0.01/0.05/0.10`; cell `0.01/0.025/0.05` over `p=[0,.3)/[.3,.7)/[.7,1]`, with 100-step warmup.
12. Training steps: 50-step independent diagnostic, then a fresh 1000-step main run.
13. Main training wall time: `323.656 s`.

14. Base gradient mean: `0.160981`.
15. Atomic TCL gradient mean: `0.069034`; gradient/base mean `0.4850`; cosine vs base `0.5098`.
16. Position TCL gradient mean: `0.002317`; gradient/base mean `0.0165`.
17. Position-anchor gradient mean: `0.001675`; gradient/base mean `0.0117`.
18. Cell-anchor gradient mean: `0.000784`; gradient/base mean `0.0051`.
19. Maximum measured balanced auxiliary/base gradient ratio: `0.877780×`. All balance scales remained 1.0, so the guardrail did not need to hide an unstable objective.
20. Gradient clipping: `0/1000` at L2 threshold `1.0` (original TCL: 1000/1000).
21. NaN/Inf: none; teacher frozen/eval/no optimizer = `True/True/True`; initial student/teacher field max-absolute differences were all zero.

22. TCL late-position relative L2 drift vs C0: `0.205963`.
23. GBSA-TCL late-position relative L2 drift vs C0: `0.052908`.
24. Late-position drift reduction: `74.31%`, evaluated on all 24 new P0 structures at 10 fixed noise bins with shared corruption and CFG=2. Late-cell drift also fell from `0.116445` to `0.038039`.

25–27. Absolute P0 quality, geometry, force, and relaxation results:

| Method | E-hull (eV/atom) | Stable | NUS | Novel | Unique | RMSD (Å) | Atomic force | Max force | Relax steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.077954 | 62.5% | 25.0% | 62.5% | 100.0% | 0.028622 | 0.172185 | 0.348243 | 42.000 |
| TCL | 0.099393 | 50.0% | 37.5% | 87.5% | 100.0% | 0.148818 | 0.190957 | 0.327073 | 70.000 |
| GBSA-TCL | 0.069278 | 87.5% | 75.0% | 87.5% | 100.0% | 0.034297 | 0.060137 | 0.098325 | 21.625 |

28–32. Tail counts:

| Method | RMSD>0.5 | MaxF>1 | MaxF>2 | Steps>200 | Steps>400 |
|---|---:|---:|---:|---:|---:|
| C0 | 0 | 0 | 0 | 0 | 0 |
| TCL | 1 | 1 | 0 | 1 | 0 |
| GBSA-TCL | 0 | 0 | 0 | 0 | 0 |

33. Every paired-seed result:

| Method | Seed | Formula | E-hull | RMSD | Atomic force | Max force | Steps | Tail |
|---|---:|---|---:|---:|---:|---:|---:|---|
| C0 | 84000 | LiFe3O4 | 0.05483 | 0.01575 | 0.10901 | 0.17577 | 22 | - |
| C0 | 84001 | Fe5Co3 | 0.04420 | 0.00398 | 0.05212 | 0.11569 | 16 | - |
| C0 | 84002 | Eu3Ti(FeO3)3 | 0.13727 | 0.11089 | 0.45727 | 0.89732 | 122 | - |
| C0 | 84003 | Eu2HNO6 | -0.02157 | 0.03829 | 0.36790 | 0.90223 | 68 | - |
| C0 | 84004 | Mn2GeS4 | 0.05317 | 0.00450 | 0.04605 | 0.06172 | 29 | - |
| C0 | 84005 | SrEu2Hg | 0.03042 | 0.00137 | 0.00884 | 0.01739 | 1 | - |
| C0 | 84006 | Mn3SiC | 0.13665 | 0.03573 | 0.19509 | 0.30190 | 40 | - |
| C0 | 84007 | EuGdO3 | 0.18866 | 0.01846 | 0.14120 | 0.31391 | 38 | - |
| TCL | 84000 | EuZnGe2 | 0.08572 | 0.02633 | 0.11591 | 0.27109 | 25 | - |
| TCL | 84001 | EuFeN2 | 0.08869 | 0.01278 | 0.08474 | 0.15605 | 29 | - |
| TCL | 84002 | Mn12PC3 | 0.14553 | 0.04356 | 0.63396 | 1.05954 | 54 | MaxF>1 |
| TCL | 84003 | LaGd4DyCo3I | 0.14346 | 0.61721 | 0.09159 | 0.16527 | 208 | RMSD>0.5;Steps>200 |
| TCL | 84004 | Gd2Al3Fe2 | 0.02003 | 0.03513 | 0.05758 | 0.09733 | 28 | - |
| TCL | 84005 | EuHg | 0.01134 | 0.00135 | 0.01617 | 0.02543 | 12 | - |
| TCL | 84006 | Mn5VSiGeC2 | 0.10603 | 0.17800 | 0.28803 | 0.43253 | 88 | - |
| TCL | 84007 | Eu2Mn3C5 | 0.19433 | 0.27618 | 0.23967 | 0.40934 | 116 | - |
| GBSA-TCL | 84000 | Eu3Pt2Pb3 | 0.04029 | 0.01362 | 0.03140 | 0.06757 | 17 | - |
| GBSA-TCL | 84001 | Eu3Pb | 0.03361 | 0.13843 | 0.13133 | 0.15983 | 26 | - |
| GBSA-TCL | 84002 | EuPt3 | 0.22698 | 0.00203 | 0.02148 | 0.03908 | 26 | - |
| GBSA-TCL | 84003 | Gd6Pb2CN | 0.03284 | 0.00193 | 0.02583 | 0.04713 | 7 | - |
| GBSA-TCL | 84004 | Li4Eu2Tl | 0.05629 | 0.07798 | 0.04041 | 0.06777 | 27 | - |
| GBSA-TCL | 84005 | EuHg | 0.01132 | 0.00018 | 0.00239 | 0.00346 | 10 | - |
| GBSA-TCL | 84006 | Sc2Zn2C | 0.08767 | 0.03883 | 0.20372 | 0.35439 | 40 | - |
| GBSA-TCL | 84007 | Eu5MgAu4 | 0.06523 | 0.00138 | 0.02454 | 0.04737 | 20 | - |

34. Three required repairs: gradient repair **YES**; score-drift repair **YES**; real-geometry repair **YES**.
35. Final P0 decision: **GO**.
36. Independent 32-seed P1 is scientifically justified, but was not started in this round.
37. Failure layer: not applicable. The n=8 uncertainty remains, and no formal claim should be made from P0.
38. Strict Atomic-Only TCL Residual Adapter: **not recommended as the immediate next step** because GBSA-TCL passed P0; retain it only as the preregistered fallback if independent P1 fails.

## Scientific interpretation

The root-cause chain is supported by a direct intervention: removing unbounded cell consistency and freezing the pretrained geometry path eliminated gradient domination (`0/1000` clips), while C0 score anchoring/partial freezing reduced late-position drift by about `74.3%`. On the same new paired seeds, GBSA-TCL reduced RMSD, atomic force, maximum force, and relaxation cost relative to frozen TCL without a visible quality collapse. The effect is strong enough for P1, but the sample size is deliberately too small for confirmatory inference.
