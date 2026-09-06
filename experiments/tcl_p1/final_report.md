TCL P1 FINAL:
CLEAR GO

DFT_VERIFIED=False

# Frozen Cross-Timestep Consistency Learning P1

This is a 32-new-paired-seed independent validation of the exact P0 checkpoints.
No model was retrained or modified. No DML, TCL+DML, Adaptive CFG, P2, or new
sampling method was run. All energy, force, stability, and relaxation results are
MatterSim-5M surrogate results, not DFT.

## 1–8. Branch, independent seeds, and frozen models

- Branch: `experiment/tcl-p1`
- Parent: `2f95aa0c361cc539f13a9184ddb94f2a04d2efa0`
- Final HEAD: reported by `git rev-parse HEAD` in the final handoff because a
  commit cannot contain its own hash.
- Actual P1 seed range: 81000–81031, 32 paired seeds.
- Historical independence: confirmed before creating any P1 output by a
  project-wide exact textual seed scan that excluded decimal substrings, plus a
  directory-name scan. There were no historical uses of this interval.
- C0: official local MatterGen `dft_mag_density` pretrained checkpoint; base
  checkpoint SHA256
  `01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e`.
- FT0: frozen P0 `experiments/tcl_dml_p0/checkpoints/FT0/model.pt`; SHA256
  `1cd254568463ae4d7774193fa0a0e97925cdad3b6c505c1e5f5dd4e7968f6960`.
- TCL: frozen P0 `experiments/tcl_dml_p0/checkpoints/TCL/model.pt`; SHA256
  `8dd9879f11edea7f25f587ae72f928a6f3ab2e5c81fc9400c13642364ad90443`.
- P1 training steps: 0. Checkpoint hashes were checked against both hard-coded
  frozen values and the P0 training summary. The P0 generator was called
  directly; TCL objective code, lambda, warmup, pairing, teacher, and fields were
  unchanged.

## 9–14. Generation and MatterSim success

All three methods used the same paired seeds, target `dft_mag_density=0.1`,
constant CFG=2, original 1000-step Predictor/Corrector, one corrector step per
time, batch size 1, and deterministic seed-level GPU generation.

| Method | Generation | MatterSim-5M relaxation | Mean generation seconds |
|---|---:|---:|---:|
| C0 | 32/32 (100%) | 32/32 (100%) | 116.303 |
| FT0 | 32/32 (100%) | 32/32 (100%) | 116.666 |
| TCL | 32/32 (100%) | 32/32 (100%) | 116.065 |

MatterSim used the frozen 5M potential and `fmax=0.05 eV/A`.

## 15–26. Complete quality, geometry, force, and robustness metrics

| Method | E-hull mean | median | P95 | max | Stable | NUS | Novel | Unique |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.110993 | 0.076359 | 0.337312 | 0.459128 | 59.375% | 28.125% | 68.750% | 100.000% |
| FT0 | 0.144492 | 0.140453 | 0.228714 | 0.241159 | 21.875% | 9.375% | 84.375% | 96.875% |
| TCL | 0.126695 | 0.097185 | 0.301317 | 0.403529 | 50.000% | 31.250% | 75.000% | 100.000% |

E-hull units are eV/atom.

| Method | RMSD mean | median | P95 | max |
|---|---:|---:|---:|---:|
| C0 | 0.050350 | 0.018685 | 0.281331 | 0.441483 |
| FT0 | 0.117457 | 0.038621 | 0.429455 | 1.223281 |
| TCL | 0.091053 | 0.039525 | 0.263821 | 1.111283 |

RMSD units are angstrom.

| Method | Atomic-force mean | Max-force mean | median | P95 | max |
|---|---:|---:|---:|---:|---:|
| C0 | 0.149360 | 0.327101 | 0.136477 | 1.266641 | 1.800696 |
| FT0 | 0.293451 | 0.582211 | 0.547338 | 1.354409 | 1.809878 |
| TCL | 0.200569 | 0.430989 | 0.197293 | 1.718366 | 2.378615 |

Force units are eV/angstrom.

| Method | Relax mean | median | P95 | max | >100 | >200 | >400 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 47.125 | 26.5 | 173.15 | 285 | 3 (9.375%) | 2 (6.250%) | 0 (0%) |
| FT0 | 75.094 | 49.0 | 234.40 | 502 | 6 (18.750%) | 3 (9.375%) | 1 (3.125%) |
| TCL | 52.938 | 45.5 | 139.50 | 178 | 5 (15.625%) | 0 (0%) | 0 (0%) |

## 27. Complete severe-outlier list

Every sample satisfying RMSD >0.5 A, structure max-force >1 eV/A, relaxation
>200, or relaxation >400 is retained below. Stable/NUS are binary per-structure
statuses from the official evaluation.

| Method | seed | formula | E-hull | Stable | NUS | RMSD | atomic force | max force | steps | flags |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| C0 | 81005 | Eu2ZnFeO7 | 0.459128 | 0 | 0 | 0.436062 | 0.664611 | 1.272564 | 285 | force; >200 |
| C0 | 81013 | Eu3NiH5O7 | 0.137594 | 0 | 0 | 0.103008 | 0.449390 | 1.261796 | 96 | force |
| C0 | 81022 | Eu3C3O10 | -0.074023 | 1 | 1 | 0.154732 | 0.536668 | 1.124939 | 221 | force; >200 |
| C0 | 81026 | Eu2Fe(PO3)2 | 0.448161 | 0 | 0 | 0.441483 | 0.648135 | 1.800696 | 134 | force |
| FT0 | 81000 | Mn3HC | 0.233356 | 0 | 0 | 0.072815 | 0.491538 | 1.060452 | 53 | force |
| FT0 | 81006 | Mn5As4 | 0.098197 | 1 | 1 | 1.223281 | 0.166396 | 0.266232 | 163 | RMSD |
| FT0 | 81011 | Mn3C | 0.172837 | 0 | 0 | 0.175320 | 0.626216 | 1.549376 | 165 | force |
| FT0 | 81022 | Mn5GaC2 | 0.196658 | 0 | 0 | 0.299537 | 0.573194 | 1.809878 | 274 | force; >200 |
| FT0 | 81029 | Mn8AlC3 | 0.224916 | 0 | 0 | 0.279398 | 0.464284 | 1.046762 | 202 | force; >200 |
| FT0 | 81031 | Mn2C | 0.220457 | 0 | 0 | 0.588244 | 0.504578 | 1.194890 | 502 | RMSD; force; >200; >400 |
| TCL | 81002 | Mn7Co2 | 0.049012 | 1 | 1 | 1.111283 | 0.316154 | 0.631316 | 111 | RMSD |
| TCL | 81003 | Li2Mn5FeO4 | 0.403529 | 0 | 0 | 0.081159 | 0.745587 | 1.212054 | 51 | force |
| TCL | 81013 | Mn10GaNiPC3 | 0.265605 | 0 | 0 | 0.099909 | 0.722417 | 2.378615 | 62 | force |
| TCL | 81030 | NdReO4 | 0.223340 | 0 | 0 | 0.097936 | 0.863517 | 2.337192 | 118 | force |

TCL has four severe-tail structures, equal to C0's four and fewer than FT0's six.
It has one RMSD tail versus C0 zero/FT0 two, three force tails versus C0 four/FT0
five, and no >200 or >400 relaxation tail versus C0 two/zero and FT0 three/one.
The occurrence is not systematically higher than C0 or FT0, although the TCL
maximum force of 2.379 eV/A must remain visible in later validation.

## 28. FT0 minus C0

Ordinary continued full finetuning did not reproduce its P0 trade-off. In P1 it
worsened mean E-hull by 0.03350 eV/atom, Stable by 37.5 percentage points, NUS by
18.75 points, RMSD by 0.06711 A, atomic force by 0.14409 eV/A, max force by
0.25511 eV/A, and relaxation by 27.97 steps. Stable CI
[-56.25,-15.625] pp and both force CIs excluded zero in the unfavorable
direction. Novel rose 15.625 points, while Unique fell 3.125 points. Thus
ordinary finetuning itself degraded core quality and geometry on these new seeds.

## 29–36. TCL minus C0 paired effects

All confidence intervals are percentile 95% intervals from 20,000 paired
seed-level bootstrap resamples. Proportions are percentage points.

| Metric | TCL-C0 mean delta | 95% CI | favorable / unfavorable / tie |
|---|---:|---:|---:|
| E-hull | +0.015702 eV/atom | [-0.032946,+0.063927] | 12 / 20 / 0 |
| Stable | -9.375 pp | [-31.250,+12.500] | 6 / 9 / 17 |
| NUS | +3.125 pp | [-18.750,+21.875] | 6 / 5 / 21 |
| Novel | +6.250 pp | [-15.625,+28.125] | 8 / 6 / 18 |
| RMSD | +0.040702 A | [-0.027703,+0.127280] | 11 / 21 / 0 |
| Atomic force | +0.051209 eV/A | [-0.029624,+0.137381] | 10 / 22 / 0 |
| Max-force mean | +0.103888 eV/A | [-0.087770,+0.309182] | 10 / 22 / 0 |
| Relaxation mean | +5.8125 steps | [-19.000,+28.470] | 10 / 21 / 1 |

TCL is not demonstrated to be globally superior to C0: its mean E-hull, Stable,
and geometry means are numerically worse, though every corresponding CI crosses
zero. It satisfies the preregistered C0-side gate narrowly: no clear E-hull
worsening, a positive NUS direction, positive Novel, unchanged 100% Unique, and
no systematic new tail.

## 37–44. TCL minus FT0 paired effects — primary comparison

| Metric | TCL-FT0 mean delta | 95% CI | favorable / unfavorable / tie |
|---|---:|---:|---:|
| E-hull | -0.017797 eV/atom | [-0.057059,+0.021977] | 18 / 14 / 0 |
| Stable | +28.125 pp | [+9.375,+46.875] | 11 / 2 / 19 |
| NUS | +21.875 pp | [+6.250,+37.500] | 8 / 1 / 23 |
| Novel | -9.375 pp | [-25.000,+6.250] | 2 / 5 / 25 |
| RMSD | -0.026404 A | [-0.128775,+0.069754] | 19 / 13 / 0 |
| Atomic force | -0.092882 eV/A | [-0.173351,-0.006803] | 24 / 8 / 0 |
| Max-force mean | -0.151221 eV/A | [-0.363951,+0.082483] | 24 / 8 / 0 |
| Relaxation mean | -22.1563 steps | [-59.032,+5.469] | 18 / 12 / 2 |

The mechanism comparison is positive across thermodynamic quality and
geometry/robustness. Stable, NUS, and atomic-force confidence intervals exclude
zero favorably. E-hull, RMSD, max force, and relaxation retain favorable P0
directions but their CIs cross zero. Novel does not reproduce its P0 gain versus
FT0, but its -9.375 pp CI crosses zero; TCL remains more novel than C0 and has
100% Unique, so this is not a novelty/uniqueness collapse.

## 45–55. Win counts, P0 replication, and tails

Paired win counts are included in the two tables above. For the continuous
primary TCL-vs-FT0 metrics, wins were E-hull 18/32, RMSD 19/32, atomic force
24/32, max force 24/32, and relaxation 18/32 with two ties. Versus C0 they were
12/32, 11/32, 10/32, 10/32, and 10/32 respectively.

P0 direction replication versus FT0:

| P0 signal | P1 direction | Replicated? |
|---|---|---|
| E-hull down | -0.017797, CI crosses zero | yes, direction only |
| NUS up | +21.875 pp, CI entirely positive | yes |
| Novel up | -9.375 pp, CI crosses zero | no |
| RMSD down | -0.026404, CI crosses zero | yes, direction only |
| Atomic force down | -0.092882, CI entirely negative | yes |
| Max force down | -0.151221, CI crosses zero | yes, direction only |
| Relaxation down | -22.156, CI crosses zero | yes, direction only |

Six of seven P0 directions reproduced. The two strongest P1 mechanism signals
are NUS recovery and lower atomic force, supplemented by a clear Stable recovery.
TCL did produce one RMSD and three force severe outliers, so the P0 claim of zero
severe tails did not literally reproduce. However, total tail occurrence did not
exceed C0 and was lower than FT0, and TCL had no >200-step relaxation cases.

## 56–61. Scientific answer and decision

TCL is stably better than FT0 in the sense required for this frozen P1:
Stable +28.125 pp and NUS +21.875 pp both have paired bootstrap intervals above
zero, atomic force has an interval below zero, and E-hull/RMSD/max-force/
relaxation all point favorably. This cannot be attributed to ordinary continued
finetuning because FT0 is the same-budget control and performed materially worse.

The evidence therefore supports the bounded statement that Cross-Timestep
Consistency provides additional benefit over ordinary continued full finetuning
for this frozen model. It does not yet support claiming TCL is uniformly superior
to the original pretrained C0: C0 comparisons are mixed and uncertain.

TCL P1 FINAL: **CLEAR GO** under the preregistered criteria. Relative to FT0 it
combines a core generation-quality gain (Stable/NUS) with a geometry/force gain
(atomic force), protects Unique, avoids a clear Novel collapse, and has no
systematic robustness degradation.

Recommendation to enter 64-seed P2: **yes**. The one next step is a fresh
64-paired-seed frozen C0/FT0/TCL validation using this exact generator,
checkpoints, and MatterSim pipeline, with special attention to whether TCL's
C0-relative E-hull/geometry and force outlier maximum converge favorably. P2 was
not started here.
