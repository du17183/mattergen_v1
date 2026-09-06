# Distribution-Balanced Quality Adapter (M2-DB) — Final Report

## Decision

**16-seed decision: FAIL. No 32-seed confirmation was started.**

The clean-x0 analysis supports a composition-dependent CHGNet quality signal, and
within-cluster ranking successfully removes the observed cluster imbalance in
training weights. However, that intervention does not produce the required
quality–distribution Pareto improvement on 16 new paired seeds. M2-DB does not
beat C0 on mean E-hull or structure max-force mean, restores only one novel
sample relative to M2, and does not recover composition-family or element
coverage. Its RMSD and relaxation tails are substantially worse.

This is evidence that the observed generation contraction is not explained only
by global quality-weight assignment. Under the frozen stop rule, the current
Quality Adapter route should stop; the recommended next route is a Global
Transformer Adapter. No Transformer experiment was run in this branch.

All physical-quality results use the MatterSim-5M surrogate and are marked
`DFT_VERIFIED=False`; they are not DFT results.

## 1. Branch and scope

- Branch: `experiment/distribution-balanced-quality-adapter`
- Parent P1b Final HEAD: `71f4619f9c555dc2aef2e656fde2dcacb589b573`
- Experiment directory: `experiments/distribution_balanced_quality_adapter/`
- Training source: the frozen original-M2 clean-x0 subset only (96 train, 32 validation)
- Screen seeds: 73000–73015, disjoint from the P1/P1b 70000–72031 ranges
- Real generations: 4 methods × 16 seeds = 64/64 successful
- Sampling: dft_mag_density 0.1, CFG 2.0, original Predictor and Corrector,
  1000 diffusion steps, one independent seed per GPU process

No historical formal/test/generated sample was used for clustering or training.
No MatterSim result, E-hull, novelty label, or final-test information entered the
composition descriptors.

## 2. Composition descriptors and quality correlation

The 15 composition-only descriptors were:

`num_atoms`, `num_unique_elements`, `mean_atomic_number`,
`std_atomic_number`, `mean_atomic_mass`, `std_atomic_mass`,
`oxygen_fraction`, `transition_metal_fraction`, `alkali_fraction`,
`alkaline_earth_fraction`, `halogen_fraction`,
`mean_electronegativity`, `std_electronegativity`,
`mean_atomic_radius`, and `std_atomic_radius`.

Atomic statistics are stoichiometry-weighted. Atomic radii use pymatgen's
reported atomic radius with its calculated radius as a fallback. The scaler and
KMeans fit used only the 96 training structures; validation structures were only
transformed and assigned to fitted centers.

The following descriptors passed the fixed force-max signal threshold
`|Spearman rho| >= 0.25 and p < 0.05`:

| Descriptor | Spearman rho | p-value |
|---|---:|---:|
| mean_atomic_radius | -0.3985 | 0.0000578 |
| num_atoms | +0.3105 | 0.002078 |
| mean_electronegativity | +0.3077 | 0.002288 |
| mean_atomic_number | -0.2619 | 0.009943 |
| mean_atomic_mass | -0.2544 | 0.012392 |

Element-family fractions alone were weak in this small sample: oxygen,
transition-metal, alkali, alkaline-earth, and halogen force-max correlations all
had `|rho| <= 0.105`. The full force-max/force-mean/force-RMS/energy table is in
`composition_bias_analysis.csv`. Cross-composition energy/atom is reported only
as an auxiliary CHGNet label and is never treated as E-hull.

## 3. Clustering and mechanism sanity check

The requested K=4 fit produced train-cluster sizes **37 / 6 / 13 / 40**. The
6-sample cluster violates the minimum-size rule, so the one permitted fallback
to K=3 was used. Final train-cluster sizes are **21 / 39 / 36**; validation
assignments are 9 / 15 / 8.

Force-max group statistics for the selected K=3 clusters were:

| Cluster | Train n | Force-max mean | Force-max median |
|---:|---:|---:|---:|
| 0 | 21 | 0.2476 | 0.1691 |
| 1 | 39 | 0.1911 | 0.1390 |
| 2 | 36 | 0.1485 | 0.1188 |

The omnibus Kruskal p-value is 0.216 and is descriptive rather than a training
gate. Descriptor-level correlations and, more directly, the global-M2 high
weight concentration justified the intervention.

| Cluster | Original M2 mean weight | Original high-weight share | M2-DB mean weight | M2-DB high-weight share |
|---:|---:|---:|---:|---:|
| 0 | 0.9286 | 4.76% | 1.0000 | 28.57% |
| 1 | 1.0192 | 41.03% | 1.0000 | 30.77% |
| 2 | 1.0208 | 33.33% | 1.0000 | 30.56% |

The original high-weight-share range was 36.26 percentage points. Cluster 0 had
only one globally high-weight training sample (chemical system Cl-K-La-Zn),
whereas clusters 1 and 2 contained 16 and 12. Within-cluster ranking changed all
three cluster mean weights to exactly 1.0 and reduced the high-weight-share range
to 2.20 points. Every cluster contains high/middle/low weights.

**Mechanism conclusion:** composition-dependent global-weight imbalance exists
at an actionable level, and M2-DB balancing itself unquestionably worked.

The fitted scaler, standardized centers, and memberships are retained in the
project-local ignored runtime directory as `composition_clustering.npz` and
`cluster_membership.csv`.

## 4. Frozen training configuration and outcome

M2-DB was initialized from the same official dft_mag_density pretrained
MatterGen checkpoint as original M2, not from the M2 adapter checkpoint.

| Field | Value |
|---|---:|
| Adapter placement | interaction blocks 1 and 2 |
| Adapter shape | 512 → 64 → 512 |
| Trainable parameters | 134,272 |
| Steps / batch size | 200 / 16 |
| Learning rate | 0.0003 |
| Replay / output-anchor lambda | 50% / 0.05 |
| Weights | 1.25 / 1.0 / 0.75 within cluster |
| Elapsed / peak CUDA memory | 17.30 s / 1199.65 MiB |
| First / last 20 train-loss mean | 0.27776 / 0.28508 |
| Validation loss step 1 / 200 | 0.31775 / 0.32514 |
| Best recorded validation loss | 0.23505 |
| Checkpoint SHA-256 | `38fffcb25e577afa897ed437875ec0c59740c16ab2ead43dfcad35dd7ba40779` |

Training was numerically stable: all losses were finite, zero initialization
recovered the baseline, all 12 adapter parameter tensors received gradients,
no frozen parameter received a gradient, all frozen parameters retained the same
digest, and all 12 saved adapter tensors are nonzero. The curve is not monotonic,
but there is no NaN/Inf or training-integrity failure.

## 5. New 16-seed real-generation results

| Method | E-hull ↓ | Stable ↑ | NUS ↑ | Novel ↑ | Unique ↑ | RMSD mean ↓ | Atomic force mean ↓ | Structure max-force mean ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.09787 | 50.00% | 43.75% | 87.50% | 100% | 0.03785 | 0.13503 | 0.24069 |
| M1 | **0.08625** | **62.50%** | **50.00%** | 81.25% | 100% | 0.06194 | 0.16469 | 0.39083 |
| M2 | 0.13310 | 43.75% | 18.75% | 68.75% | 100% | **0.03347** | 0.15553 | 0.28154 |
| M2-DB | 0.11282 | 50.00% | 31.25% | 75.00% | 100% | 0.15024 | **0.12746** | 0.25537 |

E-hull is in eV/atom; RMSD is in Å; forces are in eV/Å.

M2-DB versus C0:

- E-hull is 0.01495 eV/atom higher (15.3% worse).
- Atomic force mean is 5.6% lower, but structure max-force mean is 6.1% higher.
- Stable is unchanged; NUS and Novel are each 12.5 percentage points lower.
- Therefore the required clear quality gain over C0 is absent.

M2-DB versus original M2:

- E-hull is 0.02028 eV/atom lower and atomic/structure-max force means are
  18.0%/9.3% lower.
- Novel rises only from 11/16 to 12/16 (+6.25 points), and NUS rises from 3/16
  to 5/16 (+12.5 points).
- Composition families fall from 16 to 15 and unique elements fall from 18 to
  17, so composition coverage is not restored.
- Mean RMSD is much worse because of a severe tail.

M2-DB versus M1:

- E-hull is 0.02657 eV/atom higher, Stable is 12.5 points lower, NUS is 18.75
  points lower, and Novel is 6.25 points lower.
- Force means are lower, but this does not compensate for quality/distribution
  and relaxation-tail regressions.

## 6. Force, RMSD, and relaxation tails

| Method | Atomic force P95 ↓ | Structure max-force P95 ↓ | Force max ↓ | RMSD median ↓ | RMSD P95 ↓ | RMSD max ↓ | Relax steps mean / P95 / max ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.3625 | **0.5297** | **0.7135** | 0.02967 | 0.08016 | 0.08650 | 41.25 / 84.75 / 93 |
| M1 | 0.4241 | 1.0356 | 2.3290 | 0.02790 | 0.27429 | 0.32709 | 51.50 / 134.25 / 162 |
| M2 | 0.5063 | 0.8669 | 0.9914 | 0.03214 | **0.07261** | **0.07480** | **33.25 / 62.00 / 62** |
| M2-DB | **0.3240** | 0.7067 | 1.2915 | **0.02200** | 0.59323 | 1.33742 | 58.13 / 212.25 / 213 |

M2-DB has favorable force central tendency and atomic-force P95, but its absolute
force maximum, RMSD P95/max, and relaxation P95/max are poor. Seed 73012
(Eu5NiN4) has RMSD 1.3374 Å; seeds 73003 and 73015 require 212 and 213 relaxation
steps. Thus the mean regression is not merely a reporting artifact.

## 7. Generated-distribution coverage

| Method | Composition families | Unique reduced formulas | Unique elements | num_atoms mean / median / min–max |
|---|---:|---:|---:|---:|
| C0 | 16 | 16 | 19 | 11.3125 / 12 / 4–20 |
| M1 | 15 | 16 | **21** | 11.3125 / 12 / 4–20 |
| M2 | **16** | 16 | 18 | 11.3125 / 12 / 4–20 |
| M2-DB | 15 | 16 | 17 | 11.3125 / 12 / 4–20 |

All methods have the same seed-controlled atom-count distribution:
4:2, 5:1, 8:2, 10:2, 12:4, 14:1, 16:2, 18:1, 20:1. All 16 formulas are unique
within every method. M2-DB does not restore family coverage toward C0, and its
element coverage is lower than C0, M1, and M2.

## 8. Paired 20,000-bootstrap sensitivity statistics

Each interval is a 95% paired bootstrap CI for `M2-DB minus reference`.

| Reference | Metric | Mean delta | 95% CI | Interpretation |
|---|---|---:|---:|---|
| C0 | E-hull | +0.01495 | [-0.04512, +0.08662] | no credible improvement |
| C0 | Novel | -0.1250 | [-0.3750, +0.1250] | recovery absent/inconclusive |
| C0 | RMSD | +0.11239 | [-0.01020, +0.29986] | adverse tail-driven direction |
| C0 | structure force mean | +0.01804 | [-0.05412, +0.11068] | no clear benefit |
| C0 | max-force mean | +0.01468 | [-0.13378, +0.20031] | no clear benefit |
| M1 | E-hull | +0.02657 | [-0.04014, +0.10906] | no added E-hull value |
| M1 | Novel | -0.0625 | [-0.3125, +0.1875] | no distribution advantage |
| M1 | structure force mean | -0.02713 | [-0.12857, +0.06159] | favorable but inconclusive |
| M1 | max-force mean | -0.13546 | [-0.48472, +0.11830] | favorable but inconclusive |
| M2 | E-hull | -0.02028 | [-0.08944, +0.05654] | favorable but inconclusive |
| M2 | Novel | +0.0625 | [-0.1250, +0.2500] | only one-sample recovery |
| M2 | RMSD | +0.11677 | **[+0.00206, +0.29607]** | credibly worse |
| M2 | structure force mean | -0.01239 | [-0.10278, +0.08549] | favorable but inconclusive |
| M2 | max-force mean | -0.02617 | [-0.18261, +0.16440] | favorable but inconclusive |
| M2 | relaxation steps | +24.875 | **[+1.125, +54.189]** | credibly worse |

The full Stable/NUS/Unique and pair-count results are in
`paired_statistics.csv`. These screen16 intervals are exploratory and do not
replace an independent 32-seed confirmation; the latter was not authorized by
the GO rule.

## 9. Final scientific answer

1. Global M2 quality weights were composition-cluster imbalanced.
2. Within-cluster 30/40/30 ranking corrected that imbalance almost exactly.
3. Correcting the training-weight imbalance did **not** restore generation
   coverage while retaining a clear C0-relative E-hull/force advantage.
4. Therefore global quality ranking is not a sufficient explanation for the
   quality–distribution contraction.
5. Final status is **FAIL**, not BORDERLINE: no 32-seed run, no P2, and no
   additional K/algorithm/loss sweep.
6. The Quality Adapter route has now accumulated P0, P1, P1b, and M2-DB evidence
   and should stop. The next recommended research route is **Global Transformer
   Adapter**, subject to a separate explicit experiment request.
