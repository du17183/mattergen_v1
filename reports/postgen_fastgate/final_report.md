# MatterGen post-generation module fast-gate: final report

## Final decision

`Q3_E3_PCR_FINAL_GO=True`

The first candidate to pass both frozen gates on 32 completely new C0 seeds is
the **Learned-Gated Equivariant Post-Generation Crystal Refiner**:

- original MatterGen backbone: frozen;
- original sampling trajectory: unchanged;
- atomic species: unchanged;
- lattice: unchanged in the MVP;
- trainable module: a 129-parameter invariant MLP gate;
- geometric update: CHGNet force-vector steps, which are rotation equivariant;
- safety: five-step maximum, 0.02 Å per-step trust radius, energy backtracking,
  short-distance validation, and exact baseline fallback.

The 32-seed comparison uses the first trajectory from each of 32 new four-way
C0 pools (seeds 33000–33127). MatterSim-5M is used only after the refiner is
frozen. The CHGNet teacher and MatterSim evaluator are independent.

## Six-candidate outcome

| Candidate | Evaluation stage | E-hull change (eV/atom) | Stable change | NUS change | RMSD change | Pre-relax max-force change | Novel change | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Q1 UQ-PQR | historical held-out | -0.07046 | +43.80 pp | +19.45 pp | -61.02% | — | -14.86 pp | No-Go |
| Q2 RFR | historical held-out | -0.07938 | +39.91 pp | +7.14 pp | -78.64% | — | -30.25 pp | No-Go |
| Q4 CPRC | historical held-out | -0.07199 | +44.44 pp | +17.44 pp | -61.64% | — | -17.08 pp | No-Go |
| Q6 NS-SetRank | new 32-pool blind test | -0.03372 | +28.13 pp | +9.38 pp | -52.95% | -45.59% | -12.50 pp | No-Go |
| Q5 CQPS | new 32-pool blind test | -0.03140 | +21.88 pp | +3.13 pp | -51.96% | -36.81% | -15.63 pp | No-Go |
| **Q3 E3-PCR** | **new 32-seed blind test** | **-0.00056** | **0.00 pp** | **0.00 pp** | **-0.01%** | **-20.45%** | **0.00 pp** | **GO** |

Q1, Q2, and Q4 were stopped at the held-out historical gate because they
violated the frozen novelty constraint. Q6 and Q5 were evaluated on new data
because their historical gates passed; both delivered large quality gains but
again violated novelty safety. Q3 changes each generated structure only
slightly instead of selecting a different trajectory and therefore preserves
all official discrete quality metrics.

## Q3 final metrics

| Metric | Original C0 | Q3 refiner | Change |
|---|---:|---:|---:|
| Generation/refinement success | 32/32 | 32/32 | 0 failures |
| Composition validity | 75.00% | 75.00% | 0.00 pp |
| Structure validity | 100.00% | 100.00% | 0.00 pp |
| Average E-hull | 0.159748 | 0.159185 | -0.000562 eV/atom |
| Stable | 28.125% | 28.125% | 0.00 pp |
| NUS | 9.375% | 9.375% | 0.00 pp |
| Novel | 71.875% | 71.875% | 0.00 pp |
| Unique | 100.00% | 100.00% | 0.00 pp |
| Relaxation RMSD | 0.071895 | 0.071887 | -0.010% |
| Pre-relax maximum force | 0.287507 | 0.228701 eV/Å | **-20.454%** |
| MatterSim force convergence | 96.875% | 100.00% | +3.125 pp |

For pre-relax maximum force, the paired mean difference is
`-0.058806 eV/Å`, bootstrap 95% CI `[-0.104730, -0.022368]`, Wilcoxon
`p=0.000328`, with 16 wins, 12 ties, and 4 losses.

## Network and trust-region details

The gate uses 14 invariant structure/CHGNet summary features, one tanh hidden
layer with eight units, and one binary output (129 trainable parameters). On the
frozen 64-structure historical dataset, eight-fold out-of-fold evaluation gave:

- AUROC: 0.6190;
- balanced accuracy: 0.5626;
- apply rate: 70.31%;
- aggregate pre-relax force change: -21.36%;
- force-worsening rate: 20.31%;
- Stable/NUS/Novel/Unique/Composition/Structure changes: all zero.

On the new blind set, the gate refined 20/32 structures. All proposed steps
passed CHGNet energy backtracking; the maximum cumulative wrapped displacement
was 0.0408 Å. Twelve structures fell back bit-for-bit to the original C0
output.

## Interpretation and limitations

This is an MVP engineering/scientific GO, not yet a formal 256-seed or DFT
confirmation. The positive result is specifically a statistically supported
reduction in MatterSim pre-relax force while the frozen official quality
metrics remain unchanged. E-hull and RMSD improvements are too small to claim
as independent benefits.

The gate's cross-validated AUROC is modest, so the thesis claim should center
on the complete learned-gate + equivariant trust-region + backtracking +
fallback system, not on classifier accuracy alone. A 64-seed frozen validation
should be the next step, followed by 256 seeds only if all quality-preservation
gates remain satisfied.

`DFT_VERIFIED=False`.
