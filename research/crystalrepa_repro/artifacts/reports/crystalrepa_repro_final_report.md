# CrystalREPA unconditional MP-20 reproduction — final handoff

- `REPA_REPRO_ENGINEERING_GO=False`
- `REPA_REPRO_SCIENTIFIC_GO=False`
- `REPA_BASE_REPRODUCED=False`
- `REPA_REPRO_NO_GO=True`

| Metric | U0 | R1 | Change |
|---|---:|---:|---:|
| Composition validity | 0.8438 | 0.8125 | -3.125 pp |
| Structure validity | 1.0000 | 1.0000 | +0.000 pp |
| Mean E-hull (eV/atom) | 0.201931 | 0.296167 | +0.094236 |
| Metastable | 0.2188 | 0.1562 | -6.250 pp |
| Stable | 0.0000 | 0.0000 | +0.000 pp |
| Relaxation RMSD | 0.205452 | 0.238742 | +0.033290 |

E-hull coverage is U0 64/64 and R1 59/64. TRI2024 lacks terminal references for Pm, Pu, Tc, and U; the paired 59-seed E-hull mean difference is +0.092235 eV/atom (95% bootstrap CI [-0.056828, +0.345338], Wilcoxon p=0.927830).

The frozen Teacher is CHGNet 0.3.0, a controlled deviation from the paper. MatterSim-5M is used only as the independent evaluation surrogate.
