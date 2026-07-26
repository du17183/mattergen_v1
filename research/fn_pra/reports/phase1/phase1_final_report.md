# FN-PRA Phase-1 final report

## Decision

- `PHASE1_ENGINEERING_GO=False`
- `PHASE1_SCIENTIFIC_GO=False`
- `FN_PRA_PHASE1_NO_GO=True`

## A0 versus P1

| Metric | A0 | P1 | Difference |
|---|---:|---:|---:|
| Median generation time (s) | 145.9635 | 143.3818 | -1.769% |
| Composition validity | 0.9375 | 0.8750 | -6.250 pp |
| Structure validity | 1.0000 | 1.0000 | +0.000 pp |
| Mean E-hull (eV/atom) | 0.143783 | 0.147569 | +0.003786 |
| Stable fraction | 0.4375 | 0.3750 | -6.250 pp |
| NUS fraction | 0.1562 | 0.2188 | +6.250 pp |
| Mean relaxation RMSD | 0.086858 | 0.061947 | -0.024910 |

All stability quantities are MatterSim-5M surrogate results with TRI2024
correction. No DFT verification or direct magnetic-property verification was
performed in Phase-1.
