# CG-TDR V1 eight-seed report

- Checkpoint: `best.pt`, strictly verified at step 100
- Seeds: 23000--23007, paired A0/T1/T2
- Independent evaluator: MatterSim-5M

## Summary

| Method | E-hull mean | Stable | NUS | RMSD mean | RMSD median | Max force mean | Max force median | Median time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 0.150689 | 50.000% | 25.000% | 0.048643 | 0.028157 | 0.259136 | 0.184432 | 73.10s |
| T1 | 0.150614 | 50.000% | 25.000% | 0.051845 | 0.033309 | 0.261436 | 0.177223 | 69.20s |
| T2 | 0.149849 | 50.000% | 25.000% | 0.053918 | 0.037379 | 0.311018 | 0.223248 | 68.94s |

## Frozen decisions

- `CG_TDR_V1_EIGHT_SEED_SAFE=False`
- `CG_TDR_V1_EIGHT_SEED_POSITIVE=False`
- `GATE_SELECTIVITY_VALID=False`
- `CG_TDR_V1_DIRECT_GO=False`
- `CG_TDR_GATE_V2_REQUIRED=True`
- Selected V1 diagnostic candidate: `T1`

V1 is evaluated as a near-always-on terminal refiner. Gate V2 is required whenever gate selectivity is invalid, even if a V1 quality metric improves. No 32-seed task is launched by this analysis.
