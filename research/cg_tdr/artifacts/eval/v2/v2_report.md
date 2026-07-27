# CG-TDR Gate V2 eight-seed report

- Same paired seeds: 23000--23007
- A0 and V1/T1 results were reused; successful tasks were not rerun.
- Independent evaluator: MatterSim-5M

| Method | E-hull mean | Stable | NUS | RMSD mean | RMSD median | Max force mean | Max force median | Median time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 0.150689 | 50.000% | 25.000% | 0.048643 | 0.028157 | 0.259136 | 0.184432 | 73.10s |
| T1 | 0.150614 | 50.000% | 25.000% | 0.051845 | 0.033309 | 0.261436 | 0.177223 | 69.20s |
| V2P | 0.150672 | 50.000% | 25.000% | 0.049124 | 0.027917 | 0.259570 | 0.183429 | 69.40s |
| V2C | 0.150344 | 50.000% | 25.000% | 0.053052 | 0.033307 | 0.261259 | 0.193126 | 68.59s |

## Decision

- `CG_TDR_GATE_V2_VALID=True`
- `CG_TDR_V2_EIGHT_SEED_GO=False`
- `CG_TDR_ROUTE_STOPPED=True`
- Selected V2 candidate: `None`

No Gate V2 candidate passed every frozen safety, positive, and selectivity gate. The CG-TDR route is stopped and no 32-seed task is permitted.
