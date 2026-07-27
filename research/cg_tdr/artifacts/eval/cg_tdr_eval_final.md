# CG-TDR Phase 0 final evaluation

## Final decision

```text
CG_TDR_GATE_V2_VALID=True
CG_TDR_V2_EIGHT_SEED_GO=False
CG_TDR_MVP_GO=False
CG_TDR_MVP_NO_GO=True
CG_TDR_ROUTE_STOPPED=True
THIRTY_TWO_SEED_STARTED=False
SIXTY_FOUR_SEED_STARTED=False
FORMAL_SEEDS_STARTED=False
```

The utility-calibrated Gate V2 fixed the engineering defect (near-always-on gates), but no candidate reached a frozen positive quality threshold. The only permitted V2 repair is exhausted, so the CG-TDR route stops at eight seeds.

## Test attribution

- main: 148 passed, 11 failed
- feature/cg-tdr: 155 passed, 11 failed
- Failure node IDs and exception types are identical.
- `CG_TDR_INTRODUCED_TEST_FAILURES=0`
- CG-TDR/V2 targeted tests: 24/24 passed.

## V1 residual diagnosis

- Position loss: zero 0.00057390842, V1 0.0011953564 (-108.28% improvement).
- Cell loss: zero 5.2499037e-05, V1 0.0011179028 (-2029.38% improvement).
- Position cosine mean/median: -0.0444/-0.0463.
- Position/cell gate >0.9: 100.0%/100.0%.

## V1 eight-seed result

| Method | E-hull | Stable | NUS | RMSD mean | RMSD median | Max force mean |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 0.150689 | 50.0% | 25.0% | 0.048643 | 0.028157 | 0.259136 |
| T1 | 0.150614 | 50.0% | 25.0% | 0.051845 | 0.033309 | 0.261436 |
| T2 | 0.149849 | 50.0% | 25.0% | 0.053918 | 0.037379 | 0.311018 |

V1 was neither safe nor positive: T1 worsened median RMSD by 18.30%; T2 worsened mean RMSD by 10.84% and mean maximum force by 20.02%.

## Gate V2 repair

- Train high-confidence target rate: 28.39%
- Train zero/low target rate: 52.08%
- Best step: 1100 of 1500; seed 3101
- Test gate--utility Spearman: position 0.751, cell 0.758
- Inference gate std: position 0.195, cell 0.125

## V2 eight-seed result

| Method | E-hull | Stable | NUS | RMSD mean | RMSD median | Max force mean | Max force median |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 0.150689 | 50.0% | 25.0% | 0.048643 | 0.028157 | 0.259136 | 0.184432 |
| V2P | 0.150672 | 50.0% | 25.0% | 0.049124 | 0.027917 | 0.259570 | 0.183429 |
| V2C | 0.150344 | 50.0% | 25.0% | 0.053052 | 0.033307 | 0.261259 | 0.193126 |

V2P is quality-safe but effectively flat and misses every positive threshold. V2C remains unsafe due to +18.29% median RMSD. Stable, NUS, composition validity, and structure validity are unchanged for both.

## Limitations and next action

- Eight-seed development screen only.
- MatterSim-5M results are surrogate evaluation, not DFT proof.
- The one allowed Gate V2 repair has been used; no V3 is permitted.
- Stop CG-TDR and move to a different second-innovation candidate.
