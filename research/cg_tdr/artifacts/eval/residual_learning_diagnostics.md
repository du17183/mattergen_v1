# CG-TDR V1 residual-learning diagnostics

Checkpoint `best.pt` was strictly loaded at step 100. Diagnostics use the frozen 64-structure test split only.

## Residual direction and magnitude

| Field | Cosine mean | Cosine median | Positive cosine | Cosine > 0.5 | Magnitude ratio median |
|---|---:|---:|---:|---:|---:|
| Position | -0.044357 | -0.046256 | 41.94% | 4.84% | 1.155001 |
| Cell | 0.064582 | 0.221468 | 54.10% | 34.43% | 23.164724 |

## Gate selectivity

| Gate | Mean | Median | Std | <0.1 | <0.5 | >0.9 |
|---|---:|---:|---:|---:|---:|---:|
| Position | 0.993723 | 0.997022 | 0.008384 | 0.00% | 0.00% | 100.00% |
| Cell | 0.987731 | 0.992184 | 0.013193 | 0.00% | 0.00% | 100.00% |

`GATE_SELECTIVITY_VALID=False`. The frozen diagnostic criterion requires both gate standard deviations >= 0.05, no more than 80% of either gate above 0.9, and positive Spearman correlation with both Teacher utility and residual magnitude.

Per-structure values are stored in `/data/dxl/results/cg_tdr/phase0/residual_learning_per_structure.csv`.
