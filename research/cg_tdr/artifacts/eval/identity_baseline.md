# CG-TDR V1 zero-output identity baseline

- Frozen split: seeds 30448--30511 (64 structures)
- Checkpoint: `/data/dxl/results/cg_tdr/phase0/training/checkpoints/best.pt` (strictly verified step 100)
- Loss definition: the same Smooth-L1 residual losses used in V1 training

| Field | Zero output | V1 model | Teacher oracle | V1 improvement vs zero |
|---|---:|---:|---:|---:|
| Position | 0.0005739084239 | 0.001195356408 | 0 | -108.28% |
| Cell | 5.24990367e-05 | 0.001117902781 | 0 | -2029.38% |

The Teacher oracle row is the target residual evaluated against itself. This report does not use MatterSim and does not modify labels or checkpoints.
