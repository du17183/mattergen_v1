# FN-PRA Phase-1

This directory contains the source, tests, and reproducibility utilities for the Phase-1 validation of Feature-Normalized Physics Representation Alignment (FN-PRA).

The operational state is stored outside Git under `/data/dxl/results/fn_pra/phase1`, reports under `/data/dxl/reports/fn_pra/phase1`, teacher cache under `/data/dxl/data/fn_pra_teacher_cache`, and logs under `/data/dxl/logs/fn_pra/phase1`.

The first resumable stage audits MP-20 and compares fixed atom-wise representations from CHGNet 0.3.0 and MatterSim-5M over a fixed 1,000-structure train-only manifest using eight GPU shards.

Formal validation, 64-seed experiments, and 30000-series seeds are explicitly outside Phase-1 and are never started by this runner.
