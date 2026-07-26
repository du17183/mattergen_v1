# FN-PRA Teacher Probe

Generated: `2026-07-25T20:19:06+08:00`

Selected: **CHGNet 0.3.0**

Independent from MatterSim evaluator and passed atom-wise mapping/stability/environment probes.

| candidate | dim | mapping | repeat max abs | within-element variance | energy ridge MAE | coordination R² | structures/s | peak VRAM GiB | independent |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| CHGNet 0.3.0 | 64 | True | 3.58e-06 | 1.33016 | 0.236621 | 0.6709 | 1224.61 | 0.873 | True |
| MatterSim-5M | 256 | True | 1.07e-06 | 0.011638 | 0.110146 | 0.8299 | 1370.11 | 0.147 | False |

Selection is not based only on energy MAE. Mapping integrity, determinism, environment sensitivity, cost, and evaluator independence are hard considerations.
