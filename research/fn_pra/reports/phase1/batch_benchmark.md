# A0 lossless multi-trajectory batch benchmark

|   batch_size |   single_gpu_samples_per_hour |   single_gpu_throughput_gain_vs_b1 |   fixed8_samples_per_hour |   fixed8_throughput_gain_vs_b1 | same_seed_final_level1   | level2_numeric_equivalence   | hard_gate_passed   |
|-------------:|------------------------------:|-----------------------------------:|--------------------------:|-------------------------------:|:-------------------------|:-----------------------------|:-------------------|
|            1 |                       55.3586 |                           0        |                   430.537 |                       0        | True                     | True                         | True               |
|            2 |                       84.7252 |                           0.530481 |                   679.735 |                       0.578807 | False                    | False                        | False              |
|            4 |                      130.548  |                           1.35823  |                   996.931 |                       1.31555  | False                    | False                        | False              |
|            8 |                      189.117  |                           2.41621  |                  1526.48  |                       2.54553  | False                    | False                        | False              |

- Each configuration used three warmups and at least five timed repeats.
- Single-GPU measured sample count was fixed at 40 per configuration.
- The fixed-eight-GPU benchmark used the same per-GPU protocol.
- Per-seed initial states and final RNG states are independent of batch membership/order:
  `True`.
- Recommended batch size: `1`.
- `BATCH_ENGINEERING_GO=False`.

Final Level-2 equivalence requires identical atom types and Cartesian
positions/cell numerically close at `rtol=1e-4, atol=1e-4`. This is deliberately
stricter than merely preserving validity or distributional quality.
