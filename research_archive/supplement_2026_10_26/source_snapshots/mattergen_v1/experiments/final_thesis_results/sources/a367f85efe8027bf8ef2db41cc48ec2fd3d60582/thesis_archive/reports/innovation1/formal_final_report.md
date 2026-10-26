# MatterGen two-innovation formal 256-seed independent validation

- FORMAL_256_COMPLETED=True
- FORMAL_PARAMETERS_RETUNED=False
- Seeds: 20000-20255; paired n=256
- C0/A0/G3 generation: 256/256 each
- C0/A0/G3 MatterSim relaxation: 256/256 valid each
- Cross-method initial state paired: True
- Determinism: Level 1

## Innovation 1: C0 vs A0

- E-hull change (A0-C0): -0.003435 eV/atom
- Stable change: +5.859%
- NUS change: +3.516%
- FORMAL_INNOVATION1_CONFIRMED=True
- ehull_degradation_le_0_02: True
- stable_decline_le_3pp: True
- NUS_decline_le_3pp: True
- at_least_two_primary_metrics_directionally_improve: True

## Innovation 2: A0 vs G3

- Physical forward reduction: 35.368%
- Corrector forward reduction: 70.736%
- Median speedup: 50.607% (1.506x)
- Fixed-concurrency throughput gain: 44.764%
- E-hull change: +0.022423 eV/atom
- Stable change: -9.766%
- NUS change: -9.375%
- Composition-validity change: -0.391%
- FORMAL_INNOVATION2_CONFIRMED=False
- physical_forward_reduction_ge_30_percent: True
- median_wall_time_speedup_ge_25_percent: True
- fixed_concurrency_throughput_gain_ge_30_percent: True
- stable_rate_decline_le_3pp: False
- NUS_rate_decline_le_3pp: False
- average_Ehull_degradation_le_0_02: False
- composition_validity_no_significant_decline: True
- basic_structure_validity_not_lower: True

## Claim boundaries

- STABILITY_SOURCE=MatterSim-5M surrogate
- DFT_VERIFIED=False
- PROPERTY_TARGET_VERIFIED=False
- No formal-test result was used to retune either method.
