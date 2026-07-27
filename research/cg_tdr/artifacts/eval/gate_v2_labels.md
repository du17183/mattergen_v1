# CG-TDR Gate V2 utility labels

- Reused all 512 frozen A0/CHGNet records; no structure or Teacher regeneration.
- Utility weights: energy 0.45, maximum force 0.45, stress 0.10.
- Component normalization and all q40/q70/q90 cutoffs use the train split only.
- Targets are forced to zero for identity/noise residuals, material worsening, or trust-radius impacts.
- MatterSim is not imported or used.

```json
{
  "train": {
    "n": 384,
    "position_target_mean": 0.2840166541664282,
    "position_high_confidence_rate_ge_0_5": 0.2838541666666667,
    "position_zero_low_rate_le_0_1": 0.5208333333333334,
    "cell_target_mean": 0.2774689356032492,
    "cell_high_confidence_rate_ge_0_5": 0.2760416666666667,
    "cell_zero_low_rate_le_0_1": 0.5286458333333334,
    "position_target_utility_spearman": 0.9134532800655436,
    "cell_target_utility_spearman": 0.8944249570636175
  },
  "validation": {
    "n": 64,
    "position_target_mean": 0.26741285455493935,
    "position_high_confidence_rate_ge_0_5": 0.28125,
    "position_zero_low_rate_le_0_1": 0.5625,
    "cell_target_mean": 0.25525542410413193,
    "cell_high_confidence_rate_ge_0_5": 0.265625,
    "cell_zero_low_rate_le_0_1": 0.578125,
    "position_target_utility_spearman": 0.96114647480012,
    "cell_target_utility_spearman": 0.9210919528788077
  },
  "test": {
    "n": 64,
    "position_target_mean": 0.18253737616140897,
    "position_high_confidence_rate_ge_0_5": 0.171875,
    "position_zero_low_rate_le_0_1": 0.671875,
    "cell_target_mean": 0.18253737616140897,
    "cell_high_confidence_rate_ge_0_5": 0.171875,
    "cell_zero_low_rate_le_0_1": 0.671875,
    "position_target_utility_spearman": 0.8594966280218483,
    "cell_target_utility_spearman": 0.8594966280218483
  }
}
```
