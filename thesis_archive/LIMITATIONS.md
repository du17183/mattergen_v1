# 科学与工程限制

- STABILITY_SOURCE=MatterSim-5M surrogate。
- DFT_VERIFIED=False；PROPERTY_TARGET_VERIFIED=False。
- training-overlap 和 Mixed 256 不得用于独立正式结论。
- 两批兼容性 64 seeds 必须同时报告。
- displacement_mean 没有逐 seed 原值，保留 NaN。
