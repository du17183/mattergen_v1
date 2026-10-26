# Robust Adaptive CFG V2 algorithm

The frozen training-free controller applies the following chain independently at every predictor/corrector score call:

1. Compute RMS conditional–unconditional residuals for atomic numbers, positions, and cell.
2. Transform each residual with phase/field/20-bin `log(r+1e-12)` calibration and clip z to ±3.0.
3. Aggregate the three standardized fields with their median.
4. Update a predictor- or corrector-specific EMA with beta=0.95.
5. Compute signal confidence outside the baseline-derived dead zone and multiply it by exponential field-agreement confidence.
6. When signal confidence is zero, target the trusted fixed anchor CFG=2.0; otherwise use `g0 + c*0.5*tanh(z_ema)`.
7. Clip to [1.5, 2.5] and apply a per-score-call slew cap of 0.05.

Stage gating is OFF. There is one global CFG shared by all fields. No learned weights, MLP, RL, gradient guidance, adapter, LoRA, or diffusion-model retraining is used. `A_NORM` is the normalization+median ablation without confidence, EMA control, or slew limiting; `A_OLD` is the original frozen V1.
