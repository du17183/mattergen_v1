# Seed and Random Tape Audit

## Deterministic seed audit

- Same-method three-repeat deterministic: `True`
- A0/G1 random tape match: `True`
- Different seeds have different initial states: `True`
- Different seeds have different structures: `True`
- Seed collision found: `False`
- Seed pipeline valid: `True`
- Paired random tape valid: `True`

The audit records the actual categorical and Gaussian tensors produced at every
prior, predictor, and corrector call. Trace recording performs no random draw.
State hashes are recorded after the prior and every predictor/corrector update.

## Frozen RP-QTFG G1_P75_S robustness

- Mean RMSD change: `0.025487437`
- Relative mean RMSD change: `68.28%`
- Median RMSD change: `0.000024629`
- Geometric mean RMSD ratio: `1.264176`
- Win/tie/loss: `{'wins': 4, 'ties': 0, 'losses': 4}`
- Trimmed mean change: `0.001210244`
- Remove-worst-one mean: `0.001022538`
- Leave-one-out range: `[0.0010225381756068996, 0.029143312785472896]`
- Maximum sample contribution: `96.39%`
- Bootstrap 95% CI: `[-2.030021900230016e-05, 0.07468596781933398]`
- Wilcoxon p-value: `0.4609375`
- Outlier dominated: `True`
- RP-QTFG route stopped before exploratory audit: `False`

## Serial gate

`PHASE_A_GATE_FOR_CG_TDR=True`

## Post-hoc exploratory audit

- Seeds: `22100–22115` (16 paired samples).
- Generation: `32/32`; MatterSim relaxation: `32/32`.
- Initial-state pairing: `True`.
- Mean RMSD change: `-8.01%`; median change: approximately `0`.
- RMSD win/tie/loss: `9/0/7`; Wilcoxon p-value: `0.7436`.
- RMSD bootstrap 95% CI: `[-0.01856, 0.00366]` (crosses zero).
- Pre-relax maximum-force change: `-9.08%`; wins: `11/16`.
- E-hull change: `+0.006095 eV/atom`, exceeding the `+0.002` safety bound.
- Safety gate: `False`.
- Original pre-registered Gate 1 No-Go remains unchanged: `True`.
- RP-QTFG route stopped: `True`.

The exploratory RMSD mean is itself sensitive to one sample: removing seed 22103
changes the mean delta from improvement to `+0.001112`. It therefore does not
provide robust evidence to reopen RP-QTFG. The failed E-hull safety gate independently
requires freezing the route.

## Final serial gate

All three required conditions are true: seed pipeline valid, paired random tape
valid, and RP-QTFG route stopped. CG-TDR Phase B is authorized.
