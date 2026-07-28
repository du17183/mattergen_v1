# A0 + E3-G training-test leakage diagnostic

## Status

`LEAKAGE_INFLATION_DETECTED`

This is an intentionally contaminated diagnostic. Seeds `20000–20063` were
used to train the frozen Q3 gate. The mixed 256 result is not an independent
validation and must not be used as a formal thesis result.

## Primary pre-relaxation force effect

| Cohort | N | Gate rate | A0 mean | A0+E3-G mean | Relative change | Wins/Ties/Losses |
|---|---:|---:|---:|---:|---:|---:|
| Train overlap | 64 | 67.188% | 0.249489 | 0.179262 | -28.148% | 43/21/0 |
| Held out | 192 | 66.146% | 0.359698 | 0.261446 | -27.315% | 96/65/31 |
| Mixed (contaminated) | 256 | 66.406% | 0.332146 | 0.240900 | -27.472% | 139/86/31 |

Train-minus-heldout mean force-effect gap: `0.028026` eV/Å,
bootstrap 95% CI `-0.026390, 0.083697`.

Training-overlap harm rate: `0.000%`; held-out harm
rate: `16.146%`. Their difference is `-16.146%`,
bootstrap 95% CI `-21.354%, -10.938%`, one-sided
Fisher exact `p=6.8659e-05`.

## Interpretation

- A negative gap means the measured force improvement is stronger on samples
  seen by the gate during training.
- Although the mean relative force reductions are similar, the training
  cohort has zero harmful gate decisions while the held-out cohort has 31.
  This is statistically significant safety/per-sample leakage inflation.
- The held-out 192 cohort is the only scientifically informative cohort in
  this run, although it still reuses an already selected seed pool and is not
  a replacement for a prospectively frozen independent validation.
- A0 generation and all 256 A0 relaxations were reused. Only gate-on A0
  structures received new MatterSim single-point force probes.
- Stability is a MatterSim-5M surrogate; DFT and target-property verification
  were not performed.
