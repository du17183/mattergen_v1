# Cell--Atom Fusion frozen-backbone training

## Outcome

The 1000-step run completed without instability, the strict identity gate passed, and every official parameter remained frozen. The validation gate did not pass: the best candidate checkpoint was step 100 with total loss `0.25830894`, versus frozen Official FT `0.25830071` (relative improvement `-0.00318%`). Position and cell relative improvements were respectively `-0.00305%` and `-0.02240%`.

This result permits only the requested 16-seed small smoke. It does not permit Formal32 by itself.

## Audit

| Item | Result |
|---|---:|
| Full denoiser atomic max diff at zero init | 0 |
| Full denoiser position max diff at zero init | 0 |
| Full denoiser cell max diff at zero init | 0 |
| Required tolerance | `1e-5` |
| Official/base parameters | 48,760,443 |
| New/trainable parameters | 330,370 |
| Trainable percentage | 0.6775% |
| Train / validation structures | 1024 / 128 |
| Batch size / steps / LR | 16 / 1000 / `1e-3` |
| Fixed validation noise | yes |
| Frozen digest unchanged | yes |
| Peak allocated training memory | 915.43 MiB |
| Training elapsed time | 67.57 s |

The normalization was fit only on 1,024 noisy cells drawn from the training partition. `signed_log1p` standardized features had maximum absolute value 3.536 in that fit set and all values were finite.

## Validation and mechanism

| Metric | Official FT | Best fusion (step 100) | Relative improvement |
|---|---:|---:|---:|
| Total | 0.25830071 | 0.25830894 | -0.00318% |
| Atomic | 0.11181063 | 0.11181298 | -0.00210% |
| Position | 1.39257947 | 1.39262193 | -0.00305% |
| Cell | 0.00723213 | 0.00723376 | -0.02240% |

At the selected checkpoint, `alpha_A2C=0.0010601`, `alpha_C2A=-0.0014470`, `delta_cell_rms=0.10306`, `delta_atom_rms=0.43968`, and residual/hidden RMS ratio is `0.001083` (0.108%). Both gates are effectively near zero and the residual is far below the preregistered 1% usage boundary, so `FUSION_UNUSED=True`. Later checkpoints briefly used larger residuals but had worse validation and cannot be substituted for the selected model.

The first backward pass behaved as predicted for two zero-initialized scalar gates: only `alpha_C2A` had a nonzero gradient; all internal modules became reachable after that scalar left zero. All 14 trainable parameter tensors changed during the full run.
