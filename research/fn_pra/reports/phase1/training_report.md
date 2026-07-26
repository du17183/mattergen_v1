# FN-PRA Phase-1 training report

Generated: 2026-07-25T22:31:42+08:00

## Outcome

- V1-S completed: `1000` steps, passed=`True`.
- V1-D completed: `5000` steps, passed=`True`.
- Best validation loss: `0.297920555` at logged step `2749`.
- Selected P1 checkpoint: `/data/dxl/results/fn_pra/phase1/training/v1_decision/checkpoints/best-step=003000-loss_val=0.297921.ckpt`.
- Frozen official backbone mismatch count: `0`.
- Teacher and projection heads are training-only; inference uses only the low-rank adapter.

## Smoke signal

- Total training loss: `1.278742` → `0.602018`.
- Diffusion training loss: `0.452679` → `0.354452`.
- Alignment training loss: `8.260629` → `2.475656`.

## Decision signal

- Validation loss: `0.327960` → `0.326240`.
- Minimum validation loss: `0.297921`.
- All summarized losses finite: `True`.
- Training engineering gate: `PASS`.

Generation quality has not yet been evaluated; no scientific Go/No-Go is claimed here.
