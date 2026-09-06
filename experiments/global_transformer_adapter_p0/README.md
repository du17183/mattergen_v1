# Periodic Geometry-aware Global Transformer Adapter — P0

This directory contains the minimal real-checkpoint comparison requested for the
second-innovation architecture screen:

- `C0`: frozen official MatterGen `dft_mag_density` checkpoint;
- `MLP`: one atom-local, parameter-matched residual MLP;
- `Transformer`: one full intra-crystal atom–cell–time Transformer with periodic
  scalar distance bias.

The frozen result is **FAIL** for this Transformer version. Although it improved
NUS/Novel and several force statistics over the matched MLP, it was worse in
mean E-hull and showed a large RMSD/relaxation regression, including one
884-step relaxation outlier. No P1 run was started.

## Frozen protocol

- Branch: `experiment/global-transformer-adapter-p0`
- Data: official MatterGen Alex-MP `alex_mp_20/train.csv`, reservoir-selected
  1024 train + 128 validation structures (selection seed `20260907`).
- Training: frozen base, original mixed-field diffusion objective, AdamW,
  batch 16, LR `1e-4`, 1000 steps, identical budget for both adapters.
- Generation: paired seeds `75000–75007`, `dft_mag_density=0.1`, constant
  original CFG 2.0, original Predictor and Corrector, 1000 diffusion steps.
- Evaluation: MatterSim-5M GPU relaxation (`fmax=0.05`) and the project formal
  Alex-MP metric pipeline. All values have `DFT_VERIFIED=False`.

## Reproduction order

```bash
cd /mnt/lis-wam-data/dxl/mattergen_v1
source .venv/bin/activate

python experiments/global_transformer_adapter_p0/train_adapters.py
CUDA_VISIBLE_DEVICES=0 python experiments/global_transformer_adapter_p0/offline_validation.py
python experiments/global_transformer_adapter_p0/run_generation.py
python experiments/global_transformer_adapter_p0/aggregate_generation.py
python experiments/global_transformer_adapter_p0/run_relaxation.py
python experiments/global_transformer_adapter_p0/run_quality_metrics.py
python experiments/global_transformer_adapter_p0/analyze_results.py
```

The main compact outputs are `training_summary.csv`, `validation_summary.csv`,
`generation_results.csv`, `quality_results.csv`, and `final_report.md`. Runtime
checkpoints, structures, logs, reference caches, and relaxation artifacts remain
project-local and are intentionally git-ignored.
