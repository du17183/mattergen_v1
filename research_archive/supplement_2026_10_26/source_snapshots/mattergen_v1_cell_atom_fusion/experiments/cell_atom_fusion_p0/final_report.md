# Global Cell--Atom Bidirectional Fusion P0 — FINAL

## Verdict

`CELL_ATOM_FUSION_P0=FAIL`; `FORMAL32_ELIGIBLE=False`; `NEXT=STOP`.

No ablation, Formal32, or Formal256 was run. The result does not revive or combine with any previously failed method.

## Architecture and identity

The only insertion is after GemNet interaction Block3 and before Block4. A row-lattice `L L^T` six-component metric descriptor is encoded to 512 dimensions; Block3 atom states are mean-pooled per graph for Atom→Cell, then the fused cell token is broadcast through Cell→Atom. Both scalar gates are zero-initialized and no time gate is used.

The official checkpoint was loaded before attaching the module. All three zero-init output differences—atomic numbers, position, and cell—were exactly `0` (required ≤`1e-5`). The official-state digest was unchanged after training. The fusion adds 330,370 trainable parameters to 48,760,443 official parameters, or 0.6775%.

## Frozen-backbone training

Training used the fixed 1024/128 train/validation split, batch size 16, AdamW at `1e-3`, and 1000 steps. Validation reused exactly the same pre-corrupted held-out batches every 100 steps.

| Loss | Official baseline | Best fusion, step 100 | Relative improvement |
|---|---:|---:|---:|
| Total | 0.25830071 | 0.25830894 | -0.00318% |
| Atomic | 0.11181063 | 0.11181298 | -0.00210% |
| Position | 1.39257947 | 1.39262193 | -0.00305% |
| Cell | 0.00723213 | 0.00723376 | -0.02240% |

The validation GO gate failed. At the selected checkpoint, `alpha_A2C=0.001060`, `alpha_C2A=-0.001447`, and residual/hidden RMS=0.108%; therefore `FUSION_UNUSED=True` under the near-zero-gate plus <1% residual rule. Later checkpoints used larger residuals but had worse validation.

## Fresh paired 16-seed smoke

Seeds 110000–110015 were generated in full for both B0 and C0 with target 0.2, constant CFG 2, 1000 steps, one corrector step, and Adaptive CFG off.

| Metric | B0 Official | C0 Fusion | Change |
|---|---:|---:|---:|
| Mag MAE ↓ | 0.009398 | 0.007051 | 24.98% better |
| Mag Hit ↑ | 75.00% | 68.75% | -6.25pp |
| Validity ↑ | 100.00% | 100.00% | 0pp |
| RMSD ↓ | 0.029899 Å | 0.044181 Å | 47.77% worse |
| Atomic force ↓ | 0.057635 | 0.049602 eV/Å | 13.94% better |
| MaxF ↓ | 0.106192 | 0.093132 eV/Å | 12.30% better |
| E-hull ↓ | 0.073566 | 0.041325 eV/atom | 43.83% better |
| Stable ↑ | 75.00% | 87.50% | +12.50pp |
| Novel ↑ | 37.50% | 18.75% | -18.75pp |
| Unique ↑ | 81.25% | 75.00% | -6.25pp |
| NUS ↑ | 12.50% | 6.25% | -6.25pp |
| Runtime ↓ | 83.501 s | 87.821 s | 5.17% worse |
| Peak VRAM ↓ | 277.916 MiB | 278.955 MiB | 0.37% worse |
| Score calls ↓ | 2000 | 2000 | unchanged |

The directional Mag MAE, MaxF, atomic-force, and E-hull gains are not confirmatory: their paired-bootstrap 95% CIs all include zero. Mag MAE relative improvement CI is `[-17.34%, +59.27%]`; MaxF `[-24.44%, +41.82%]`; atomic force `[-20.27%, +41.00%]`; E-hull `[-5.98%, +72.11%]`. RMSD also has a wide CI, while its point estimate, Novel, NUS, and Hit all move adversely.

No hard 16-seed fail-fast condition fired. Nevertheless, Formal32 is forbidden because the independent training gate failed, the selected fusion is unused, and the generation result is mixed with important quality/geometry regressions. These are explicit final FAIL conditions, so the favorable small-sample point estimates cannot be promoted.

## Scope

Magnetic density is the frozen CHGNet 0.3.0 surrogate (held-out MAE 0.001449, Pearson 0.9499, tau 0.01); energy/force/relaxation metrics use MatterSim-5M. `SURROGATE_PROPERTY_EVAL=True`; `DFT_VERIFIED=False`.
