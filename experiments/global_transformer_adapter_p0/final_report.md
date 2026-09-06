# Periodic Geometry-aware Global Transformer Adapter — P0 final report

## Conclusion

**Verdict: FAIL. Do not freeze this Transformer configuration for a 32-seed P1.**

The test cleanly answers the central control question. A matched local MLP gave
the stronger overall P0 result. The global Transformer had real positive signals
in NUS, novelty, and several force statistics, but did not provide a credible
net gain over the MLP: mean E-hull was worse, mean/median RMSD were worse, and
relaxation cost regressed sharply. Transformer seed 75003 (`Gd4Pd2O9`) required
884 MatterSim steps and moved by 0.5845 Å RMSD. The regression is not solely a
mean distorted by that point: excluding seed 75003 from both methods,
Transformer mean RMSD is 0.0521 Å versus MLP's 0.0317 Å.

This is an eight-seed P0 and is not a significance claim. It is a frozen
screening decision under the predeclared failure rule for RMSD/relaxation
breakdown. No P1, geometry-bias ablation, or architecture sweep was run.

## 1. Branch and code state

- Branch: `experiment/global-transformer-adapter-p0`
- Parent commit: `d8f8014404df6280523cf527533da79c10c14ead`
- Final HEAD is reported in the handoff; a commit cannot embed its own hash in
  its committed contents.
- Project, environment, weights, caches, data, and runtime results all remain
  under `/mnt/lis-wam-data/dxl/mattergen_v1`; no root-directory environment or
  model cache was created.

## 2. Real MatterGen structure and insertion point

The loaded checkpoint is `GemNetTAdapter -> GemNetTCtrl`. After atom embedding
and the existing timestep merge, `h` is `[N_total_atoms, 512]`; `batch` is a
length-`N_total_atoms` crystal index; the timestep embedding `z` is `[B, 512]`.
The network has four interaction blocks. Position and lattice contributions are
accumulated after every block.

The new residual hook is called inside the real `GemNetTCtrl.forward` loop and
acts only at zero-based `block_index=1`: after interaction block 1 and its
existing output heads, before interaction block 2. Its modified `h` therefore
passes through the final two GemNet interaction blocks and their geometry heads.
The official per-block `dft_mag_density` adapter is unchanged.

## 3. Transformer design

- Tokens: atom, cell, and timestep. No separate condition token was used because
  extracting it would intrude on the official condition-adapter mechanism.
- Atom token: `LayerNorm(512) -> Linear(512,256)` from the real intermediate `h`.
- Cell token: MatterGen uses row lattice vectors (`cart = frac @ lattice`), so
  the rotation-invariant Gram matrix is `L @ L.T`. Six independent Gram entries
  use signed `log1p`, followed by `log1p(volume)`, `log1p(volume/atom)`, and
  `log1p(num_atoms)`. The nine scalars pass through `9 -> 128 -> 256`.
- Timestep token: the existing 512-dimensional `z`, projected by
  `LayerNorm(512) -> Linear(512,256)`.
- Attention: one Pre-LN block, full attention between every valid atom in the
  same crystal plus its cell/time tokens; padded atoms and different crystals
  are masked. This is not restricted to GemNet neighbor edges.
- Periodic bias: the exact `edge_index` and periodic distances `D_st` produced by
  MatterGen's interaction graph are reused. Multiple periodic images reduce by
  minimum distance. Thirty-two Gaussian RBFs over 0–7 Å map to one scalar bias
  per head. Atom pairs outside the local cutoff remain globally connected and
  receive the saturated 7 Å bias. No Cartesian direction is fed to scalar
  attention.
- Configuration: `d_model=256`, 4 heads, 1 layer, FFN 768, cutoff 7 Å, 32 RBFs.
- Output: `Linear(256,512)` is exactly zero-initialized and added residually to
  `h`.
- Trainable parameters: **1,089,924**.

The H20 driver rejected batched `torch.linalg.det` on 3x3 cells. Cell volume is
therefore computed by the mathematically equivalent differentiable scalar triple
product. This was an implementation compatibility fix, not a scientific change.

## 4. Matched local MLP control

The MLP is inserted at the identical block and uses
`LayerNorm(512) -> 512x650 -> SiLU -> 650x650 -> SiLU -> 650x512`, with the final
projection zero-initialized. It has no atom–atom, cell, or timestep interaction.

- MLP trainable parameters: **1,090,936**.
- Transformer trainable parameters: **1,089,924**.
- Difference: 1,012 parameters; MLP is only **0.0929%** larger than Transformer.

Thus parameter count is effectively controlled.

## 5. Data, objective, and correctness checks

- Source: official MatterGen Alex-MP training member `alex_mp_20/train.csv` from
  the project-local `alex_mp_20.zip` archive.
- Eligible official rows: 597,339 with finite `dft_mag_density` and 1–20 sites.
- Fixed selection seed: `20260907`.
- Clean-x0 split: **1024 train / 128 validation**; both adapters use the exact
  same structures.
- Objective: original MatterGen mixed-field diffusion corruption/loss and field
  weights only. No quality/reward/novelty/composition/physics auxiliary loss.
- Base atom embedding, GemNet blocks, official condition adapter, and all output
  paths are frozen with `requires_grad=False`; the forward graph is not wrapped
  in global `no_grad`.

The real checkpoint zero-init test passed for atomic, position, and cell outputs.
GPU scatter causes small run-to-run numerical variation, but enabling the
zero-output adapter stayed within the measured repeat tolerance. A real H20
backward produced finite gradients for all 29 Transformer parameter tensors; at
step 1 only the zero-initialized output projection had nonzero gradients, as
expected, and later all trainable state entries except the fixed RBF-center
buffer changed. Frozen-parameter digests before and after both trainings matched.

## 6. Training results

Both runs used AdamW, LR `1e-4`, weight decay `1e-4`, batch 16, training seed
`20260907`, 1000 optimization steps, and validation every 100 steps.

| Method | Params | Steps | First-50 train loss | Last-50 train loss | Final online val loss | Atomic / Pos / Cell val loss | Time | Peak GPU memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP | 1,090,936 | 1000 | 0.277249 | 0.297555 | 0.290381 | 0.101900 / 1.784611 / 0.010020 | 54.27 s | 1223.85 MiB |
| Transformer | 1,089,924 | 1000 | 0.277421 | 0.297741 | 0.291242 | 0.101994 / 1.791700 / 0.010078 | 57.92 s | 1228.91 MiB |

All losses and gradients remained finite. The train loss is high-variance across
sampled timesteps, so first/last-window values are descriptive rather than a
monotonic convergence claim.

## 7. Identical-corruption offline validation

All three methods were evaluated on the 128 validation structures with the same
corruption RNG seed `20260927`.

| Method | Total | Atomic | Position | Cell | Total delta vs MLP |
|---|---:|---:|---:|---:|---:|
| C0 | 0.308678 | 0.103764 | 1.922116 | 0.012702 | -0.000904 |
| MLP | 0.309582 | 0.103621 | 1.930200 | 0.012940 | 0 |
| Transformer | 0.310570 | 0.103685 | 1.936729 | 0.013213 | +0.000989 |

Transformer was slightly worse than MLP offline, driven mainly by position loss
(`+0.006530`), but this small auxiliary result did not stop real generation.

## 8. Real generation and MatterSim evaluation

- Paired fresh seeds: **75000–75007**.
- Methods: C0, MLP, Transformer; 24 total generations.
- Generation: `dft_mag_density=0.1`, constant original CFG 2.0, original
  Predictor, original one-step Corrector, 1000 diffusion steps, batch size 1.
- Generation success: **8/8 for every method**.
- MatterSim-5M relaxation success: **8/8 for every method**.
- Relaxation used GPU, `fmax=0.05`, with the existing H20 3x3 determinant
  workaround. Convex-hull/novelty/structure matching then ran on CPU against the
  frozen Alex-MP reference.
- These are surrogate results: **DFT_VERIFIED=False**.

| Metric | C0 | Matched MLP | Global Transformer |
|---|---:|---:|---:|
| Generation success | 1.000 | 1.000 | 1.000 |
| MatterSim success | 1.000 | 1.000 | 1.000 |
| E-hull mean (eV/atom) ↓ | 0.135299 | **0.055264** | 0.072788 |
| E-hull median (eV/atom) ↓ | 0.100016 | **0.029017** | 0.065777 |
| E-hull max (eV/atom) ↓ | 0.396672 | 0.179695 | **0.169247** |
| Stable ↑ | 0.500 | **0.750** | **0.750** |
| NUS ↑ | 0.125 | 0.500 | **0.625** |
| Novel ↑ | 0.625 | 0.750 | **0.875** |
| Unique ↑ | 1.000 | 1.000 | 1.000 |
| RMSD mean (Å) ↓ | 0.036992 | **0.035512** | 0.118675 |
| RMSD median (Å) ↓ | **0.015317** | 0.023779 | 0.042823 |
| Atomic force mean (eV/Å) ↓ | 0.230852 | 0.219529 | **0.192903** |
| Structure mean-force mean (eV/Å) ↓ | **0.172637** | 0.177558 | 0.175478 |
| Structure max-force mean (eV/Å) ↓ | 0.651778 | 0.347862 | **0.313383** |
| Atomic force P95 (eV/Å) ↓ | **0.373178** | 0.709209 | 0.519581 |
| Force max (eV/Å) ↓ | 3.677327 | 0.969645 | **0.698061** |
| Relaxation steps mean ↓ | 62.375 | **43.625** | 166.750 |
| Relaxation steps max ↓ | 200 | **104** | 884 |

## 9. Signals and outliers

Relative to C0, MLP showed a coherent positive P0 signal: mean E-hull fell by
0.0800 eV/atom, Stable rose by 0.25, NUS by 0.375, mean RMSD slightly improved,
force maximum fell strongly, and mean relaxation steps fell by 18.75.

Relative to C0, Transformer also improved mean E-hull by 0.0625 eV/atom,
Stable/NUS/Novel by 0.25/0.50/0.25, atomic mean force, structure max-force mean,
and force maximum. However, atomic force P95 worsened (0.5196 vs 0.3732), mean
RMSD rose by 0.0817 Å, and mean relaxation steps rose by 104.4.

The central comparison is Transformer versus MLP:

- positive: NUS `+0.125`, Novel `+0.125`, atomic mean force `-0.0266 eV/Å`,
  structure max-force mean `-0.0345 eV/Å`, force P95 `-0.1896 eV/Å`, and force
  maximum `-0.2716 eV/Å`;
- negative: mean E-hull `+0.0175 eV/atom`, median E-hull `+0.0368 eV/atom`, mean
  RMSD `+0.0832 Å`, median RMSD `+0.0190 Å`, mean relaxation steps `+123.1`, and
  maximum relaxation steps `+780`.

The severe Transformer outlier is seed 75003 (`Gd4Pd2O9`): E-hull 0.13584
eV/atom, initial max force 0.6981 eV/Å, RMSD 0.58450 Å, and 884 relaxation
steps. Seed 75005 also has elevated RMSD 0.18673 Å and 166 steps. C0 has a
different force outlier at seed 75002 (3.6773 eV/Å), but its RMSD and relaxation
tails remain much smaller than the Transformer's.

## 10. Scientific decision

1. Structural correctness: pass.
2. Training stability and frozen-base behavior: pass.
3. Generation and MatterSim success: pass (24/24 generation, 24/24 relaxation).
4. Positive Transformer signals: present in NUS/Novel and several force metrics.
5. Global interaction beyond matched parameters: **not supported overall**.
   The parameter-matched local MLP is better on E-hull and dramatically better
   on RMSD/relaxation robustness.

Therefore this is **FAIL**, not BORDERLINE: it is not merely Transformer≈MLP;
the geometry-relaxation regression triggers the predeclared failure condition.
The result is more consistent with a useful generic PEFT/new-parameter effect
captured by the MLP than with a robust added benefit from this global attention
design.

**Single next recommendation:** stop this Transformer version and use the already
independently validated Q3 E3-PCR result as the thesis second-innovation backup,
rather than spending 32 new seeds or adding Transformer complexity.
