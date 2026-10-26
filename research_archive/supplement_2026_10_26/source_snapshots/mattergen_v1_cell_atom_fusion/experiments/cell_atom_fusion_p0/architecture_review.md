# Global Cell–Atom Bidirectional Fusion — Architecture Review

## Frozen baseline

- Branch: `experiment/cell-atom-fusion-p0`; base HEAD `1af301c5e534ff6b9b9f96ea5dab48036d9c46af`.
- Official `dft_mag_density` Full FT checkpoint SHA-256: `01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e`.
- `GemNetTAdapter → GemNetTCtrl`, with `num_blocks=4`, `emb_size_atom=512`, `emb_size_edge=512`, `latent_dim=512`, and stress/cell regression enabled.

## Tensor and conditioning audit

For batch size `B`, total atoms `M`, and periodic edges `E`:

| Tensor | Shape | Role |
|---|---:|---|
| `frac_coords` | `[M,3]` | fractional atomic positions |
| `distorted_lattice` | `[B,3,3]` | current noisy cell, row lattice vectors |
| `batch` | `[M]` | atom-to-structure graph id |
| Block3 output `h` | `[M,512]` | atom/node hidden stream |
| Block3 output `m` | `[E,512]` | edge hidden stream |
| time embedding `z` | `[B,512]` | injected into initial atom hidden |
| property-adapt embedding | `[B,512]` | broadcast and injected before every block |

The model has no persistent native `[B,512]` cell token. Cell information currently enters implicitly through Cartesian positions, periodic graph geometry, distances, and lattice cosines; cell output is the accumulated five-head stress/lattice score. The proposed module therefore adds an explicit global cell token—it is not the first cell–atom interaction in MatterGen.

## Fixed insertion point

The zero-based loop maps `i=0/1/2/3` to `IntBlock_1/2/3/4`. The only P0 insertion is:

```text
IntBlock_3 returns h:[M,512], m:[E,512]
→ CellAtomBidirectionalFusion updates h only
→ OutBlock_3 / lattice head 3
→ original property residual before IntBlock_4
→ IntBlock_4
→ remaining heads
```

The hook is directly after the `IntBlock_3` call and before its output heads, hence strictly before Block4. Earlier output heads remain unchanged; the fusion affects the final atom logits directly through final `h` and affects continuous position/cell outputs through the Block3/4 downstream edge/head path. This asymmetry is a limitation and will not be hidden.

## Cell representation and normalization

MatterGen stores lattice vectors as rows and uses `cart = frac @ L`. Thus the rotation-invariant basis Gram under `L→LR` is `G=L Lᵀ`, even though the task statement wrote `LᵀL`. The current symmetric lattice gauge makes them nearly equal, but the implementation follows the repository convention.

The six features are `(Gxx,Gyy,Gzz,Gxy,Gxz,Gyz)`, transformed componentwise by `signed_log1p`, then standardized with training-only frozen mean/std. BatchNorm and validation statistics are prohibited. This is invariant to global Cartesian rotation, not to arbitrary unimodular basis changes.

## Bidirectional module

```text
cell6 → Linear(6,128) → SiLU → Linear(128,512) = h_cell
mean_pool(Block3 h, graph) = h_atom_global
h_cell_fused = h_cell + alpha_A2C · MLP_512→128→512(h_atom_global)
delta_atom = MLP_512→128→512(h_cell_fused)
h_new = h + alpha_C2A · delta_atom[batch]
```

`alpha_A2C` and `alpha_C2A` are scalar trainable parameters initialized exactly to zero, as required. Directional MLPs use ordinary nonzero initialization. Zeroing both scalars and residual output projections would make the entire path permanently gradient-dead. With the selected design, `alpha_C2A` can receive gradient on backward 1; internal C2A/cell parameters wake after it moves, and A2C wakes later through the nested path. Training explicitly audits this staged gradient reachability. No timestep gate is used because Block3 already contains time conditioning and a second zero gate would create avoidable dead-gradient risk.

## Parameter and checkpoint compatibility

| Component | Parameters |
|---|---:|
| Cell encoder `6→128→512` | 66,944 |
| A2C `512→128→512` | 131,712 |
| C2A `512→128→512` | 131,712 |
| two scalar alphas | 2 |
| **Total new/trainable** | **330,370** |

Against the official 48,760,443 parameters this is approximately 0.678%, below 1M and far below the 5% cap.

The official checkpoint is loaded strictly first. Only then is `cell_atom_fusion` dynamically attached, all official parameters frozen, and the fusion-only state saved separately with base SHA, normalization, config, identity audit, and training history. The official checkpoint is never overwritten.

## Identity and mechanism gates

1. Module-level zero-alpha output must be bitwise identical to input `h`.
2. Full denoiser compares disabled versus enabled zero-init on the exact same already-corrupted batch and timestep; all atomic/position/cell max-absolute differences must be ≤`1e-5`.
3. Backward 1 must give finite nonzero `alpha_C2A` gradient; subsequent steps must show C2A/cell and finally A2C gradient activation. Frozen parameters must never receive gradients and their digest must remain unchanged.
4. Record both alphas, directional delta norms, residual/hidden ratio, field validation losses, score calls, runtime, and VRAM.

No Adaptive CFG, LoRA, full-backbone training, attention pooling, Transformer, or alternate insertion-point sweep is used.
