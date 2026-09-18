# Thesis release inventory

Release branch: `release/thesis-final-2026`

Base commit: `1af301c5e534ff6b9b9f96ea5dab48036d9c46af`

This inventory records selective integration into the thesis release. Historical
worktrees remain untouched. `UNTRACKED_SOURCE` means that the experiment output
was complete on disk but had not previously been committed; it does not mean
that the source or result is missing.

| Source worktree | Source branch | Source commit | Source file or group | Target | Reason | Status |
|---|---|---|---|---|---|---|
| `mattergen_v1_field_cfg` | `experiment/field-decoupled-adaptive-cfg` | `fca6b8d90e869c8c9535d7e5910e606efd722fdf` | `mattergen/diffusion/sampling/field_decoupled_cfg.py` | same path | Frozen field schedules, shared-prefix cloning, RNG/state preservation | INTEGRATED |
| `mattergen_v1` | `experiment/raab-sc-p0` | `1af301c5e534ff6b9b9f96ea5dab48036d9c46af` | `experiments/reference_preserved_budgeted_cfg/*.py` | `thesis_release/innovation1/method/` | Phase-B branching, features, oracle audit | INTEGRATED_FROM_UNTRACKED_SOURCE |
| `mattergen_v1_linear_k2_confirm` | `experiment/frozen-linear-k2-confirmation` | `1af301c5e534ff6b9b9f96ea5dab48036d9c46af` | `experiments/frozen_linear_k2_confirmation/*.py` | `thesis_release/innovation1/method/` | Frozen C1 sampler and evaluation pipeline | INTEGRATED_FROM_UNTRACKED_SOURCE |
| `mattergen_v1_linear_k2_confirm` | `experiment/frozen-linear-k2-confirmation` | `1af301c5e534ff6b9b9f96ea5dab48036d9c46af` | reconstructed allocator and audit | `thesis_release/innovation1/negative_results/` | Exact Phase-B reconstruction and negative confirmation provenance | INTEGRATED_FROM_UNTRACKED_SOURCE |
| `mattergen_v1_matersim_guidance` | `experiment/mattersim-late-force-guidance-p0` | `1af301c5e534ff6b9b9f96ea5dab48036d9c46af` | P0 late-force sampler and finalization sources | `thesis_release/innovation2/method/` | Frozen RC-NFGD implementation and ablations | INTEGRATED_FROM_UNTRACKED_SOURCE |
| `mattergen_v1_linear_k2_confirm` | `experiment/frozen-linear-k2-confirmation` | `1af301c5e534ff6b9b9f96ea5dab48036d9c46af` | C1 summaries, paired rows, protocol, 256 registered seeds | `thesis_release/innovation1/{results,seeds,configs}/` | Confirmatory source data without generated structures | INTEGRATED |
| `mattergen_v1` | `experiment/raab-sc-p0` | `1af301c5e534ff6b9b9f96ea5dab48036d9c46af` | Phase-B and branch-oracle summaries | `thesis_release/innovation1/{results,negative_results}/` | Historical mechanism and allocator evidence | INTEGRATED |
| `mattergen_v1_matersim_guidance` | `experiment/mattersim-late-force-guidance-p0` | `1af301c5e534ff6b9b9f96ea5dab48036d9c46af` | P0, Formal32, Formal256, CHGNet and ablation summaries | `thesis_release/innovation2/{results,seeds,configs}/` | Formal evidence without model weights or raw structures | INTEGRATED |

## Deliberately excluded from GitHub

- MatterGen, MatterSim and CHGNet weights.
- `*.pt`, `*.ckpt`, `*.safetensors`, LMDB files and environment directories.
- Raw generated structures, relaxation caches, generation directories and logs.
- Python caches, Matplotlib caches and temporary runtime files.
- Full third-party datasets already governed by their own release/licence terms.

The original data remain in their source worktrees; this release is a compact,
auditable thesis evidence package rather than a duplicate of every experiment.
