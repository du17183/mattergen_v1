# Innovation 1 method sources

The deployable thesis method is the frozen Fixed-K2, shared-prefix branching
pipeline with an exact C0 fallback. `phase_b_sampler.py`,
`field_decoupled_cfg.py` (in the repository's MatterGen package), and the C1
confirmation sources preserve the implementation used by the experiments.

These files are archival execution sources. Some orchestration scripts retain
machine-specific paths because changing them after confirmation could alter the
execution contract. Portable reproduction of the published tables and figures
is provided in `thesis_release/scripts/` and requires no GPU inference.

The learned Linear-K2 allocator is not the final method. Its reconstruction
source and evidence are kept under `../negative_results/`.
