# Stage A0 reference review

## Decision

`REUSE_EXISTING_CLEAN_STRUCTURES=False`

The historical 64-seed field/site diagnostics under
`/mnt/datasets-livsyn/dxl/mattergen_v1_async_freeze/experiments/` were audited
before starting any new generation.

- `site_level_convergence_diagnostic/rerun/<seed>/site_clean_estimates.npz`
  stores all 1,000 clean position and clean cell estimates, including the
  required late times.
- It does **not** store the clean atomic-number argmax. Therefore these arrays
  are not complete `x0_hat(t)` crystals and cannot be evaluated by MatterSim
  without substituting future final atomic identities.
- `shared_field_dynamics/trajectories/<seed>/field_dynamics.csv` stores only
  aggregate atomic entropy/confidence/agreement, not per-site clean atomic
  identities. It cannot repair the omission.
- Final atomic identities will not be backfilled into historical clean
  positions, because that would leak a future state into a supposedly online
  late-stage diagnostic.

Consequently Stage A0 uses 32 new official trajectories and saves complete
clean estimates (atomic-number argmax, fractional positions, and cell) at the
four requested nonzero late times plus the actual final generated structure.

## Frozen provenance

- MatterGen Git HEAD: `1af301c5e534ff6b9b9f96ea5dab48036d9c46af`
- Branch: `experiment/mattersim-late-force-guidance-p0`
- MatterGen checkpoint:
  `/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/hf_mattergen/checkpoints/dft_mag_density/checkpoints/last.ckpt`
- MatterGen checkpoint SHA256:
  `01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e`
- MatterSim-5M checkpoint:
  `/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/mattersim/mattersim-v1.0.0-5M.pth`
- MatterSim checkpoint SHA256:
  `e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5`
- MatterSim package used by the available CUDA runtime: `1.2.3`
- CHGNet package available to the frozen property evaluator: `0.3.0`.
  CHGNet is not used to decide A0.

## Seed and GPU audit

All MatterGen worktrees under `/mnt/datasets-livsyn/dxl/mattergen_v1*` were
searched before registration. Neither `710000-710099` nor `720000-720099`
occurred in historical paths or text. Stage A0 freezes `710000-710031`.
The future A P0 namespace is disjoint from B and will be frozen only after A0
GO. GPU 2 is excluded because it was occupied; collection is limited to the
audited idle GPUs and records the physical GPU for every process.
