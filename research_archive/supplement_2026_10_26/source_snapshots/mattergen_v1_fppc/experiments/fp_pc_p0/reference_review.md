# FP-PC reference and protocol review

## Frozen baseline

- Worktree: `/mnt/datasets-livsyn/dxl/mattergen_v1_fppc`
- Branch: `experiment/fp-pc-p0`
- MatterGen HEAD: `1af301c5e534ff6b9b9f96ea5dab48036d9c46af`
- Model: official `dft_mag_density` Full FT; target `0.2 Å^-3`
- Checkpoint: `/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/hf_mattergen/checkpoints/dft_mag_density/checkpoints/last.ckpt`
- Checkpoint SHA256: `01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e`
- Constant CFG `2.0`, 1000 reverse steps, one Corrector step per time, Adaptive CFG **OFF**.
- Geometry evaluator for eventual P0: MatterSim package `1.2.3`, 5M checkpoint SHA256
  `e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5`.
- Property evaluator: frozen CHGNet `0.3.0` surrogate, checkpoint SHA256
  `d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1`
  (`MAE=0.001449`, `Pearson=0.9499`, `tau=0.01`).
  `SURROGATE_PROPERTY_EVAL=True`; `DFT_VERIFIED=False`.

## Why the old 64-trajectory asset must be replayed

The historical target-0.2 cohort `99000–99063` has 64 official deterministic
trajectories and final MatterSim outcomes.  Its read-only trace observes a clean
estimate every 20 steps at `predictor_pre_update_after_official_corrector`.  It
does **not** contain before/after tensors for both official Corrector and
Predictor proposals.  Reusing only that CSV therefore cannot answer B0 Q1/Q2.

B0 deterministically replays the same seeds, checkpoint and sampler and adds
side-effect-free measurements after every proposal.  It does not modify or
accept/reject any proposal.  The replayed final structure is checked against the
historical structure before outcomes are joined.

## Frozen feasible region

Thresholds are computed before looking at B0 outcomes from a deterministic
4096-structure reservoir sample of the official Alex-MP training split.  The
repository path is a Git-LFS pointer; the byte-identical downloaded archive is
read from `.cache/datasets/alex_mp_20.zip` (LFS object SHA256
`6688e652e45fac4ac8b016dbc4a3280afde9ba687bdacfd8c676775ed19b6228`). Eligible
reference structures have a finite `dft_mag_density`, a CIF and 1–20 sites.  The
frozen rule is:

- `d_min_floor = reference q0.5%`;
- `VN_low = reference q0.5%`;
- `VN_high = reference q99.5%`.

Minimum periodic distance enumerates the 27 nearest lattice images and includes
self images (except the zero image).  The same implementation is used on B0
states.  No generated held-out quality result is used to choose a threshold.

## Diagnostic estimands and gate

Both absolute candidate infeasibility and the actionable event are retained.
The preregistered trajectory-level gate uses an **entered violation**:

```text
before proposal is feasible AND candidate proposal is infeasible
```

This is the event FP-PC backtracking can prevent.  Counting an already-infeasible
early noisy state again at every step would mechanically inflate the rate while
providing no feasible backtracking solution.  Corrector and Predictor are
reported separately; stages are never pooled to hide a failure.

The diagnostic is GO only if at least 20% of trajectories have an entered
violation and that group has at least one of: mean final MaxF ≥25% worse, RMSD
≥25% worse, or E-hull ≥0.02 eV/atom worse.  Rate below 10%, or absence of final
quality association, is FAIL.  FAIL immediately stops B; implementation and
P0 files become explicit `NOT_RUN_DIAGNOSTIC_FAIL` placeholders.

## Independence and safety

The five historical stashes are untouched.  A repository-wide/worktree-wide
seed scan found no reference to `720000–720099`; conditional-on-GO B P0 seeds
are frozen as `720000–720015`, disjoint from A's `710000` namespace.  B never
changes atomic types, uses forces/energies, retrains MatterGen, enables Adaptive
CFG, combines A+B, or runs Formal32/256.
