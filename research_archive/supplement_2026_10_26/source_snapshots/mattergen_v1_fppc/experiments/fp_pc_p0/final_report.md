# FP-PC P0 final report

## Decision

```text
FP_PC_MOTIVATED = NO
FP_PC_P0 = NOT_RUN_DIAGNOSTIC_FAIL
Adaptive CFG used? NO
SURROGATE_PROPERTY_EVAL=True
DFT_VERIFIED=False
NEXT=STOP_FP_PC_NO_RESCUE
```

The frozen B0 gate failed: **violation/no-violation outcome comparison is not estimable**.  Per protocol, FP-PC was not
implemented, the reserved fresh seeds `720000–720015` were not consumed, no
16-seed generation/evaluation was run, and no rescue threshold or sweep was
attempted.

## Frozen setup

- Worktree/branch: `/mnt/datasets-livsyn/dxl/mattergen_v1_fppc` /
  `experiment/fp-pc-p0`.
- MatterGen HEAD: `1af301c5e534ff6b9b9f96ea5dab48036d9c46af`.
- Official target-0.2 checkpoint SHA256:
  `01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e`.
- Constant CFG 2.0, 1000 steps, one Corrector step/time; Adaptive CFG OFF.
- Reference-only bounds: `d_min >= 1.332666 Å`,
  `9.257931 <= V/N <= 57.985272 Å³/atom`.
- Threshold source: fixed 4096-structure reservoir from the official Alex-MP
  training split, q0.5%/q99.5%; 4096/4096 parsed, no held-out outcome used.

## B0 execution integrity

- 64 historical target-0.2 seeds (`99000–99063`) were deterministically replayed.
- Every trajectory produced 1000 Corrector plus 1000 Predictor observations:
  128,000 proposal rows total.
- Diagnostic accepted every official proposal unchanged; it did not implement
  FP-PC during data collection.
- Replay equivalence: atomic numbers/formula exact; maximum cell difference
  `6.274e-07 Å`,
  maximum periodic fractional-position difference
  `3.762e-09`.
- Uncontended diagnostic sampling runtime had median `88.245 s/trajectory`
  (maximum `93.730 s`).  Seed `99014` took `424.270 s` after an unrelated
  external process occupied GPU1; it still completed with the exact 2000-call
  contract.  The next seed's interrupted empty attempt was retained in the
  audit log and then executed exactly once to completion on GPU6.  This resource
  event changes neither the stochastic seed nor any scientific measurement.

## Diagnostic questions

**Q1 — Predictor violation trajectories.**
`64/64`
(`100.00%`) had at least one actionable
Predictor entry violation.

**Q2 — Corrector violation trajectories.**
`64/64`
(`100.00%`) had at least one actionable
Corrector entry violation.

**Combined trajectory rate.**
`64/64`
(`100.00%`) had either stage.  The GO rate
threshold was 20% and the hard low-rate failure threshold was 10%.

**Q3 — final-quality association.**  Group sizes were violation
`n=64` and no-violation `n=0`;
estimable=`False`.  Mean relative/absolute worse
effects (violation minus no-violation; positive means worse) were:

- MaxF relative: `NOT_ESTIMABLE`;
- RMSD relative: `NOT_ESTIMABLE`;
- E-hull absolute: `NOT_ESTIMABLE eV/atom`.

None met the preregistered degradation test of +25%, +25%, or +0.02 eV/atom.
As a secondary, non-gating analysis, total entry count was *negatively*
associated with MaxF rather than positively associated with harm: Spearman was
`-0.486` overall,
`-0.705` in
calibration and
`-0.262` in
validation.  This supplies no rescue evidence for suppressing these proposals.

**Q4 — timing.** Actionable entry event counts were early/mid/late =
`2654/2310/960`.
Predictor total=`1931` and Corrector
total=`3993`.  Timing is descriptive;
it was not used to redefine the threshold or rescue the route.

Absolute candidate infeasibility occurred in
`62.76%` of
proposal rows, but this includes intentionally noisy states already outside the
clean-reference region.  It is not the preregistered actionable gate and cannot
motivate backtracking from an infeasible current state.

## Interpretation

The method required both a sufficiently common **preventable** proposal crossing
and a materially worse final outcome for affected trajectories.  The observed
data failed that joint requirement.  Implementing a trust region now would be a
post-hoc intervention without the required causal/mechanistic motivation.  FP-PC
therefore stops at diagnostic FAIL.  Empty P0 tables are explicit NOT_RUN
artifacts, not missing or zero-valued experimental results.

No MatterGen training, force/energy optimization, atomic-type modification,
A+B combination, Formal32 or Formal256 was performed.  The five historical
stashes and all historical checkpoints/experiments remain untouched.
