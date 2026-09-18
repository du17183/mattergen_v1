# Counterfactual Adaptive CFG V3 — Final Report

## Final decision

```text
ORACLE_HEADROOM = FAIL
ORACLE_PREDICTABILITY = NOT_RUN
V3_P0 = NOT_RUN
V3_FORMAL256 = NOT_RUN
INNOVATION1_FINAL_STATUS = MIXED
NEXT = STOP_ADAPTIVE_CFG_RESEARCH
```

The frozen stop rule was applied. No controller was trained, no V3 sampler was
implemented, and no P0 or Formal256 samples were generated. No rescue tuning or
post-outcome parameter change was performed.

## Frozen design and execution completeness

- Worktree: `/mnt/datasets-livsyn/dxl/mattergen_v1_cfg_oracle`
- Branch: `experiment/counterfactual-adaptive-cfg-v3`
- Base commit: `1af301c5e534ff6b9b9f96ea5dab48036d9c46af`
- Oracle config SHA-256:
  `d8347fd692421721013ed455d9ad3404497e0145498e32baa040c86328839099`
- Frozen candidates: DOWN=1.75, KEEP=2.00, UP=2.25.
- Frozen decision steps: 100, 250, 400, 550, 700, 800, 900, 950.
- Short horizon: 15 reverse steps; full validation: exactly 25% of decision
  states.
- Fresh Oracle seeds: 32/32 completed with zero historical overlap.
- Generated outcomes: 768 short, 192 full, 32 fixed-CFG2 base structures.
- Online state rows: 256; branch-manifest rows: 768.
- Exact pairing: all 64 full KEEP checks matched the corresponding CFG2 final
  structure with identical atomic numbers and zero coordinate/cell error.
- Generation score calls: 243,104. Per-seed wall time median 334.56 s, range
  233.32–462.79 s.
- Property evaluation: 992 total outcomes; 944 valid and 48 assigned the frozen
  invalid penalty.
- MatterSim official evaluation: 912 valid candidate structures across all six
  SHORT/FULL × DOWN/KEEP/UP groups; relaxation success was 100% in every group.
- Bootstrap: 20,000 seed-clustered resamples with frozen seed 20260917.

All property and stability results are surrogate evaluations. No result in this
report is DFT verified (`SURROGATE_PROPERTY_EVAL=True`, `DFT_VERIFIED=False`).

## Effect summary

| Selection | States | CFG2 property MAE | Selected property MAE | Frozen objective improvement | CFG2 E-hull | Selected E-hull | Stable | NUS | Validity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short-horizon oracle | 256 | 0.059414 | 0.047571 | +51.810% | 0.151554 | 0.161359 | 43.750% → 42.969% | 14.063% → 13.281% | 93.359% → 98.828% |
| Full-outcome oracle | 64 | 0.030446 | 0.022158 | +27.222% | 0.106236 | 0.113015 | 46.875% → 42.188% | 10.938% → 12.500% | 100% → 100% |
| Short label evaluated on full outcome | 64 | 0.030446 | 0.032548 | **−6.903%** | 0.106236 | 0.120815 | 46.875% → 46.875% | 10.938% → 17.188% | 100% → 100% |

The full-outcome hindsight oracle has real retrospective headroom. Its mean
selected-minus-CFG2 normalized objective was −0.082880 with seed-clustered 95%
CI [−0.116350, −0.052684]. However, the online-usable short-horizon label does
not transfer to the final outcome: it worsens the frozen objective by 6.903%
instead of meeting the pre-registered minimum 5% improvement.

## Action and timestep evidence

- Short labels: DOWN 17.578%, KEEP 63.281%, UP 19.141%.
- Full labels: DOWN 23.438%, KEEP 62.500%, UP 14.063%.
- Short/full exact label agreement: 62.500%.
- All 32 seed trajectories had at least one non-KEEP short action; 24/32 had at
  least one non-KEEP full action in their two selected full-rollout states.
- Full-oracle actions at steps 100–550 included DOWN, KEEP, and UP, showing
  sample/timestep variation. At steps 900 and 950, all full labels were KEEP.
- Short/full agreement was weakest at the most adaptive middle region: 25.0% at
  step 400, versus 100% at steps 900 and 950.

These observations answer the diagnostic question carefully: local optimal CFG
does vary under final-outcome hindsight, especially early and mid trajectory,
but the frozen 15-step proxy is not a reliable representation of that final
choice.

## Harm analysis

- Full hindsight oracle: property harm 0%, E-hull harm 18.750%, stable harm
  9.375%, validity harm 0%.
- Short-selected full outcomes: property harm 15.625%, E-hull harm 15.625%,
  stable harm 7.813%, validity harm 0%.

Although mean guardrails passed, paired harmful decisions remain material for
the short-horizon policy. This is consistent with the negative final-outcome
transfer result.

## Frozen Headroom Gate

The following checks passed:

- Short frozen-objective improvement ≥10%.
- Full frozen-objective improvement ≥10%.
- Short and full non-KEEP fractions ≥20%.
- Full maximum class share ≤80%; both DOWN and UP shares ≥5%.
- Short/full exact agreement ≥50%.
- Mean E-hull worsening ≤0.02 eV/atom.
- Stable, NUS, and validity drops ≤5 percentage points.

The decisive check failed:

```text
short-selected full-outcome improvement = -6.903%
required minimum = +5.000%
```

Because every pre-registered check was conjunctive, this single failure makes
`ORACLE_HEADROOM=FAIL`. The predictability gate cannot be run: training a model
to predict short labels would optimize a proxy that is already harmful on the
final outcome.

## Scientific conclusion

Fixed CFG=2 is not globally identical to the retrospective oracle, so the
experiment found genuine hindsight variation. The failure is more specific and
more important for an online method: the permitted short-horizon feedback does
not expose the final optimal action reliably enough to support Adaptive CFG V3.
Under the one-shot frozen protocol, Adaptive CFG research therefore stops here.

For the thesis, Innovation 1 should remain `MIXED`: V1 had non-stable cohort
effects, V2 failed fresh P0, and the counterfactual study now explains why a
simple online feedback upgrade is not scientifically justified. The negative
result is still useful evidence about proxy mismatch and harmful adaptation.

## Execution transparency

- A pre-outcome Hydra target error was corrected by pointing the sampler target
  to its inherited `from_pl_module` factory; the protocol was re-frozen before
  successful sampling.
- After generation, GPU 0 became occupied by an unrelated process. Property and
  quality evaluation were moved to GPUs 1/3/4/5/6/7 without changing scientific
  code or metrics.
- A pandas duplicate-column error occurred after state labels were written but
  before any Gate result. A wrapper verified the two stored label sources were
  identical for every state, selected the same frozen labels, and resumed the
  original analysis. Full details are retained in `technical_fix_log.md`.
- The main worktree and all five historical stashes were left unchanged.

## Retained artifacts

- `oracle/oracle_dataset.csv`, `oracle/oracle_headroom.csv`,
  `oracle/action_distribution.csv`, and `oracle/oracle_summary.json`.
- All 32 per-seed generation directories and all short/full/base structures.
- CHGNet property metrics and all six official MatterSim detailed evaluations.
- `oracle/oracle_bootstrap.json`, harm metrics in the summary, and four Oracle
  plots in `plots/`.
- Predictability and P0 `decision_summary.json` files explicitly record
  `NOT_RUN`; the Formal256 directory does not exist.
