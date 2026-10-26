# Counterfactual Oracle technical incident log

## Attempt 1: seed 816000

- Scope: registered Oracle seed `816000`, before any diffusion sampling step.
- Outcome exposure: none. No structures, online features, branch metrics, property
  predictions, Oracle labels, or gate results were produced.
- Error: Hydra instantiated `CounterfactualCFGOracleSampler` as a constructor and
  forwarded `pl_module` into `PredictorCorrector.__init__`, which does not accept
  that argument.
- Root cause: `sampler_partial._target_` pointed to the sampler class instead of
  its inherited `from_pl_module` factory, unlike the official guided sampler.
- Technical fix: change only the target path to
  `CounterfactualCFGOracleSampler.from_pl_module`.
- Scientific protocol unchanged: Oracle candidates, decision timesteps, horizon,
  full-rollout allocation, RNG pairing, seeds, features, objective, deadband,
  guardrails, models, bootstrap, and all GO/FAIL thresholds are unchanged.
- Evidence retained: `logs/oracle_first_seed.log`,
  `oracle/technical_failures/816000_attempt1.partial`, and
  `frozen_manifest_attempt1.json`.
- Old worker SHA-256: `e73bbe2c126cbe3a1229e975b2891fdb8d3c38bc52db327f0e5527ac7e3e4dc5`.
- Fixed worker SHA-256: `a6c8ddd2db3bade6e4599280eb886a040ede7c34f0ca5511be80b50c0d82dbc5`.
- Frozen Oracle config SHA-256 remains
  `d8347fd692421721013ed455d9ad3404497e0145498e32baa040c86328839099`.

Because no scientific outcome was observable, this is a pre-outcome engineering
correction rather than a data-dependent method change.

## Attempt 2: seed 816000

- Outcome exposure: none. The process exited during Python imports, before model
  loading and before creation of a seed output directory.
- Error: `ModuleNotFoundError` for the new in-worktree Oracle sampler.
- Root cause: the manually created tmux command omitted the repository root from
  `PYTHONPATH`; the frozen pipeline already supplies it.
- Resolution: relaunch the identical frozen worker with
  `PYTHONPATH=/mnt/datasets-livsyn/dxl/mattergen_v1_cfg_oracle`.
- Evidence retained: `logs/oracle_first_seed_attempt2.log`.
- Protocol and frozen source: unchanged; no manifest update was required.

## Resource reassignment after Oracle generation

- All 32 registered generation seeds completed before this event.
- When the pipeline advanced to CHGNet property evaluation, an unrelated task
  (PID `3105853`) began actively using physical GPU 0.
- At the time of the process audit, the Oracle property evaluator had not
  allocated GPU memory. Its process was terminated with `SIGTERM` to avoid
  sharing an occupied device; the parent pipeline recorded return code `-15`
  and exited.
- Frozen generation data and all 32 per-seed pairing checks were retained.
- Resume placement: property evaluation on GPU 1; the unchanged six quality
  groups on GPUs 1/3/4/5/6/7. GPUs 0 and 2 are excluded.
- The frozen evaluator, candidate outcomes, objective, metrics, and gate rules
  remain unchanged. Supplemental resume wrappers only control physical device
  placement and pipeline continuation.

### Aggregate resume detail

- The first property process completed deterministic aggregation before it
  received `SIGTERM`: 768 short structures, 192 full structures, 32 base
  structures, 256 online feature rows, and 768 branch-manifest rows. It produced
  no property metrics and no evaluation-structure directory.
- A first resume invocation rewrote the three derived aggregate EXTXYZ files
  from the unchanged per-seed sources, then stopped at the create-only existing
  `online_features_raw.csv`. Again, it produced no property results.
- All five aggregate counts were independently validated. The next resume calls
  the unchanged frozen `evaluate_properties` function directly on those
  aggregates and refuses to overwrite any property result.

## Analysis column-name collision

- All six official quality groups completed before this event.
- The frozen analysis computed and wrote enriched outcomes plus short/full
  state labels, then failed before headroom summaries, bootstrap, harm metrics,
  plots, or a Gate decision.
- Cause: candidate tables already contain `oracle_label`; merging the same label
  column again created pandas suffixes and made the unsuffixed lookup fail.
- A supplemental wrapper leaves the frozen analysis source unchanged. It first
  asserts that candidate-table labels and independently stored state labels are
  identical for every `(seed, sampling_step)`, then selects rows using that same
  frozen label and calls the original analysis `main`.
- No label, outcome, objective, metric, bootstrap, guardrail, or threshold is
  changed.
