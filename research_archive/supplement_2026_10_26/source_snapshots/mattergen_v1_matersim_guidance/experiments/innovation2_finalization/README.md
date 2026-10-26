# Innovation 2 finalization

This directory controls the user-authorized, training-free F0 evidence chain and independent F1 P0. It does not change the thesis body or frozen P0/Formal32 scientific sources.

## Current execution

- GPU 5/6: A1 Formal256, followed by A5 compatibility (64 paired seeds).
- GPU 7: A3 direction ablation, then A4 bounded/unbounded, B F1 16-pair P0, and A6 equal-budget POST.
- Each GPU runs one seed process at a time. CPU official-quality evaluation overlaps the next cohort's generation.
- `master_status.md` refreshes after each completed pair and each completed analysis. `progress_snapshot.json` is a timestamped operational estimate, not an effect estimate.
- `pipeline_process.json`, per-cohort `active_stage.json`, and `logs/pipeline_*.log` identify running jobs. A stage can remain listed after completion; inspect its corresponding `*_stage_result.json`.

All 432 distinct registered seeds were searched against 12,641 historical text files. Six nonoverlapping cohort ranges are registered in `experiment_registry.csv`; A6 reuses its own C0 structure for POST, so the matrix requires 1,120 diffusion draws plus 32 postprocess-only structures.

## Validation completed before new-cohort inference conclusions

- Five CPU mechanism tests: norm preservation, translation control, periodic-coordinate mapping, accept/retry/shrink behavior and failure fallback.
- Four synthetic analysis gate tests: confirmation, zero effect, call-budget failure and non-estimable retention denominators.
- Historical analysis regression reproduces Formal32 mean forces and all 32 paired wins. Its new bootstrap uses the preregistered new seed and is not a replacement for the frozen historical interval.
- Adaptive controller bytes are identical to the frozen A0 commit; multi-field RMS and unaccelerated fusion ASTs match after removing disabled corrector-gating branches.
- First A3 registered pair passes timing, norm and actual-call accounting checks; its scientific outcomes were not used to change the method.

The frozen sampler, protocol and generation implementation locks must not be edited while experiments run. Analysis/reporting code is separate; implementation fixes require an explicit audit record, not silent replacement or seed substitution.

## Completed independent evidence

CHGNet 0.3.0 evaluates forces independently of MatterSim generation. Formal32 showed a 4.3752% mean MaxF reduction, 19/0/13 wins/ties/losses and a 20,000-resample relative 95% interval of approximately [0.1325%, 9.8036%]. This triggered Formal256 cross-potential evaluation. CHGNet also supplies the preexisting magnetic-property guardrail; its training data are not asserted to be disjoint from MatterSim's.

## Important interpretation boundaries

- G2/G3/G4 replay the paired G1 magnitude sequence. G4 reverses the paired G1 force direction, not the force of its own evolving structure. Report this reference-trajectory limitation explicitly.
- A4 removes the effective 0.005 Å saturation as well as the redundant 0.01 Å hard cap. Geometry safety remains; no cap sweep is allowed.
- F1 has frozen epsilon 1e-4 eV/Å, radii 0.005/0.0025/0.00125 Å, at most two retries, and nonincreasing radius per sample. It does not implement early stopping or cross-sample budget allocation.
- F1 P0 GO only suggests next-round Formal32. It does not automatically replace F0 or start another experiment.
- A5 70% retention is an interpretation threshold, not statistical significance. A nonpositive reference gain is not estimable and cannot pass.
- Actual force-model calls count `calculator.calculate`, including failed calls; cached energy/force requests are not double counted. Final evaluation calls are separate for all methods.
- Scientific failures and execution failures are different. Missing seeds or incomplete evaluation prevent confirmation. No failed seed is replaced.

SURROGATE_PROPERTY_EVAL=True. DFT_VERIFIED=False.

## Read-only progress commands

From the F0 worktree, use `/mnt/datasets-livsyn/dxl/alm/.venv/bin/python -m experiments.innovation2_finalization.status` or the `progress_report` module. Do not start a duplicate pipeline while the recorded jobs are active.

The automatic pipeline writes per-cohort `final_report.md`, `audit_results.json`, `decision_summary.json`, metric tables and plots, followed by a unified report here. A pending report is not evidence that an experiment passed.
