# Adaptive Budget Search — design only, not implemented

Status: proposal for a **future, separately preregistered study**. It is not part of the completed Fixed-K2, Best-of-2 or K0–4 outcomes. No router, threshold, new seed, model change or experimental claim is introduced here.

## Goal and safe execution path

Run the frozen C0 sampler while retaining its exact step-400 diffusion/RNG checkpoint; complete C0 and evaluate the predeclared proxy signal. If a **frozen** difficulty rule classifies the sample easy, return C0. If hard, restore the saved checkpoint and run the unchanged Fixed-K2 GPulse/PPulse suffixes, then apply the unchanged SAFE-A selection against the already generated C0. If no candidate passes SAFE-A, return C0. The router changes **whether** a sample spends extra suffix budget, not how the branches, verifier or acceptance work.

If fraction `p` of samples is routed hard, nominal generator score calls are `2000 + p×2400` per sample, or `1 + 1.2p` C0 multiples. This excludes checkpoint I/O, CHGNet/MatterSim difficulty assessment, per-terminal verifier calls, wall time and memory. If routing requires C0 CHGNet/MatterSim results, count those calls for **every** sample; hard samples add two candidate calls of each proxy. Report observed total cost, not only the nominal formula. Maintain one-sample-per-worker deterministic RNG behavior and a byte-for-byte C0 fallback; no stochastic batch merging.

## Candidate difficulty signals — hypotheses, not tuned rules

| Signal available after C0 | Why it might help | Main danger / extra cost |
|---|---|---|
| C0 CHGNet absolute error to the known target | Large error may create opportunity for a safe suffix improvement | Same verifier drives selection and primary endpoint, producing regression-to-mean/selection optimism; one CHGNet call for every sample |
| Verifier confidence / ensemble disagreement | Could distinguish uncertain proxy predictions from confidently good C0 terminals | A single CHGNet has no calibrated uncertainty by itself; an ensemble needs training/evaluation cost and independent calibration; do not invent “confidence” from one score |
| C0 MatterSim E-hull, stability and evaluation validity | Could route low-quality or invalid structures to additional search | MatterSim latency for all samples; SAFE-A forbids worsening, but routing high-risk C0 may not yield improvable candidates |
| Prefix/trajectory diagnostics known before C0 terminal | Could decide early and save some downstream work | Requires a new predictive model or calibrated diagnostic; may leak branch outcomes if fitted on confirmation data |

The simplest initial candidate is a **single C0-property-error threshold**, optionally guarded by C0 quality risk, because it reuses already specified proxies. This does **not** mean it is a valid or winning rule. Calibration must test whether threshold rank predicts *incremental safe Fixed-K2 benefit*, not merely C0 error. A no-uncertainty signal must not be labelled uncertainty-aware.

## Prospective calibration and confirmation protocol

1. On a **new calibration set** disjoint from historical C1, Best-of-2, Budget Scaling and future confirmation, freeze C0/Fixed-K2 behavior, collect potential outcomes and full cost ledgers. Check whether the proposed C0-only difficulty signal predicts the paired increment `C0 MAE − Fixed-K2 MAE` without using any post-C0 branch metric at inference time. Explore a small declared threshold grid, including always-C0 and always-Fixed controls; all tuning stays inside calibration.
2. Freeze and serialize exactly one routing rule, feature definitions, threshold, missing-value behavior, model/proxy versions, branch strategy, SAFE-A acceptance, primary endpoint, quality guardrails, cost metric and decision criteria. Hash code/config/seed manifests before any confirmation result is read. Prespecify the allowed hard fraction or cost ceiling and a **cost-matched** Fixed-width/Independent Best-of-N comparison where feasible.
3. Register a **fresh confirmation set** and run one pass. Primary question should be either noninferior property MAE at lower measured all-in cost, or improved MAE under a fixed all-in budget; choose one before opening results. Report paired 20,000-bootstrap CI, W/T/L, E-hull, stable, NUS, validity, novel and unique rates, score calls, CHGNet/MatterSim calls, wall time and memory. Include the fraction routed hard and fallback rate.
4. Apply the preregistered go/stop rule. No threshold retuning, additional seeds, branch substitutions, verifier changes or rescue experiments after seeing confirmation. Keep a negative result as a negative result.

## Current decision

**Design is feasible but experimental value is unproven.** The completed K0–4 experiment has many fallback ties, yet those are not an ex-ante classifier. The newer Best-of-2 comparison does not justify claiming that routed Fixed-K2 will beat independent sampling. Prioritize accurate thesis reporting first; execute this only as a new, independently powered study if resources and a specific cost-saving endpoint warrant it.
