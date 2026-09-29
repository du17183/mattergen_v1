# Innovation 1 trajectory-search evidence report

Status: both fresh 128-seed cohorts complete; surrogate-only, DFT_VERIFIED=false.

## Method positioning

Fixed-K2 is a frozen-model, one-checkpoint, width-two, verifier-guided suffix search with exact C0 fallback. This is a search-framework instance, not a newly trained model. Historical C1-128 and the negative Linear-K2 finding remain unchanged.

## Independent-sampling comparison

|Method|Budget|Property MAE|95% CI paired gain vs C0|Stable|NUS|E-hull|Validity|W|T|L|
|---|---|---|---|---|---|---|---|---|---|---|
|C0|1.0|0.0376231805065835|[0.00000000, 0.00000000]|0.515625|0.3125|0.1238719216141893|1.0|0|128|0|
|Independent_Best_of_2|2.0|0.0280965940380731|[0.00643895, 0.01289100]|0.625|0.3515625|0.0963933965806988|1.0|36|92|0|
|Fixed_K2|2.2|0.0295062533265306|[0.00553347, 0.01105391]|0.6171875|0.328125|0.1006293145277358|1.0|53|75|0|

Paired Independent Best-of-2 minus Fixed-K2 mean = -0.00140966, 95% CI [-0.00503855, 0.00216743], W/T/L=41/59/28.

The prespecified evidence does **not** support claiming that Fixed-K2 beats Independent Best-of-2 under the current asymmetric comparison. Fixed-K2 remains a valid historical C1 method, but its distinction from extra-budget Best-of-N is unproven.

## Budget scaling

|Method|Budget|Property MAE|95% CI paired gain vs C0|Stable|NUS|E-hull|Validity|W|T|L|
|---|---|---|---|---|---|---|---|---|---|---|
|K0|1.0|0.0351405614691196|[0.00000000, 0.00000000]|0.625|0.3671875|0.0942536873178761|1.0|0|128|0|
|K1|1.6|0.0316355616310306|[0.00210758, 0.00506161]|0.6796875|0.375|0.0819283546654295|1.0|39|89|0|
|K2|2.2|0.0284539591984912|[0.00453692, 0.00910157]|0.6875|0.3828125|0.0805505072467488|1.0|59|69|0|
|K3|2.8|0.0284189563800685|[0.00453841, 0.00914017]|0.6875|0.390625|0.0807869426212388|1.0|59|69|0|
|K4|3.4|0.0246688980331377|[0.00777420, 0.01339804]|0.7109375|0.40625|0.0789596358988643|1.0|72|56|0|

The K=0–4 curve is a prespecified nested-prefix diagnostic. It may motivate K=2 as a pragmatic trade-off, but it does not retrospectively tune or change historical C1. Marginal comparisons remain exploratory unless separately preregistered/confirmed.

## Thesis and publication recommendation

For a master's thesis, the title ‘Budget-Constrained Trajectory Search with Verifier-Guided Selection for Frozen Diffusion Models’ is acceptable only with the above evidence limits stated. For a paper, add an exact-cost/terminal-count comparator and independent verifier or DFT validation before asserting a trajectory-search advantage beyond Best-of-N. Preserve the Linear-K2 null result. Do not implement Adaptive Budget Search in this stage; if evidence remains favorable, design it as a separate preregistered study.

## Evidence locations

Audit: `docs/search_framework_audit.md`; preregistration: `docs/trajectory_search_design.md`; fresh seed registry: `experiments/trajectory_search/protocol/seed_manifest.json`; per-seed generation, proxy evaluation, 20k bootstrap, tables and figures: `results/search_baseline/` and `results/budget_scaling/`.
