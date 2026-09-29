# Innovation 1 academic repositioning after trajectory-search tests

Status: complete new-study evidence; **surrogate-only**, no DFT verification. Do not edit the thesis body yet. Historical C1-128 and negative Linear-K2 conclusions remain frozen.

## What the method is

Old label: “reference-trajectory-preserving multi-branch CFG guidance with safe fallback.” It describes implementation but can sound like a CFG schedule and underspecifies the selection/search mechanism.

Proposed bounded label: “budget-constrained diffusion trajectory search with verifier-guided terminal selection for a frozen MatterGen sampler.” A search state is the complete intermediate diffusion graph (`atomic_numbers`, positions, cell), step/timestep, condition, model/schedule identity and RNG state. A search action chooses a predeclared suffix policy. The frozen sampler advances that cloned state. SAFE-A selects a terminal only after CHGNet property and MatterSim quality checks; otherwise the exact C0 continuation is returned. Fixed-K2 is **one instance**: a shared step-400 checkpoint, two suffix actions, three terminals including C0, 4,400 score calls (`2.2×`). It is not a newly trained model, general-purpose learned planner, or proof of an optimal search strategy. Its old prefix artifact does not separately serialize the condition or numerical timestep tensor/schedule hash; cross-version stand-alone replay is not established.

## Evidence, separated by cohort

1. **Historical C1-128 (unchanged):** Fixed-K2 reduced CHGNet Property MAE from `0.034796` to `0.026332` (24.32% relative), with paired gain CI `[0.005817,0.011391]` at 2.2× score calls. This supports an improvement over C0 on those historical fresh seeds, not a win over independent Best-of-N. Linear-K2's learned allocation did not beat Fixed-K2 on C1; its negative outcome is retained.
2. **New independent-sampling cohort (128 fresh paired seeds):** C0 `0.037623`, Independent Best-of-2 `0.028097` at 2.0×, Fixed-K2 `0.029506` at 2.2×. The direct paired Best-of-2-minus-Fixed-K2 contrast is `−0.001410`, 95% CI `[−0.005039,+0.002167]`. Its mean favors the cheaper independent method; uncertainty crosses zero. **The current evidence does not demonstrate that Fixed-K2 does more than a best-of-N-like extra-budget selection strategy.** It does demonstrate a valid shared-prefix trajectory-search *implementation* and C0-relative proxy benefit.
3. **New Budget Scaling cohort (another 128 fresh paired seeds):** K0/K1/K2/K3/K4 MAE is `0.035141/0.031636/0.028454/0.028419/0.024669` at `1.0/1.6/2.2/2.8/3.4×`. K3 gives near-zero incremental gain over K2 (`0.000035`; only 1/128 improves); K4 improves further (`0.003750`; 26/128 improve). Thus K2 can be described as a moderate-budget point but **not the best observed width**. Because SAFE-A nests candidate pools and never accepts a worse property proxy, monotone MAE as K increases is largely a rule-level property, not independent proof of a smooth inference-time scaling law. The irregular marginal pattern depends on the frozen branch order.

Both new cohorts satisfy the specified proxy quality guardrails versus their within-cohort C0. Do not pool means across cohorts or mix their C0 baselines. All property and quality claims refer to CHGNet/MatterSim proxies; same-proxy selection/reporting may favor overly optimistic outcomes. Novel/unique/NUS are outcome metrics, not verifier constraints. Refer to `docs/best_of_two_fairness_audit.md` for the unequal call/candidate/wall budgets.

## Decision

**For a computer-science master's thesis:** yes, it is defensible to describe Innovation 1 as a *budget-constrained trajectory-search framework instance* around a frozen diffusion model, provided the algorithm interface, state/RNG replay, fixed branch policies, explicit score/verifier budget, safety fallback and negative Best-of-2 comparison are stated. The contribution is the controlled implementation and evaluation of this search/selection formulation, not a demonstrated superiority over independent Best-of-N. Keep the title qualified; do not rewrite “framework” as an empirically superior general search algorithm.

**For a research-paper claim of trajectory-specific advantage:** no, current evidence is insufficient. The main comparator is 2.0×/2 candidates versus 2.2×/3 candidates, with mean direction opposite to the hoped-for claim and CI crossing zero. A paper would require a frozen, exactly cost- and terminal-count-matched independent or shared-prefix control, plus ideally an independent verifier or DFT confirmation. These are future work, not results of this study. No new experiments are authorized by this report.

## Thesis chapter changes to propose, not yet apply

- Rename the subsection around the exact search-state/action/transition/verifier/budget interface; make Fixed-K2 the width-two instance, rather than simply “multi-branch CFG.”
- Retain historical C1 as the confirmatory C0 comparison and separately add the two new cohorts. Do not overwrite the old table or claim C1 establishes a best-of-N advantage.
- Place the Independent Best-of-2 null comparison immediately beside the Fixed-K2 result, with budget/candidate asymmetry and paired CI stated in the caption.
- Add the K0–4 curve and marginal-gain figure as an exploratory cost/width analysis; label K3 plateau and K4 gain, without choosing K post hoc.
- State proxy reuse/verifier-hacking risk, surrogate-only quality, absence of DFT and Linear-K2's unsupported improvement in limitations.

`figures/search_framework/` contains the proposed plots and legends; the thesis body itself was not changed.

## Adaptive Budget recommendation

Worth a **separate preregistered feasibility study only if there is time and a true compute-saving goal**. The present fixed-width data identify possible wasted calls on easy samples (many fallback ties), but do not show that a reliable C0 difficulty signal exists or that a router will beat a fixed-width/independent baseline at equal total cost. First calibrate a single frozen difficulty rule on a separate set; then test on fresh confirmation seeds, accounting for verifier calls and the saved-prefix/resume overhead. Do not implement or claim this as Innovation 1's current validated result. Design only: `docs/adaptive_budget_design.md`.
