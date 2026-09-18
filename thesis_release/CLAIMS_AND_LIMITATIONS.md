# Claim–Evidence Matrix

This document overrides broader interpretations in earlier exploratory reports.
`SUPPORTED` means supported under the frozen cohort, metric and surrogate
evaluation protocol; it does not imply universality or DFT confirmation.

| Claim | Evidence | Status | Allowed wording | Forbidden wording |
|---|---|---|---|---|
| Branch-compatible guidance headroom exists | Historical branch oracle, test n=12, property MAE 0.028799 to 0.020041 (30.41%) | SUPPORTED_MECHANISM | A branchable oracle shows actionable headroom | The oracle is a deployable selector |
| Fixed-K2 improves property MAE over C0 | Fresh C1-128; 0.034796 to 0.026332; 95% CI of gain [0.005817, 0.011391] | SUPPORTED | Fixed-K2 improved surrogate property MAE on C1-128 | Fixed-K2 is universally optimal or physically validated |
| Fixed-K2 meets frozen surrogate guardrails | C1 E-hull 0.104296 to 0.081135; Stable 58.59% to 70.31%; NUS 32.81% to 37.50%; validity unchanged at 100% | SUPPORTED | No frozen surrogate guardrail was violated | Guaranteed safe generation |
| Learned Linear-K2 is better than Fixed-K2 | C1 MAE 0.027522 vs 0.026332; Linear-minus-Fixed improvement CI crosses/extends below zero | NOT_SUPPORTED | Linear-K2 is a negative confirmatory result | Learned allocation is the validated contribution |
| Adaptive CFG reliably improves MatterGen | V1 mixed; V2/V4/V5/Field failed; Stage mixed | NOT_SUPPORTED | Adaptive-CFG exploration motivated the branch framework | Adaptive CFG is robustly validated |
| RC-NFGD reduces MatterSim MaxF | Formal256, 0.226408 to 0.158911; 29.81%; 254/0/2 | SUPPORTED | RC-NFGD reduced MatterSim-predicted MaxF on Formal256 | RC-NFGD lowers true DFT force |
| RC-NFGD reduces MatterSim mean force and RMSD | Formal256 reductions 29.72% and 14.17%, respectively | SUPPORTED | Paired surrogate metrics improved | All structural-quality metrics improved |
| RC-NFGD preserves Stable and Validity | Formal256: Stable 80.86% for both; validity 100% for both | SUPPORTED_GUARDRAIL | These two frozen guardrails were maintained | All quality attributes were improved |
| RC-NFGD improves property MAE | Formal256 0.009757 to 0.009868 (1.14% worsening) | NOT_SUPPORTED | Property MAE changed slightly in the unfavorable direction | RC-NFGD improves the target property |
| CHGNet agrees directionally | Paired n=256; MaxF −14.36%, mean force −9.59%, positive absolute bootstrap CIs | SUPPORTED_INDEPENDENT_SURROGATE | An independent CHGNet surrogate showed directionally consistent force reductions | Independent DFT validation confirms RC-NFGD |
| RC-NFGD has DFT validation | No DFT calculation is present in this release | FALSE | DFT validation has not yet been performed | DFT validated / physically verified |
| The trust-region cap is a proven performance core | Frozen unbounded comparator was stronger and passed its tested guardrails | NOT_SUPPORTED | The cap is a conservative design choice | The cap is proven necessary for performance |
| RC-NFGD is optimal versus post-processing | Equal-budget post-processing had lower MatterSim MaxF on n=32 | NOT_SUPPORTED | Online RC-NFGD is distinct from, and weaker than, this tested post-processing baseline | RC-NFGD dominates relaxation/post-processing |
| Innovation 1 and 2 have synergy | Frozen compatibility track A5 failed | NOT_SUPPORTED | The contributions are reported independently | The combined method is validated |

## Interpretation boundaries

- `MatterSim` and `CHGNet` are surrogate evaluators here. CHGNet independence
  refers to the evaluator used after generation, not independence of all possible
  training data.
- Oracle studies establish mechanism headroom, not deployable performance.
- The C1 result supports the prespecified Fixed-K2 comparator. It does not rescue
  the independently failed learned Linear-K2 allocator.
- The post-generation `POST` comparator is not part of the online RC-NFGD method.
- Historical reports are retained for provenance and may contain interpretations
  superseded by the final evidence matrix above.

`DFT_VERIFIED = false`
