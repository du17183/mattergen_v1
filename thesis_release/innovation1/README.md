# Innovation 1: Reference-Preserved Multi-Branch Guidance

## Final method

The thesis method is **Reference-Preserved Budget-Constrained Multi-Branch
Guidance with Safety-Constrained Fallback**, instantiated as frozen Fixed-K2.
It uses an exact C0 prefix and continuation, cloned sampler/RNG state, the
prespecified `GPulse` and `PPulse` branches, terminal surrogate selection and C0
fallback. Deployment generation compute is approximately 2.2× C0.

## Confirmatory result

On 128 fresh paired C1 seeds, property MAE decreased from 0.034796 to 0.026332
(24.32%; absolute 20,000-bootstrap CI [0.005817, 0.011391]). The paired outcome
was 53 improvements, 75 ties/fallbacks and 0 losses. Frozen surrogate guardrails
passed: E-hull and Stable improved, NUS increased, and validity stayed at 100%.

The final MAE ordering is:

```text
Fixed-K2 (0.026332) < Linear-K2 (0.027522)
< Random-K2 (0.028533) < C0 (0.034796)
```

## Negative and mechanism evidence

The 179-feature learned Linear-K2 allocator, reconstructed with C=1.0 and exact
16/16 held-out Top-2 reproduction, did not beat Fixed-K2 independently. It is
archived under `negative_results/`. Earlier adaptive variants include MIXED and
FAIL outcomes and are retained rather than hidden. Counterfactual and
branch-compatible oracles support headroom as a mechanism, not a deployable
learned selector.

## Directory map

- `method/`: exact archived branch sampler, feature and analysis code.
- `configs/`, `seeds/`: frozen protocol, manifests and registered seeds.
- `results/`: C1 and oracle compact source evidence.
- `negative_results/`: Phase B allocator and historical V1–V5/Stage/Field record.
- `tables/`, `figures/`: publication-ready derived outputs.
