# Confirmatory protocol amendment — artifact reconstruction

Before any confirmatory seed registration or outcome generation, the artifact
audit found that Phase B retained the frozen method, C value, raw feature list,
splits, labels, and held-out selections, but did not serialize the fitted
scaler or logistic-regression parameters.

The user explicitly authorized one deterministic reconstruction using only the
original Phase-B train=32 cohort, original frozen source, original feature
list, and C=1.0.  The reconstructed pipeline is admissible only if it exactly
reproduces the stored held-out Top-2 choices and the prior validation/test
results.  No confirmatory seed may be registered before this check passes.

The historical Phase-B decision remains permanently:

```text
ADAPTIVE_ALLOCATION_VALUE = NOT_SUPPORTED
```

No model, feature, hyperparameter, candidate, selector, or historical outcome
may be changed during reconstruction.
