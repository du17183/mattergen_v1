# Validation summary

## Focused validation

```text
16 passed
syntax checks passed
git diff --check passed
```

The focused suite covers protocol gates, invariant features, learned gates,
quality scoring, set ranking, new-data evaluation, Pareto selection, and the
equivariant trust-region refiner.

## Full repository suite

With the project's frozen compatibility setting:

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -m pytest -q
```

the result was:

```text
169 passed
6 failed
```

The six failures are pre-existing environment/baseline incompatibilities in
unmodified code:

- one randomized periodic-boundary translation-invariance failure;
- four stale corrector tests that omit the currently required `dt` argument;
- one PyTorch 2.7 incompatibility caused by the removed
  `ReduceLROnPlateau(verbose=...)` argument.

No core MatterGen source file is modified by this branch.
