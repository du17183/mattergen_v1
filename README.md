# MatterGen Thesis Extensions

This branch is the final code-and-evidence archive for two inference-time
MatterGen thesis contributions. Start with the compact release package at
[`thesis_release/`](thesis_release/README.md); it contains frozen code, data,
tables, figures, negative results, limitations and CPU-only validation.

## Thesis Contributions

### 1. Reference-Preserved Budget-Constrained Multi-Branch Guidance

The frozen Fixed-K2 method shares an exact C0 prefix, preserves sampler/RNG
state, evaluates two prespecified branches and falls back safely to C0. On 128
fresh confirmatory seeds it reduced property MAE from **0.034796** to
**0.026332**, with all predefined surrogate guardrails satisfied.

The learned Linear-K2 allocator did not outperform Fixed-K2 under independent
confirmation and is reported as a negative result rather than a validated
contribution.

### 2. Reliability-Calibrated Late-stage Neural Force-Guided Diffusion

RC-NFGD applies a bounded position-only correction at reliable late diffusion
steps. On paired Formal256 samples it reduced MatterSim-predicted MaxF by
**29.81%**, mean force by **29.72%**, and RMSD by **14.17%**, while Stable and
validity were maintained. CHGNet provided directionally consistent independent
surrogate evidence.

No DFT validation has been performed: `DFT_VERIFIED=false`.

## Key Results

| Result | Status |
|---|---|
| Fixed-K2 vs C0, C1-128 | SUPPORTED |
| Learned Linear-K2 vs Fixed-K2 | NOT_SUPPORTED |
| RC-NFGD MatterSim Formal256 | SUPPORTED |
| CHGNet independent surrogate direction | POSITIVE |
| Equal-budget post-processing superiority claim | NOT_SUPPORTED; comparator was stronger on n=32 |
| Innovation 1 + 2 synergy | NOT_SUPPORTED |

Headline numbers have one source of truth:
[`main_results.csv`](thesis_release/combined_summary/main_results.csv).

## Repository Structure

- [`thesis_release/innovation1/`](thesis_release/innovation1/README.md): method,
  C1-128 evidence, historical exploration and learned-allocator negative result.
- [`thesis_release/innovation2/`](thesis_release/innovation2/README.md): RC-NFGD,
  Formal256, CHGNet evidence and ablations.
- [`thesis_release/combined_summary/`](thesis_release/combined_summary/): unified
  results, experiment status, compute and figure index.
- [`thesis_release/scripts/`](thesis_release/scripts/): deterministic release
  reproduction and validation.

## Reproduction

```bash
python thesis_release/scripts/reproduce_tables.py
MPLCONFIGDIR=/tmp/mattergen-thesis-mpl python thesis_release/scripts/reproduce_figures.py
python thesis_release/scripts/validate_release.py
python -m unittest discover -s thesis_release/tests -v
```

These commands use the committed compact results and do not rerun GPU inference.
Full details are in
[`REPRODUCIBILITY.md`](thesis_release/REPRODUCIBILITY.md).

## Evidence Status and Limitations

The authoritative claim boundaries are in
[`CLAIMS_AND_LIMITATIONS.md`](thesis_release/CLAIMS_AND_LIMITATIONS.md). All
stability, property and force claims in this release are based on surrogate
models. Fixed-K2 is not established as superior to every equal-budget generic
best-of-N method; RC-NFGD does not improve every quality metric and has not been
validated by DFT. Historical reports are retained for provenance even where
later confirmation narrowed their interpretation.

## Citation

For thesis citation, pin the final commit on
`release/thesis-final-2026`. No release tag is created until final human review.
