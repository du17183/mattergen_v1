# MatterGen Thesis Extensions

## Overview

This directory is the compact, auditable evidence package for two MatterGen
inference-time thesis contributions. It contains frozen method code, registered
seeds, small result artifacts, derived tables, plots, negative results and
CPU-only validation. Model weights, raw generated structures and experiment
caches are deliberately excluded.

The single source of truth for headline numbers is
[`combined_summary/main_results.csv`](combined_summary/main_results.csv).

## Thesis Contributions

### 1. Reference-Preserved Budget-Constrained Multi-Branch Guidance

The method shares an exact C0 prefix, preserves sampler and RNG state, evaluates
two frozen guidance branches (`GPulse` and `PPulse`), and uses a terminal safe
selector with C0 fallback. On 128 completely fresh confirmatory seeds, frozen
Fixed-K2 reduced property MAE from **0.034796** to **0.026332** (absolute gain
0.008464; 24.32%; 20,000-bootstrap 95% CI [0.005817, 0.011391]) while satisfying
the predefined surrogate quality guardrails. There were 53 improvements, 75
tie/fallback outcomes and no harmful selections.

The learned sample-specific Linear-K2 allocator did **not** outperform Fixed-K2
under independent confirmation: MAE 0.027522 versus 0.026332. It is retained as
a reproducible negative result, not as the final method.

### 2. Reliability-Calibrated Late-stage Neural Force-Guided Diffusion

RC-NFGD evaluates a frozen neural force model only on reliable late predicted
clean states, maps a bounded Cartesian force displacement into periodic
fractional coordinates, modifies only the position predictor, and continues
reverse diffusion. On paired Formal256 samples, it reduced MatterSim-predicted
MaxF by **29.81%**, mean atomic force by **29.72%**, and RMSD by **14.17%**.
Stable (80.86%) and validity (100%) were unchanged. Independent CHGNet surrogate
evaluation was directionally consistent: MaxF decreased by 14.36% and mean
force by 9.59%.

## Repository Structure

- `innovation1/`: final Fixed-K2 method, C1 evidence and allocator negatives.
- `innovation2/`: RC-NFGD method, Formal256 evidence and ablations.
- `combined_summary/`: cross-study status, headline results and figure index.
- `scripts/`: deterministic table/figure reproduction and release validation.
- `tests/`: lightweight CPU-only mechanism and integrity checks.

Historical reports are preserved verbatim in dedicated result directories.
Some were generated before later confirmation; the final interpretation is this
README together with [`CLAIMS_AND_LIMITATIONS.md`](CLAIMS_AND_LIMITATIONS.md).

## Reproduction

From the repository root:

```bash
python thesis_release/scripts/reproduce_tables.py
MPLCONFIGDIR=/tmp/mattergen-thesis-mpl python thesis_release/scripts/reproduce_figures.py
python thesis_release/scripts/validate_release.py
python -m unittest discover -s thesis_release/tests -v
```

These commands use only archived compact files and do not rerun C1, Formal256,
Phase B, MatterGen generation, or neural-potential evaluation. See
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the full boundary.

## Evidence Status

| Item | Final status |
|---|---|
| Fixed-K2 versus C0, fresh C1-128 | SUPPORTED |
| Learned Linear-K2 versus Fixed-K2, fresh C1-128 | NOT_SUPPORTED |
| Branch-compatible guidance headroom | SUPPORTED (mechanism/oracle) |
| RC-NFGD MatterSim Formal256 | SUPPORTED |
| CHGNet independent surrogate direction | POSITIVE |
| RC-NFGD DFT validation | NOT PERFORMED (`DFT_VERIFIED=false`) |
| Innovation 1 + 2 synergy | NOT_SUPPORTED |

## Limitations

All physical and property metrics are surrogate-model evaluations. No DFT
validation has yet been performed. Fixed-K2 was not shown superior to a generic
equal-budget best-of-N procedure. RC-NFGD did not materially improve property
MAE, and its equal-budget post-generation comparator reduced surrogate forces
more strongly on the tested 32-sample cohort. The two contributions have not
shown a reliable combined benefit.

## Citation

This is a thesis release branch. Use the branch commit reported in the release
audit when citing results; create a version tag only after final human review.
