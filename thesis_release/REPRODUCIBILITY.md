# Reproducibility

## Frozen version and provenance

- Release branch: `release/thesis-final-2026`
- Integration base: `1af301c5e534ff6b9b9f96ea5dab48036d9c46af`
- Source worktrees and commits: [`../thesis_release_inventory.md`](../thesis_release_inventory.md)
- Frozen integrity hashes: [`release_checksums.json`](release_checksums.json)
- Main numerical source: [`combined_summary/main_results.csv`](combined_summary/main_results.csv)

The experiment-generation code was selectively archived from its original
worktrees. Existing historical reports were not rewritten to retrofit later
claims. Derived tables and figures are regenerated only from committed compact
artifacts.

## Recorded environment

The source experiments ran in the project MatterGen environment on the cluster's
NVIDIA H20 GPU pool. The archival CPU reproduction was checked with Python
3.10.12, NumPy 2.2.6, Matplotlib 3.10.9 and PyTorch 2.9.0+cu128. Exact package
requirements for the underlying MatterGen installation remain in the repository
environment files. Hardware throughput should not be compared across different
GPU, driver or filesystem configurations.

The current archive session could not query the NVIDIA driver, so no unverified
driver number is asserted here. Per-run GPU indices and timing/accounting remain
in the frozen summaries.

## External assets not included

Full experiment execution requires separately licensed or distributed assets:

- official MatterGen model checkpoints and data preprocessing assets;
- MatterSim checkpoints and runtime dependencies;
- CHGNet model weights for the independent surrogate evaluation;
- source datasets required by MatterGen and property/stability evaluators.

Obtain these from their official distributions and comply with their licences.
No third-party weights, LMDB datasets, raw generation caches or environment
directories are included in this branch.

## Seeds and frozen configuration

- Innovation 1 registered 256 seeds before confirmation; C1 uses the frozen
  128-seed subset recorded in `innovation1/seeds/confirmatory_seed_manifest_256.json`.
- Innovation 2 P0, Formal32 and Formal256 seeds are under
  `innovation2/seeds/`; Formal256 contains 256 paired seeds with zero recorded
  historical overlap.
- Frozen parameters and implementation locks are under each innovation's
  `configs/` directory.
- The reconstructed Linear-K2 artifact has SHA-256
  `bd3f5280957a6418e7d5275a3e7a576fc57872b3ad1bd19ae9ef7bef97d56d3a` and
  reproduces the held-out Phase B Top-2 selections exactly for 16/16 test seeds.

## CPU-only release reproduction

Use a Python environment containing NumPy and Matplotlib:

```bash
python thesis_release/scripts/reproduce_tables.py
MPLCONFIGDIR=/tmp/mattergen-thesis-mpl python thesis_release/scripts/reproduce_figures.py
python thesis_release/scripts/validate_release.py
python -m unittest discover -s thesis_release/tests -v
```

Expected terminal markers are:

```text
TABLE_REPRODUCTION=PASS
FIGURE_REPRODUCTION=PASS
RELEASE_VALIDATION=PASS
```

The table script deterministically recomputes the 20,000-resample Fixed-K2
bootstrap using its frozen bootstrap seed. The figure script reads archived
tables and paired rows. Neither script requires model weights or a GPU.

## Full experiment entry points

The exact archival entry points are under `innovation1/method/` and
`innovation2/method/`. They intentionally preserve scientific execution order
and RNG behavior; some original cluster launch scripts still contain source
environment defaults. Treat those paths as provenance, override locations with
their CLI/environment parameters where supported, and do not use them merely to
regenerate figures.

Re-running C1 or Formal256 is outside the release validation and may require
substantial GPU time. This archive did not rerun C1, Phase B or Formal256 and did
not introduce new seeds or tune any method.

## Expected outputs

- tables: `innovation1/tables/`, `innovation2/tables/`, `combined_summary/*.csv`;
- figures: six numbered figures per innovation, each as PNG/PDF/SVG;
- validation: row-count, finite-metric, status, DFT-label, seed, artifact-hash,
  reference-equivalence, coordinate-map and force-direction checks;
- integrity: all frozen method/config/seed/result files match
  `release_checksums.json`.
