# MatterGen Research Exploration Archive

This branch preserves the research process behind the MatterGen thesis extensions, including successful, mixed, negative, inconclusive and abandoned experimental directions.

It is intentionally separated from the thesis-facing release branch to preserve a clean final release while avoiding selective reporting of exploratory work.

## Scope and frozen boundary

- Archive branch: `archive/thesis-exploration-2026`
- Base commit: `5b572c61c70c4287147616beae7f72d28955cf98`
- Thesis-facing frozen results: `release/thesis-final-2026`
- Inventory date: 2026-09-18
- DFT verified: **false**

The final release was not modified. The archive contains exploration-only commits built on top of the same frozen base commit.

## Final thesis conclusions remain unchanged

- Innovation 1 final method: **Fixed-K2 reference-preserved budgeted branching** — `SUPPORTED`.
- Learned Linear-K2: **`NOT_SUPPORTED`**.
- Innovation 2: **RC-NFGD — `SUPPORTED`**.
- DFT verification: **false**; force, relaxation and stability evidence is based on surrogate ML interatomic potentials unless a source explicitly says otherwise.

## What is here

- [Experiment index](EXPERIMENT_INDEX.md): 81 classified experiment records in research-time order.
- [Machine-readable inventory](experiment_inventory.csv) and [human inventory](exploration_archive_inventory.md).
- [Worktree inventory](worktree_inventory.csv), [branch inventory](source_branch_inventory.csv) and [all-ref history](git_all_history.txt).
- `innovation1/`: the path from V1 adaptive CFG through failed robust/risk/safe/stage/field routes to Fixed-K2.
- `innovation2/`: the path from reliability/clean-x0/coordinate mapping through P0, Formal32, Formal256, CHGNet and ablations to RC-NFGD.
- `other_explorations/`: inherited tracked studies, remote-only branches and incomplete residual artifacts.
- [External-data manifest](external_data_manifest.csv): deterministic hashes for 37 raw data trees/checkpoints intentionally kept outside ordinary Git.
- [Selected-artifact manifest](selected_artifact_manifest.csv): hashes and original paths for the 238 small artifacts copied into this archive.

## How to read historical decisions

Source reports retain their original wording and may say `GO`, `FAIL`, `CONFIRMED`, `BORDERLINE` or a route-specific label. Archive landing pages add a separate standardized status:

`SUPPORTED`, `MIXED`, `NOT_SUPPORTED`, `FAIL`, `INCONCLUSIVE`, `ENGINEERING_ONLY`, or `ABANDONED_BEFORE_EVALUATION`.

The standardized status is not written back into historical files. Where later confirmation changed the interpretation, the page explicitly separates the historical conclusion from the final thesis interpretation.

## Reproducibility levels

- `FULL`: the repository contains the frozen implementation/configuration and sufficient small evidence for the documented reproduction path.
- `PARTIAL`: key code, decisions and metrics are archived, while raw structures, checkpoints or exact runtime dependencies remain external.
- `SUMMARY_ONLY`: the archive supplies a provenance-rich decision summary and a stable branch/commit locator.
- `EXTERNAL_ONLY`: only residual raw artifacts survived; hashes are recorded and manual review is required.

This branch does **not** claim that every historical experiment can be reproduced with one command. Each experiment page states its level and missing dependencies.

## Portable paths versus provenance

Absolute `/mnt/datasets-livsyn/...` paths identify the original server artifacts. They are provenance, not portable defaults. Reproduction scripts in the frozen release use repository-relative paths or explicit data-root arguments; restored external data should be mounted under a user-selected root.

## Validation

From the repository root:

```bash
python3 research_archive/scripts/validate_archive.py
python3 research_archive/scripts/verify_hashes.py
python3 research_archive/scripts/secret_audit.py
```

Revalidating every external tree is deliberately opt-in because it reads roughly a gigabyte across thousands of files:

```bash
python3 research_archive/scripts/verify_hashes.py --external
```

See [ARCHIVE_POLICY.md](ARCHIVE_POLICY.md) for inclusion, exclusion and preservation rules.
