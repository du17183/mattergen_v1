# Thesis project closeout and upload policy

This document freezes the repository boundary for the MatterGen master's thesis
archive. It is intended to make the distinction between source evidence, large
non-weight research data, and local runtime state explicit before the final
GitHub push.

## Branch roles

| Branch | Role | Contents |
|---|---|---|
| `writing/thesis-draft-2026` | working integration branch | thesis source, template build, reviewed artwork and closeout checks |
| `release/thesis-final-2026` | thesis-facing release | frozen methods, final evidence, reproducibility scripts and final thesis source |
| `archive/thesis-exploration-2026` | complete research-history archive | successful, mixed, failed, inconclusive and abandoned exploratory routes |

The release branch must contain the final claims only. The archive branch must
retain the path by which those claims were selected, including negative results.
Historical reports are not rewritten to make them appear more successful.

## Upload classes

### Ordinary Git

The following are part of the source archive:

- MatterGen modifications, sampling methods, evaluation code and tests;
- experiment configurations, seed registries, preregistered protocols and
  pipeline status files;
- compact CSV/JSON/TXT result summaries, bootstrap outputs and audit reports;
- figure/table generation scripts and reviewed PDF/SVG/PNG artwork;
- thesis Markdown, LaTeX, the USTB class/template source used by the thesis,
  and deterministic build scripts;
- experiment indexes, source-branch maps, claims/limitations and license notes.

### Release assets, Git LFS or external archival storage

Large non-weight data may be retained and published separately when its license
allows redistribution. This includes raw CIF/EXTXYZ structures, trajectories,
relaxation outputs, large tensors, complete runtime logs and full experiment
trees. Each such bundle must have a manifest containing its original path,
size/count, SHA256, source commit, generation command, license and whether it is
needed for full reproduction. The archive's `external_data_manifest.csv` is the
provenance index for these files.

The normal Git history should not receive a new ordinary blob larger than
100 MB. A GitHub Release or LFS asset is preferred for large non-weight data.

### Deliberately excluded from GitHub

- trained model weights and checkpoints (`*.ckpt`, `*.pth`, `*.safetensors`,
  `*.onnx`, `*.pt` and model `.bin` files), except the small test fixture
  explicitly allowed by `.gitignore`;
- `.TinyTeX/`, virtual environments, package caches and Python/Matplotlib
  caches;
- LaTeX auxiliary files (`*.aux`, `*.log`, `*.toc`, `*.lof`, `*.lot`, `*.out`,
  `*.synctex.gz`) and temporary root previews;
- credentials, tokens, private service configuration and machine-specific
  temporary files;
- third-party datasets that cannot be redistributed under their license.

Excluded model weights are listed in a weight manifest with their source,
SHA256, size, license and expected local path. Excluded raw data are represented
by hashes and manifests rather than silently disappearing.

## Frozen scientific endpoint

- Innovation 1: Fixed-K2 reference-preserved budgeted branching is
  `SUPPORTED`; learned Linear-K2 is `NOT_SUPPORTED`.
- Innovation 2: RC-NFGD is `SUPPORTED` on the registered surrogate-model
  evidence.
- `DFT_VERIFIED=false`.
- Reliable synergy between the two contributions is not supported.

Adaptive CFG, risk-calibrated, stage-calibrated and field-decoupled routes stay
visible in the exploration archive with their original negative or mixed status.
ALM-GEN, the composable adapter, LoRA and RL residuals remain
`INCONCLUSIVE`/`EXTERNAL_ONLY` and are not thesis claims.

## Closeout checks

Run the following from the repository root after the final source and artwork
selection:

```bash
python thesis_release/scripts/reproduce_tables.py
MPLCONFIGDIR=/tmp/mattergen-thesis-mpl python thesis_release/scripts/reproduce_figures.py
python thesis_release/scripts/validate_release.py
python -m unittest discover -s thesis_release/tests -v
python3 research_archive/scripts/validate_archive.py
python3 research_archive/scripts/verify_hashes.py
python3 research_archive/scripts/secret_audit.py
./thesis/latex/build_ustb_preview.sh
./thesis/latex/build_ustb_polished_preview.sh
git diff --check
```

The LaTeX scripts use the local `.TinyTeX/` only when it exists. On a clean
checkout, set `TEXROOT` to an authorized TeX Live/TinyTeX installation or put
`xelatex` on `PATH`; the tool installation itself is not part of the archive.

No generation, model evaluation, GPU job or post-result parameter tuning is
performed by these closeout checks.
