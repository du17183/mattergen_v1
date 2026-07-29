# Q3 E3-PCR frozen64 reproduction

## Frozen inputs

- Base commit: `b65f42a8792004c7c820e59fa4413e1310e06143`
- Evaluation seeds: `32000–32063`
- Q3 checkpoint SHA256:
  `b2ce1800fa0fa448f57d58010c8586b5de1b6666c4f198737a2f8a4bfabcb90e`
- Frozen configuration: `configs/q3_e3_pcr_frozen64.json`

The checkpoint, MatterGen/MatterSim weights, environments, and datasets are
not included in Git. Their absolute paths and SHA256 values are recorded in
`frozen_manifest.json`.

## Run

```bash
source /data/dxl/env.sh
cd /data/dxl/mattergen_v1
/data/dxl/tools/q3_e3_pcr/frozen64/resume.sh
```

The pipeline is resumable and validates every successful output before
skipping it. It runs:

```text
freeze manifest
→ 64 single-run C0 generations
→ frozen Q3 and Always-on refinement
→ C0/Q3/Always-on MatterSim relaxation
→ official metrics and paired statistics
→ five fixed Random-gate ablations
→ mechanism analysis
→ frozen64 Go/No-Go
```

Status:

```bash
/data/dxl/tools/q3_e3_pcr/frozen64/status.sh
```

Graceful stop:

```bash
/data/dxl/tools/q3_e3_pcr/frozen64/stop.sh
```

## Tests

```bash
/data/dxl/envs/mattergen_py310/bin/python -m pytest \
  tests/test_q3_frozen64.py \
  tests/test_postgen_refiner.py \
  tests/test_postgen_new_eval.py -q
```

Observed result:

```text
9 passed
```

## Archive

`artifacts/results/` contains the complete non-weight frozen64 result tree,
including generated structures, refined structures, MatterSim outputs,
telemetry, hashes, and atomic progress files. `artifacts/logs/` contains the
small run logs. No checkpoint or environment is included.
