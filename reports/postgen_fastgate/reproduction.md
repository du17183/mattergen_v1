# Reproduction guide

## Environment

```bash
source /data/dxl/env.sh
source /data/dxl/envs/mattergen_py310/bin/activate
cd /data/dxl/mattergen_v1
```

The repository archive intentionally excludes MatterGen, CHGNet, and
MatterSim weights, Conda environments, generated checkpoints, and serialized
scikit-learn models. Configure those paths locally before running the
pipelines.

## Focused tests

```bash
python -m pytest \
  tests/test_postgen_fastgate.py \
  tests/test_postgen_features.py \
  tests/test_postgen_quality_model.py \
  tests/test_postgen_quality_scoring.py \
  tests/test_postgen_setrank.py \
  tests/test_postgen_new_eval.py \
  tests/test_postgen_pareto.py \
  tests/test_postgen_refiner.py -q
```

## Candidate pipelines

The modules expose resumable command-line entry points:

```bash
python -m research.postgen_fastgate.oracle
python -m research.postgen_fastgate.features
python -m research.postgen_fastgate.train_quality
python -m research.postgen_fastgate.evaluate_quality
python -m research.postgen_fastgate.train_setrank
python -m research.postgen_fastgate.new_eval
python -m research.postgen_fastgate.pareto_eval
python -m research.postgen_fastgate.refiner_eval
```

Use `python -m <module> --help` for the exact input/output path arguments.
Each runner skips already-valid outputs and writes summaries atomically.

## Frozen data

The complete non-weight result archive is under:

```text
reports/postgen_fastgate/artifacts/
```

It includes the 128 original MatterGen generation outputs used for the new
32-pool experiments, Q5/Q6/Q3 evaluation outputs, metrics, telemetry, and
logs. Invalid compatibility-probe outputs and all model weights are excluded.
