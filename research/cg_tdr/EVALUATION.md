# CG-TDR Phase-0 evaluation

The evaluation is frozen at `stop_for_review`.

## Reproduction commands

Use `/data/dxl/envs/mattergen_py310` from the repository root.

```bash
python -m research.cg_tdr.compare_test_baselines
python -m research.cg_tdr.diagnose_v1
python -m research.cg_tdr.experiment_generation launch --mode eight
python -m research.cg_tdr.experiment_relax launch --mode eight --workers 16
python -m research.cg_tdr.analyze_v1
python -m research.cg_tdr.build_gate_v2_labels
python -m research.cg_tdr.train_gate_v2 --max-steps 1500 --training-seed 3101
python -m research.cg_tdr.experiment_generation_v2
python -m research.cg_tdr.experiment_relax_v2 launch --workers 16
python -m research.cg_tdr.analyze_v2
python -m research.cg_tdr.finalize_eval
```

Every generation and relaxation launcher skips an existing successful task.
The V1 evaluator strictly uses the frozen `best.pt` at step 100. Gate V2
uses training seed 3101 and its frozen best checkpoint at step 1100.

## Decision

```text
CG_TDR_GATE_V2_VALID=True
CG_TDR_V2_EIGHT_SEED_GO=False
CG_TDR_MVP_GO=False
CG_TDR_MVP_NO_GO=True
CG_TDR_ROUTE_STOPPED=True
THIRTY_TWO_SEED_STARTED=False
```

Gate V2 repaired the near-always-on confidence outputs, but neither V2P nor
V2C reached a frozen positive quality threshold. The route stops after the
single allowed Gate V2 repair.
