# SPG-MatterGen Fast Gate

This package runs the frozen feasibility gate for shape-specialized MatterGen
inference. It evaluates C0 and A0 native batches, profiles periodic graph and
GemNet execution, probes field-safe BF16, audits local `torch.compile`, and
performs paired MatterSim quality evaluation. It does not implement the static
graph engine or launch the 256-seed formal experiment.

Run from `/data/dxl/mattergen_v1` with the project Python environment:

```bash
bash research/spg_fastgate/scripts/run_fastgate.sh
bash research/spg_fastgate/scripts/status_fastgate.sh
```

The pipeline is resume-safe. Successful task directories are never rerun.
`stop_fastgate.sh` writes a cooperative stop marker; it does not signal or kill
processes. Large traces, generated structures, model weights, datasets, and
MatterSim caches remain outside Git under `/data/dxl`.

The frozen seed ranges are:

- performance: 24000-24015;
- B4 quality equivalence: 24064-24127;
- BF16 endpoint probe: 24128-24135.
