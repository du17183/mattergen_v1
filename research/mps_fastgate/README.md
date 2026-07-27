# MatterGen NVIDIA MPS fast-gate

This benchmark compares the existing C0 persistent-worker runtime with NVIDIA
MPS disabled and enabled. It does not change the model, sampler, precision,
batch size, random tape, or scientific output.

## Frozen configurations

- `S0_off_w2`: GPU 0, MPS off, two persistent workers.
- `S1_mps_w2_p50`: GPU 0, MPS on, two persistent workers, 50% active threads.
- Seeds `27000-27015`, three timed repeats per configuration.
- C0 original MatterGen, batch size one per process, strict FP32, complete
  Predictor/Corrector sampling.

`S2` and the eight-GPU confirmation are conditionally run only when S1 is at
least 5% faster than S0 and remains bitwise equivalent.

## Run

```bash
source /data/dxl/env.sh
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate /data/dxl/envs/mattergen_py310
cd /data/dxl/mattergen_v1
bash research/mps_fastgate/scripts/run.sh
```

Status and cooperative stop:

```bash
bash research/mps_fastgate/scripts/status.sh
bash research/mps_fastgate/scripts/stop.sh
```

The MPS controller uses project-isolated pipe and log directories under
`/data/dxl/results/mps_fastgate/mps_runtime`. It refuses to start when another
MPS service already exists and never changes GPU compute mode.

## Frozen result

The completed run is `MPS_NO_GO`: S1 preserved bitwise outputs but was 0.428%
slower than S0 by median wall-clock throughput. S2 and eight-GPU confirmation
were therefore not started. Small, non-weight result artifacts are stored in
[`artifacts`](artifacts/).
