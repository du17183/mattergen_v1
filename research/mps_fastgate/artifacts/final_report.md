# NVIDIA MPS MatterGen fast-gate final report

## Decision

`FINAL_STATE=MPS_NO_GO`

MPS preserved bitwise outputs but changed median throughput from 71.0898 to 70.7855 samples/hour (-0.428%). This is below the frozen 3% engineering gate and the 5% paper gate.

## Frozen protocol

- GPU: NVIDIA RTX PRO 5000 72GB Blackwell, GPU 0
- Driver: 580.126.20; driver-reported CUDA: 13.0; PyTorch CUDA: 12.8
- Model: C0 original MatterGen, batch size 1 per process, FP32, full Predictor/Corrector
- S0: MPS OFF, 2 persistent workers
- S1: MPS ON, 2 persistent workers, 50% active threads per client
- Seeds: 27000-27015; three timed repeats; 48 trajectories/configuration
- Model load and one real forward warm-up excluded from timed windows

## Aggregate result

| Metric | S0 MPS OFF | S1 MPS ON |
|---|---:|---:|
| Median throughput (samples/h) | 71.0898 | 70.7855 |
| Median P50 latency (s) | 100.060 | 100.619 |
| Median P95 latency (s) | 108.421 | 108.077 |
| Mean GPU utilization (%) | 91.198 | 90.812 |
| Peak GPU memory (MiB) | 3195 | 3097 |
| Successful trajectories | 48/48 | 48/48 |
| Failures | 0 | 0 |

S1/S0 incremental speedup: **0.995719x (-0.428%)**.  S1 versus the historical one-worker reference: **1.250149x**.

## S0 repeats

| Round | Throughput (samples/h) | Wall (s) | P50 (s) | P95 (s) | Worker skew est. (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 70.9384 | 811.97 | 100.38 | 108.42 | 8.329 |
| 2 | 71.0898 | 810.24 | 100.06 | 108.59 | 2.471 |
| 3 | 71.2034 | 808.95 | 99.74 | 108.42 | 4.442 |

## S1 repeats

| Round | Throughput (samples/h) | Wall (s) | P50 (s) | P95 (s) | Worker skew est. (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 70.6720 | 815.03 | 100.85 | 108.06 | 13.079 |
| 2 | 70.7855 | 813.73 | 100.62 | 109.01 | 10.601 |
| 3 | 71.0575 | 810.61 | 99.99 | 108.08 | 6.885 |

## Correctness

| Check | Matching pairs | Total pairs |
|---|---:|---:|
| random_tape_hash | 48 | 48 |
| atomic_numbers_hash | 48 | 48 |
| final_structure_hash | 48 | 48 |
| positions_hash | 48 | 48 |
| cell_hash | 48 | 48 |

Within each configuration, all three repeats were also bitwise identical. Scientific outputs were recorded once per configuration; no MatterSim evaluation was needed because raw outputs matched exactly.

## Gate and cleanup

- S1 incremental throughput was below 3%, so S2 and 8-GPU confirmation were not started.
- MPS control/server were stopped cooperatively; project MPS pipe/log directories were cleaned.
- GPU workers after exit: 0; other processes terminated: false; SIGKILL used: false.

## Limitations

- The historical one-worker reference was reused because S0 drift was +7.953%, below the frozen 10% retest threshold.
- The original worker finish-spread field was mislabeled. The tables use a reconstructed per-worker summed-latency skew estimate; primary wall-clock throughput and latency are unaffected.
- This result applies to the tested C0 batch-1 persistent-worker workload and this driver/PyTorch stack.
