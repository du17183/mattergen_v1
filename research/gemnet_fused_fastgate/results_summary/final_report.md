# GemNet GPU acceleration fast-gate final report

## Final decision

`GPU_ACCELERATION_NO_GO`

Neither of the two authorized routes met its frozen performance gate. The result is
a clean engineering No-Go, not a quality failure: all persistent-runtime samples
completed and remained bitwise identical across concurrency levels.

## Route 1: one GemNet hotspot chain

- Original model: C0, batch size 1, strict FP32, full Predictor/Corrector, joint CFG.
- Selected unique chain: K2 AtomUpdate/OutputBlock dense-gate-scatter-residual family.
- Measured forward share: 31.8346%, nine calls per full joint-CFG forward.
- Implementation: reversible local `torch.compile` on only the nine K2 modules.
- Full GemNet and sampler remained eager; native edge ordering and scatter semantics stayed unchanged.

### Strict numerical validation

- Real states: 100, spanning atom/edge/triplet shapes and sampling phases.
- Tolerance: `atol=1e-6`, `rtol=1e-5`, cosine >= 0.999999.
- Maximum absolute error: 0.0009765625.
- Maximum aggregate relative L2 error: 1.50667708872937e-6.
- Minimum cosine: 0.9999999996569209.
- First observed failing component: `int_blocks.3.atom_update`.
- Final position score also failed strict allclose.
- Decision: `NUMERICAL_EQUIVALENT=False`; tolerances were not relaxed.

### Performance

| Gate | Original | Candidate | Speedup | Required | Result |
|---|---:|---:|---:|---:|---|
| K2 chain, CUDA median | 8.335496 ms | 7.415867 ms | 1.124008x | 1.25x | Fail |
| Full joint-CFG forward, CUDA median | 31.541383 ms | 31.255368 ms | 1.009151x | 1.08x | Fail |

Kernel/device-event count for the K2 replay fell from 893 to 692, but saved launch
overhead was insufficient at representative dynamic shapes. Because numerical,
chain, and full-forward gates all failed, seeds 26000-26007 were not started.

## Route 2: persistent multi-trajectory B1 runtime

All levels used the same 32 seeds (27000-27031), batch size 1 per process,
independent RNG state/CUDA context, strict FP32, original dynamic graph, and the
complete Predictor/Corrector sampler. Model load and warmup were excluded from the
measured task window.

| Workers/GPU | Total workers | 8-GPU samples/hour | Speedup | Median latency | Mean GPU util | Peak memory/GPU | Success | Bitwise |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 8 | 452.973 | 1.000x | 60.340 s | 39.995% | 1420 MiB | 32/32 | True |
| 2 | 16 | 526.820 | 1.163x | 104.421 s | 87.104% | 2491 MiB | 32/32 | True |
| 4 | 32 | 535.090 | 1.181x | 195.303 s | 81.849% | 4856 MiB | 32/32 | True |

The best setting was four workers/GPU, but 1.181x is below the frozen 1.25x
throughput gate. There were zero failed attempts and all 96 outputs matched the
one-worker reference seed-by-seed at the raw atomic-number, position, and cell
tensor-byte level.

## Interpretation

The GPU becomes effectively saturated at two workers/GPU: utilization rises from
about 40% to 87%, while contention doubles latency. Four workers add only 1.6%
throughput over two workers and nearly double latency again. This establishes a
practical concurrency ceiling for this workload and hardware.

The local compiler reduces K2 launches but cannot convert the 31.8% inclusive
module share into a 25% chain win or an 8% full-forward win without changing
floating-point behavior. A custom Triton kernel was therefore not justified under
the one-primary-plus-one-correction limit.

## Frozen flags

```text
FUSED_KERNEL_GO=False
FUSED_KERNEL_NO_GO_RUNTIME_PASS=False
GPU_ACCELERATION_NO_GO=True
OTHER_PROCESSES_TERMINATED=False
SIGKILL_USED=False
```

No weights, environments, datasets, structure caches, large logs, or Nsight trace
files are included in the Git branch.
