# SPG-MatterGen single-bucket static periodic graph MVP — final report

```text
SPG_STATIC_MVP_GOAL_COMPLETED:True

FINAL_TERMINATION_STATE:SINGLE_BUCKET_NO_GO

CORRECTNESS_MILESTONE_COMMIT:1c5b4468d91814ec79daede83eba16f75f8c5fa8
PERFORMANCE_COMMIT:251213cfc2fcf3b8fa102f9c4ba6deb623c9ffcb
FINAL_COMMIT:the branch head containing this report; resolve with git rev-parse HEAD

SELECTED_BUCKET:atoms_8_12_rep_le_2 (8–12 atoms, periodic repetitions <=2)
BUCKET_STATE_COVERAGE:33.6578125%
BUCKET_SAMPLE_COVERAGE:62.5%
PADDING_WASTE:atoms=15.774956%; candidate_pairs=37.590180%

VALIDATION_STATES:10000
EDGE_SET_MATCH_RATE:100%
EDGE_ORDER_MATCH_RATE:100%
OFFSET_MATCH_RATE:100%
TRIPLET_SET_MATCH_RATE:100%
TRIPLET_ORDER_MATCH_RATE:100%

GEMNET_VALIDATION_STATES:64
BLOCK1_MAX_ERROR:0
BLOCK2_MAX_ERROR:0
BLOCK3_MAX_ERROR:0
BLOCK4_MAX_ERROR:0
ATOMIC_SCORE_MAX_ERROR:0
POSITION_SCORE_MAX_ERROR:0
CELL_SCORE_MAX_ERROR:0
JOINT_CFG_MAX_ERROR:0
GEMNET_NUMERICAL_EQUIVALENT:True
JOINT_CFG_NUMERICAL_EQUIVALENT:True

EDGE_CAPACITY_UTILIZATION_P50:79.872204%
EDGE_CAPACITY_UTILIZATION_P95:96.485623%
TRIPLET_CAPACITY_UTILIZATION_P50:76.228387%
TRIPLET_CAPACITY_UTILIZATION_P95:92.679438%
INVALID_EDGE_COMPUTE_SHARE:0%
INVALID_TRIPLET_COMPUTE_SHARE:0%

DYNAMIC_BUILDER_TIME:4.683872 ms
STATIC_BUILDER_TIME:2.666656 ms
STATIC_BUILDER_SPEEDUP:1.756459x
STATIC_BUILDER_PERFORMANCE_GO:False

DYNAMIC_FORWARD_TIME:26.952528 ms
STATIC_FORWARD_TIME:27.972800 ms
BUCKET_FULL_FORWARD_SPEEDUP:0.963526x
BUCKET_FULL_FORWARD_PERFORMANCE_GO:False

ESTIMATED_GLOBAL_SPEEDUP:0.987419x
ESTIMATED_TWO_BUCKET_SPEEDUP:0.975151x
ESTIMATED_THREE_BUCKET_SPEEDUP:0.963526x

EIGHT_SEED_STARTED:False
DYNAMIC_GENERATION:NOT_RUN_GATE_FAILED
STATIC_GENERATION:NOT_RUN_GATE_FAILED
DYNAMIC_MEDIAN_TIME:NOT_RUN_GATE_FAILED
STATIC_MEDIAN_TIME:NOT_RUN_GATE_FAILED
END_TO_END_SPEEDUP:NOT_EVALUATED

BUCKET_HIT_RATE:NOT_EVALUATED_END_TO_END
FALLBACK_RATE:0% in 10,000 strict states and 64 numerical states
FALLBACK_REASONS:none in validated bucket states

STRUCTURE_HASH_MATCH:NOT_EVALUATED_8_SEED
EIGHT_SEED_QUALITY_SAFE:NOT_EVALUATED

SINGLE_BUCKET_STRONG_PASS:False
SINGLE_BUCKET_TECHNICAL_PASS:False
SINGLE_BUCKET_NO_GO:True
HARD_BLOCKED:False

COMPILE_STARTED:False
CUDA_GRAPH_STARTED:False
A0_STARTED:False
SIXTY_FOUR_SEED_STARTED:False
FORMAL_256_STARTED:False

FINAL_REPORT:/data/dxl/reports/spg_static_mvp/final_report.md
NUMERICAL_REPORT:/data/dxl/reports/spg_static_mvp/numerical_equivalence.md
CAPACITY_REPORT:/data/dxl/reports/spg_static_mvp/capacity_analysis.md
BUILDER_BENCHMARK_REPORT:/data/dxl/reports/spg_static_mvp/builder_benchmark_optimized.md
FORWARD_BENCHMARK_REPORT:/data/dxl/reports/spg_static_mvp/forward_benchmark.md
EIGHT_SEED_REPORT:NOT_CREATED_GATE_FAILED

GITHUB_BRANCH:feature/spg-static-periodic-graph-mvp
GITHUB_COMMIT:branch head after final archive push
DRAFT_PR:https://github.com/du17183/mattergen_v1/pull/8

GPU_WORKERS:verified as 0 after final push
OTHER_PROCESSES_TERMINATED:False
SIGKILL_USED:False
```

## Final interpretation

The single-bucket static periodic graph path is semantically exact. All 10,000
strict graph states match the eager implementation, and all 64 stratified
joint-CFG states are bitwise identical through GemNet blocks 1–4 and the final
atomic, position, and cell scores. This is not a correctness No-Go.

The performance hypothesis fails. After exactly one profiler-supported compact
packing optimization, the builder reaches only 1.756459× rather than the
required 2.25×. More importantly, the complete static joint-CFG score forward
is 0.963526× the dynamic baseline, about 3.79% slower. Static execution reduces
kernel launches (3,011.0 to 2,626.7), allocator calls (800.0 to 625.0), and
synchronization calls (250.1 to 144.1), but exact fixed-workspace
indexing/scatter/topology packing costs erase those savings.

Padding is not the root cause: GemNet receives compact valid edge and triplet
tensors, and the invalid edge/triplet message-passing shares are both zero.
Because both frozen performance gates fail, the protocol correctly prevents an
8-seed run; no generation-quality conclusion is inferred from an experiment
that was not authorized to start.

## Validation and test status

- SPG specialized plus GemNet regressions: 16/16 passed.
- Optimized strict graph equivalence: 10,000/10,000 passed, 0 fallback,
  maximum distance/vector error 0.
- Optimized joint-CFG/GemNet numerical equivalence: 64/64 passed, 100% bitwise,
  maximum error 0.
- Full repository suite with `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`: 165/170
  passed. The five failures are unchanged from `main`: four legacy Corrector
  tests omit the now-required `dt`, and one training test passes the removed
  `ReduceLROnPlateau(verbose=...)` argument. No SPG test fails.
- `git diff --check`: passed.

## Limitations

The full-forward test covers one frozen bucket and FP32 batch-size-one joint
CFG. It does not test BF16, `torch.compile`, CUDA Graph, multiple trajectories,
or custom fused kernels because those were explicitly outside this MVP.
Two/three-bucket values are Amdahl projections under equal-speedup,
non-overlapping coverage assumptions, not measurements.

## Next action

Keep SPG as a rigorous negative ablation and do not expand the current
fixed-workspace builder to additional buckets: the measured full-forward ratio
is already negative, so simple coverage expansion makes the projected outcome
worse. For a new exact-quality acceleration track, benchmark optimizations that
operate on the whole denoiser forward (for example `torch.compile`, CUDA Graph,
or a fused scatter/indexing path) behind independent gates rather than stacking
them onto this failed single-bucket implementation.
