# Innovation 1 trajectory-search audit (2026-09-28)

Scope: read-only audit of the frozen C1 experiment. No frozen source, seed, or result was changed. Host `h20-1`; the executable generation environment is `/mnt/datasets-livsyn/dxl/alm/.venv/bin/python` (Python 3.10.12). The primary model tree is `/mnt/datasets-livsyn/dxl/mattergen_v1`; executable frozen confirmation source/results are in `/mnt/datasets-livsyn/dxl/mattergen_v1_linear_k2_confirm/experiments/frozen_linear_k2_confirmation`. The thesis archive is in this repository under `thesis_release/innovation1/`. The main project's `.venv/bin/python` was not executable in this audit; therefore it must not be assumed to be the runtime.

## Frozen inputs and entry points

| Item | Location / entry |
|---|---|
| C0, Fixed-K2, Random-K2, Linear-K2 | `experiments/frozen_linear_k2_confirmation/run_generation.py` and `confirmatory_sampler.py` in the confirmation worktree |
| Shared-prefix implementation | main project's `experiments/reference_preserved_budgeted_cfg/phase_b_sampler.py` and field-CFG worktree's `mattergen/diffusion/sampling/field_decoupled_cfg.py` |
| Property/validity evaluation | confirmation worktree's `prepare_phase.py` (CHGNet) |
| MatterSim quality | confirmation worktree's `run_quality.py` |
| SAFE-A terminal selection/statistics | confirmation worktree's `analyze_phase.py` |
| Model | main project's `checkpoints/official/hf_mattergen/checkpoints/dft_mag_density/checkpoints/last.ckpt`, SHA-256 `01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e` |
| CHGNet | main project's `.cache/models/chgnet/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar`, SHA-256 `d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1` |
| MatterSim | main project's `checkpoints/official/mattersim/mattersim-v1.0.0-5M.pth`, SHA-256 `e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5` |
| Novelty reference | main project's `data-release/alex-mp/reference_MP2020correction.gz` |
| Frozen seeds | confirmation worktree's `protocol/seed_manifest_256.json`; C1/C2 are disjoint sets of 128, historically overlap-audited |

## Algorithm actually executed

One crystal is sampled per seed, conditioning on `dft_mag_density=0.1`. The official denoising schedule is 1000 predictor–corrector steps at CFG=2.0, two MatterGen score evaluations per step. A standalone full C0 is generated first as a reference replay. The RNG is reset and the C0 prefix is run to step 400; the prefix state and RNG are snapshotted. Its C0 continuation is run from step 400 to 1000. Fixed-K2 then replays that same state/RNG twice, with a 100-step GPulse or PPulse at CFG 1.9 on the specified fields, followed by CFG=2.0 to step 1000. Random-K2 and Linear-K2 run their own two allocated suffixes. Duplicate policy allocations are actually re-executed, not silently reused.

The saved `prefix_checkpoint.pt` contains `batch`, `rng_state`, `branch_point=400`, seed and state digest. `batch` is a clone of the sampler's PyG `ChemGraph` batch and therefore retains the condition through its `replace`/`clone` semantics; `run_generation.py` also fixes the target to 0.1. **The target is not separately serialized or hash-checked inside the checkpoint.** `branch_point` is persisted, but the numerical timestep grid (`linspace(max_t, eps_t, 1000)`) and `dt` are reconstructed from the frozen sampler configuration; **the timestep tensor and schedule/config hash are not explicitly saved in that artifact.** An exact replay requires the same code/config/checkpoint, not merely this `.pt` file. Direct deserialization of the artifact stalled in the current storage/runtime, so conditioning persistence is source-level evidence, not an artifact-level verified assertion.

RNG snapshot/restore in `risk_calibrated_cfg.py` covers Python `random`, NumPy, PyTorch CPU and the current PyTorch CUDA device. The frozen C1 generation summary reports exact standalone-versus-shared C0 reproduction for all 128 seeds. This is stronger than a tolerance-based comparison, but does not itself validate replay after changing hardware/library versions.

SAFE-A permits a candidate only if its CHGNet absolute property error improves on C0 by more than `1e-6`, its evaluation-valid and stable indicators do not worsen, and both E-hull values are finite with candidate E-hull no more than C0 + `0.01 eV/atom`. It selects lowest eligible property error, deterministic policy/rank tie-break; otherwise exact C0. Geometric validity requires positive-enough cell volume and minimum periodic distance >=0.5 Å. Novel, unique and NUS are reported cohort-level guardrails, **not terminal selection inputs**. Property and quality are surrogate estimates, not DFT verification.

## Budget and C1 reproducibility

C0 deployment: `1000×2=2000` score calls. Fixed-K2 deployment: shared prefix `400×2=800`, C0 suffix `600×2=1200`, plus two candidate suffixes `2×1200`; total `4400=2.2×` score calls and three terminal evaluations (C0 + two candidates). The full four-method C1 acquisition also includes standalone C0 and other branch outcomes: `11200=5.6×` score calls per seed. These study-acquisition and per-method deployment budgets must not be conflated. Verifier costs and wall time were not included in the 2.2× figure.

Frozen C1 has 128 complete paired seeds, 896 generated structures and 16.7795 summed GPU-hours of acquisition. Independently recomputing the means from `selected_outcomes.csv` reproduced: C0 `0.03479644444407383`, Fixed-K2 `0.026332355926007578`, Random-K2 `0.02853336177875556`, Linear-K2 `0.027522408495227555`. This is a **result-file recalculation**, not a full regeneration reproducibility test. The stored C1 gate is `FAIL` for Linear-K2, and C2 was not run. Fixed-K2's historical positive C1 result must remain unchanged.

## Audit caveats before new execution

1. Existing prefix artifact is not self-contained for cross-version replay: add an explicit condition, timestep index/value, schedule/config hash and model hash to *new* artifacts only; do not retrofit frozen C1.
2. Independent Best-of-2 costs `2.0×`, whereas Fixed-K2 costs `2.2×` and evaluates three terminals. It is an essential baseline, **not an exact budget/candidate-count matched control**. A win alone cannot logically prove trajectory search rather than extra budget.
3. The same CHGNet/MatterSim estimates drive SAFE-A and reported proxy outcomes, so selection optimism/verifier hacking is possible. Disclose and, if feasible, reserve an independent evaluator; do not label surrogate gains as DFT gains.
4. At audit time GPU 0 (~54.7 GiB) and GPU 1 (~5.4 GiB) had other-user processes. GPUs 2–7 appeared free; recheck ownership and memory immediately before any new worker launch. No user GPU process has been stopped.
