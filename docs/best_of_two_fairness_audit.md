# Best-of-2 versus Fixed-K2: fairness audit

Date: 2026-09-28. Scope: completed new `search_baseline` cohort only; historical Fixed-K2 C1-128 is untouched. The comparison is paired by 128 primary seeds, uses the same frozen model/target, CHGNet property proxy, MatterSim quality proxy, geometric validity test and SAFE-A terminal rule. Both methods include the same primary C0 terminal and fall back to that exact terminal if no safe candidate is accepted. Independent Best-of-2 adds one **full independent C0** sample; Fixed-K2 adds two **shared-prefix suffix** policies (`GPulse`, `PPulse`). The design and seed manifest preceded generation.

## Exact deployment ledger (per sample)

| Method | MatterGen score calls | Relative to C0 | Terminal candidates/CHGNet selector calls | MatterSim selector calls | Generation wall time |
|---|---:|---:|---:|---:|---|
| C0 | 2,000 | 1.0× | 1 | 1 | **86.90 s estimate**, not isolated measurement |
| Independent Best-of-2 | 4,000 | 2.0× | 2 | 2 | **173.96 s estimate**, not isolated measurement |
| Fixed-K2 | 4,400 | 2.2× | 3 | 3 | **191.17 s estimate**, not isolated measurement |

The 2,000-call C0 full sampler has 1,000 steps × 2 score evaluations. Fixed-K2 uses 800 calls of shared prefix, 1,200 calls of primary C0 suffix, then two 1,200-call candidate suffixes. Independent Best-of-2 uses two independent full samples (2 × 2,000). Thus Fixed-K2 has **10% more generator score calls than Best-of-2**, **50% more terminal candidates and verifier calls**, and an estimated **17.21 s/sample more generation time**. The methods are *not* equal-budget or equal-candidate-count controls; the full wall budget also includes proxy evaluation, which was not individually timed per deployed method.

The measured per-seed generation log is for **study acquisition**, not isolated deployment: the primary acquisition took mean 278.07 s/sample (median 274.09; p95 300.46) for 6,400 calls, including an extra exact C0 replay; the independent second sample took mean 87.06 s (median 85.86; p95 94.01) for 2,000 calls. Combined acquisition took mean 365.13 s/sample, totaling 1,075,200 score calls and 12.9825 summed GPU-hours over 128 seeds. Deployment estimates allocate the measured primary time in proportion to its 6,400 score calls, then add the measured independent run for Best-of-2. This allocation is a transparent approximation: call latency may differ by trajectory segment, GPU contention and evaluator time are excluded, and **no exact per-method end-to-end wall time was measured**. The recorded peak allocated GPU memory during acquisition was 379,476,992 bytes; no isolated deployment peak is available. The four branch candidates per seed were evaluated once for CHGNet/MatterSim selection; three selected-method mixed MatterSim cohorts were evaluated separately for reported quality. These acquisition/re-evaluation calls must not be confused with per-method terminal selector calls.

## Outcomes and compute-normalized descriptive comparison

| Method | Property MAE ↓ | C0−method gain | 20k paired-bootstrap 95% CI | Gain / extra 1.0× generator budget |
|---|---:|---:|---:|---:|
| Independent Best-of-2 | **0.028097** | +0.009527 | `[+0.006439, +0.012891]` | **0.009527** |
| Fixed-K2 | 0.029506 | +0.008117 | `[+0.005533, +0.011054]` | 0.006764 |

The normalization divides paired mean gain over the shared C0 by extra score-call multiples: `0.0095266/(2.0−1.0)` and `0.0081169/(2.2−1.0)`. It is **descriptive, not an exact-budget causal estimate**; it ignores the unequal verifier costs and does not interpolate a measured performance curve between 2.0× and 2.2× in this cohort. The direct paired contrast Independent Best-of-2 minus Fixed-K2 is `−0.001410`, 95% CI `[−0.005039,+0.002167]`, W/T/L `41/59/28` (positive gain favors Fixed-K2). Its uncertainty spans both signs. Independent Best-of-2 has numerically lower mean MAE at lower score-call and verifier budgets; Fixed-K2's distinct advantage over simple independent sampling is **not supported**.

## Remaining comparability and interpretation limits

1. **Candidate count and compute mismatch:** a trajectory-specific advantage requires an exactly score-call- and terminal-count-matched comparator, or an explicit frontier over common total cost. This test answers whether Fixed-K2 clearly beats a serious independent Best-of-2 baseline under the registered asymmetric budget; it does not isolate why either method works.
2. **Selector reuse / verifier hacking:** SAFE-A optimizes the same CHGNet property proxy later reported as primary Property MAE and uses MatterSim for both eligibility and quality outcomes. Test-set proxy improvements are valid descriptions of these pipelines, not independent proof of true DFT property gains. Independent property/quality validation is needed for stronger materials claims.
3. **Quality:** both methods passed the predeclared C0-relative proxy guardrails, but the full aggregate must be reported rather than only stable or MAE. Best-of-2 has Stable 0.625, NUS 0.3516, E-hull 0.09639 eV/atom; Fixed-K2 has 0.6172, 0.3281, 0.10063. Validity is 1.0 for both. Novelty falls relative to C0 in both and is not an acceptance input.
4. **Seed and method integrity:** new baseline primary/child seeds are independent of the historical C1 and the second Budget Scaling cohort; the generator logs report exact C0 replay for all 128 baseline seeds. Historical C1 is neither merged into this paired comparison nor reinterpreted.

Reproduction sources: `experiments/trajectory_search/protocol/seed_manifest.json`, `experiments/trajectory_search/run_generation.py`, `analyze.py`, `results/search_baseline/{compute_accounting.json,paired_results.csv,selected_outcomes.csv,bootstrap_20k.json,table_best_of_n.csv,generation/,quality_branches/,quality_mixed/}`. Derived values: `results/trajectory_search/best_of_2/`.
