# 最终实验总审计

| # | Check | Answer | Evidence / boundary |
|---|---|---|---|
| 1 | Innovation1 Formal evidence complete? | YES, WITH MIXED EFFECT EVIDENCE | 256 pairs and original directional gate. Primary CIs cross zero; minimal8 quality direction not replicated. Historical missing property/pre-relax-force fields explicitly NA. |
| 2 | Innovation2 Formal256 complete? | YES | 256 pairs; raw results and frozen primary bootstrap reproduced. |
| 3 | Independent MLIP validation complete? | YES | 512 formal structures; raw vectors reproduce MaxF and mean force; mean/median/P90/P95/max and paired CIs exported. |
| 4 | Force direction ablation complete? | YES | 32 pairs, five arms, timing/norm/acceptance checked. Reference-trajectory limitation retained. |
| 5 | Bounded correction ablation complete? | YES | Negative necessity result retained. |
| 6 | Equal-budget POST baseline complete? | YES | 32 fresh paired seeds from existing A6; actual calls 20 vs20; POST stronger on MatterSim. |
| 7 | Adaptive CFG + F0 interaction complete? | YES | 64 pairs; original retention FAIL preserved. |
| 8 | E3-G baseline audit complete? | YES | Formal256 raw seed remapping candidate_seed validated; original bootstrap reproduced; missing runtime NA. |
| 9 | DFT attempted? | ENVIRONMENT AUDIT ONLY | Actual binary/pseudopotential/workflow audit completed; no DFT calculation was launched. |
| 10 | DFT completed? | NO / OPTIONAL | No solver or production pseudopotential/workflow; n=0; does not block thesis experimental closure. |
| 11 | All seed overlaps checked? | YES | Fresh scans and pairwise final cohorts checked; intentional C0/evaluator reuse and I1-vs-later-gate training overlap disclosed. |
| 12 | Any cherry-pick? | NO | All registered samples retained, including losses and negative ablations; no replacement seeds. |
| 13 | Any primary endpoint changed? | NO | Original endpoints, thresholds and scientific decisions preserved. |
| 14 | All bootstrap settings consistent? | WITHIN EACH FROZEN PROTOCOL; NOT IDENTICAL ACROSS HISTORY | Historical RNG seeds differ; I1 original RNG/count not recoverable. New 20k seed20260915 supplementary analyses explicitly separate; no silent substitution. |
| 15 | All final figures generated? | YES | 12 required PNG plus PDFs and supplementary minimal8 plot verified. |
| 16 | All final tables generated? | YES | Full precision CSVs, readable Markdown, complete paired and distribution statistics. |
| 17 | Reproducibility manifest complete? | YES WITH DISCLOSED HISTORICAL LIMITS | Available sources, hashes, frozen configs, versions, seed lists, metric/statistical definitions all recorded. Unknown historical metadata not fabricated. |
| 18 | Any missing thesis-critical experiment? | NO | n=8 component diagnostic completed and trace-reproduced. Optional DFT unavailable; unavailable legacy property fields restrict claims, not fabricated as completed tests. |

THESIS_EXPERIMENTS_COMPLETE=YES

该标志仅表示当前冻结论文主线所需的实验、统计和素材已完成，绝不表示所有假设都获得支持，也不表示历史缺失字段被补成了真实测量。历史 I1 的方向性确认、置信区间跨零、E3 元数据缺项及可选 DFT 未运行均完整披露。
