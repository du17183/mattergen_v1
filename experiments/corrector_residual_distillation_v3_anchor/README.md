# Corrector Residual Distillation V3: Periodic Exact Anchor

This experiment adds only a periodic exact second-forward anchor to the frozen V2
policy. The registered choices are K=4, 8, and 16; V2 is K=infinity. Late exact
(progress >= 0.7) has highest priority, followed by frozen atomic-risk fallback,
then the periodic anchor. Every exact call resets the eligible Adapter streak.

Seed intervals are disjoint: smoke 67900-67903, Stage-B 68000-68031, and
Stage-C 69000-69063. Formal256 seeds 67000-67255 are permanently frozen and are
never used for V3 tuning.

The official quality evaluator uses the identical frozen reference LMDB,
decompressed once into sealed shared anonymous RAM. This avoids slow random
reads from the network filesystem; no reference entries or metric definitions
are changed. Model weights, environments, and persistent outputs remain here.

Final single-H20 measurements run after all quality jobs finish, using GPU0 and
paired seeds 68000-68003 (Stage-B), or 69000-69015 (Stage-C). The initial Stage-B
speed attempt overlapped a diagnosed network-I/O stall and was archived as a
whole under stage_b/single_h20/attempt_io_contention; it is excluded from the
final speed statistics. Its early completed samples are retained, not selected
between reruns. Sampling seeds and the configured K candidates are unchanged.

Stage-B completed: 160/160 generation and MatterSim/official evaluations, plus
20/20 final single-GPU runs. K16 is selected for independent validation: paired
speedup 1.279190 (95% bootstrap CI 1.215288-1.363417), force P95 0.643321 vs V2
0.711258, P99 0.779622 vs 1.567998, and RMSD 0.020591 vs 0.063817 angstrom.
Both V2 and K16 have zero force>2 cases, so Stage-B cannot establish improvement
for that rare event. K16 Stable is 59.375%, equal to C0 but below V2's 68.75%;
this remains an explicit independent-validation risk. Primary quality intervals
are wide and are not interpreted as proof of non-inferiority.

K4 is rejected because Stable falls to 46.875% (C0 59.375%). K8 is rejected for
the target tail question: its P95 rises to 1.029375 and force>1 increases from
1/32 to 2/32 versus V2 despite strong primary quality. No further K search is
permitted. Stage-C tests only C0, frozen V2, and frozen K16 on 69000-69063.

The V2 longest-streak/force correlation is negative in Stage-B (Spearman
rho=-0.6560). Thus these data do not support a simple longer-streak-causes-higher-
force explanation, even though the anchor limits streaks and K16 improves tail
point estimates. Reference results remain MatterSim-5M surrogate evaluations.

Stage-C is complete: 192/192 independent generation and official MatterSim
evaluations, plus 48/48 single-GPU runs on the registered 16 paired seeds.
K16 speedup is 1.299246 (95% paired bootstrap CI 1.251428-1.349452), but its
force P95/P99 worsen to 1.176992/1.828578 from V2's 0.871651/1.309170.
Force>2 rises from 0/64 to 1/64; maximum-force mean rises from 0.261336 to
0.339003 eV/angstrom. K16 Stable/NUS are 54.6875%/26.5625%, versus C0's
64.0625%/34.3750%; RMSD is 0.076131 versus C0 0.042636 angstrom.
Primary-quality and rare-event intervals remain wide; this is failure of the
registered advancement criterion, not a claim that every difference is
statistically significant. Full paired intervals are in stage_c/paired_statistics.csv.

Final decision: FAIL. Do not advance to a new formal256 or retune K/thresholds.
No further training, generation, or formal256 has been started. The supervisor
exited normally after stage_complete. See final_report.md for full Stage-B/C
tables, timing boundaries, streak analysis, failure cases, and the recommendation
to stop this acceleration route and separately consider a quality-first adapter.
