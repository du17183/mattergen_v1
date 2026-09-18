# MatterSim Late Force-Guided Diffusion

## Formal32 FINAL REPORT

========== REPRODUCIBILITY ==========

1. MatterGen HEAD: `1af301c5e534ff6b9b9f96ea5dab48036d9c46af`.
2. Branch: `experiment/mattersim-late-force-guidance-p0`.
3. Base checkpoint path: `/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/hf_mattergen/checkpoints/dft_mag_density/checkpoints/last.ckpt`.
4. Base checkpoint SHA256: `01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e`.
5. MatterSim version: `1.2.3`; checkpoint SHA256 `e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5`.
6. CHGNet version: `0.3.0`; checkpoint SHA256 `d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1`.

========== FROZEN METHOD ==========

7. Guidance window: model time `t<=0.02`; actual 20-point trigger grid `0.020...0.001`.
8. Position-only? **YES**; cell/stress/atomic/energy guidance are OFF.
9. Force source: MatterSim-5M on MatterGen predicted clean x0; raw noisy xt is not evaluated.
10. Force transform: remove translation; force becomes frozen bounded Cartesian displacement; `delta_frac=delta_cart@inverse(clean_cell)` and exact predictor coefficient maps it to score.
11. Trust-region cap: nominal 0.005 Å/atom; hard cap 0.01 Å/atom; protocol absolute cap 0.02 Å.
12. Safety guard: finite clean state, positive clean-cell determinant, candidate periodic minimum distance >=0.5 Å.
13. Fallback rule: use the original unmodified MatterGen score when the safety guard fails.
14. CFG: constant 2.0.
15. Diffusion steps: 1000.
16. Corrector steps: 1 per timestep.
17. Adaptive CFG used? **NO**.

========== SEEDS ==========

18. Seed range: paired fresh `730000-730031`.
19. Historical overlap scan: no exact seed reference in experiments/diagnostics/research/thesis across all worktrees before registration.
20. Number generated: C0 32/32 and F0 32/32; every registered pair is analyzed.

========== PRIMARY ==========

21. C0 Mean MaxF: **0.183472368 eV/Å**.
22. F0 Mean MaxF: **0.134727835 eV/Å**.
23. Absolute improvement C0-F0: **0.048744533 eV/Å**; 95% CI `[0.039327501559600325, 0.05868520359963366]`.
24. Formal relative reduction `(mean(C0)-mean(F0))/mean(C0)`: **26.568%**.
25. Wins/ties/losses for F0: **32/0/0**.
26. Relative-reduction 20,000 paired bootstrap 95% CI: **[14.274%, 52.861%]**.
27. Median MaxF C0/F0: `0.094525/0.043546` eV/Å.
28. P90 MaxF C0/F0: `0.159573/0.127527` eV/Å.
29. P75/P95/max MaxF C0: `0.128404/0.232799/2.831916` eV/Å; F0: `0.065802/0.170824/2.698463` eV/Å.

========== SECONDARY ==========

30. RMSD mean C0/F0: `0.052132/0.048991` Å.
31. Atomic force mean C0/F0: `0.073237/0.048879` eV/Å.
32. E-hull mean C0/F0: `0.039052/0.039232` eV/atom.
33. Stable C0/F0: `93.75%/93.75%`.
34. Novel C0/F0: `25.00%/25.00%`.
35. Unique C0/F0: `59.38%/59.38%`.
36. NUS C0/F0: `18.75%/18.75%`.
37. Validity C0/F0: `100.00%/100.00%`.
38. Mean CHGNet mag density C0/F0: `0.198984/0.198938` Å^-3.
39. Mag MAE C0/F0: `0.006128/0.006148` Å^-3.
40. Mag hit C0/F0: `78.12%/78.12%` at tau=0.01.

========== GUIDANCE BEHAVIOR ==========

41. Guidance attempts: 640 / 640 possible.
42. Accepted: 640.
43. Rejected: 0.
44. Fallback: 0.
45. Mean event-max correction magnitude after the bounded transform: `0.004165896` Å; median `0.005000000` Å; pre-guidance raw mean-force-norm event mean `0.071147677` eV/Å.
46. P95 event-max correction: `0.005000000` Å; max `0.005000000` Å; pre-guidance raw max-force-norm event mean `0.164514682` eV/Å. Event-level raw force and transformed displacement columns are in `guidance_trace.csv`.

========== EFFICIENCY ==========

47. C0 mean end-to-end generation runtime: `86.265` s.
48. F0 mean end-to-end generation runtime: `86.555` s.
49. Runtime ratio F0/C0: `1.00336x`; no acceleration claim is made.
50. Peak VRAM max C0/F0: `363.6/468.7` MiB.
51. MatterGen score calls mean C0/F0: `2000/2000`; MatterSim guidance overhead mean `1.160` s.

========== GUARDRAILS ==========

52. Mag MAE guardrail: `0.327%` worsening; PASS=True.
53. NUS guardrail: drop `0.000` pp; PASS=True.
54. Validity guardrail: drop `0.000` pp; PASS=True.
55. E-hull guardrail: increase `0.000180` eV/atom; PASS=True.
56. Runtime guardrail: ratio `1.00336x`; PASS=True.

========== DECISION ==========

57. MATTERSIM_FORCE_GUIDANCE_FORMAL32 = **STRONG_CONFIRMED**.
58. Primary MaxF criterion passed? **YES**.
59. Paired criterion passed? **YES** (strong threshold separately recorded).
60. Bootstrap criterion passed? **YES** (strong threshold separately recorded).
61. All guardrails passed? **YES**.
62. SURROGATE_PROPERTY_EVAL=True.
63. DFT_VERIFIED=False.
64. NEXT = **FORMAL256**; Recommended Innovation2 = **MATTERSIM_FORCE_GUIDED_DIFFUSION**.
65. Scientific conclusion: on 32 completely fresh paired seeds, the frozen position-only method is `STRONG_CONFIRMED` under the preregistered MaxF and guardrail rules; all claims are surrogate, not DFT.

## Bad-tail and mechanism analysis (non-gating)

Base MaxF versus paired improvement: Spearman rho `0.662757` (p `3.581e-05`); Pearson r `0.608170` (p `0.000222`). Positive correlation means initially worse-force samples benefit more. This analysis does not affect the gate.
