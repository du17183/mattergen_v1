# Multi-field Self-Conditioned MatterGen P0 — FINAL

1. branch: `experiment/multifield-self-conditioning-p0`.
2. HEAD: `1af301c5e534ff6b9b9f96ea5dab48036d9c46af`.
3. base checkpoint: `/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/hf_mattergen/checkpoints/dft_mag_density/checkpoints/last.ckpt` (Official `dft_mag_density` Full FT).
4. checkpoint SHA-256: `01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e`.
5. exact code modification: detached prior clean estimates are encoded as soft atom probabilities × frozen atom table, periodic sin/cos position deltas through `6→64→512`, and atom-count-normalized cell deltas through `9→64→512`; three zero-initialized time gates sum the fields once before the GemNet block stack. Sampling caches the prior score call and adds no forward.
6. trainable parameters: 26652917 (SC modules + GemNet block 4 + GemNet output/lattice heads + atomic head).
7. trainable ratio: 54.5760% of 48836350 total parameters.
8. training steps: 1000 (early stopped: False; mean SC fraction 0.5109).
9. runtime: training 150.24s; generation mean B0/S-off/S-on=87.76/86.92/90.94s.
10. peak VRAM: training 4660.45 MiB; generation mean B0/S-off/S-on=286.18/285.73/286.14 MiB.
11. seeds: paired held-out 95000–95031; training/validation 20260921/20260922.
12. Base property metrics: CHGNet surrogate Mag MAE=0.011462, Hit=19/32 (59.38%).
13. New method property metrics: S-off MAE=0.011392, Hit=20/32; S-on MAE=0.012644, Hit=17/32, paired not-worse vs B0=14/32.
14. Base RMSD: mean=0.083624 Å, median=0.004762 Å.
15. New RMSD: S-off/S-on mean=0.043469/0.077344 Å; S-on improvement vs B0=7.51%.
16. Base MaxF: mean=0.221522 eV/Å, median=0.115289 eV/Å.
17. New MaxF: S-off/S-on mean=0.290829/0.280991 eV/Å; S-on improvement vs B0=-26.85%.
18. E-hull: B0/S-off/S-on mean=0.088053/0.130831/0.109235 eV/atom.
19. Stable: B0/S-off/S-on=71.88%/78.12%/68.75%.
20. Novel: B0/S-off/S-on=46.88%/59.38%/59.38%.
21. Unique: B0/S-off/S-on=68.75%/90.62%/87.50%.
22. NUS: B0/S-off/S-on=18.75%/37.50%/25.00%.
23. paired result: S-on vs B0 MaxF wins/ties/losses=17/0/15; S-on vs S-off MaxF=12/0/20; property not-worse vs B0=14/32.
24. mechanism diagnostic: zero-init maximum output difference=0.000e+00; final field gate max |g|=0.048979, time spans={'g_atom': 1.877546310430009e-05, 'g_cell': 0.00018164142966279467, 'g_pos': 1.9332859665199646e-05}; final t=0.80 gates=[0.017741, -0.007579, -0.048797], t=0.02 gates=[0.01776, -0.007561, -0.048979]; S-on superiority={'mag_mae': False, 'max_force': False, 'rmsd': False}.
25. GO/BORDERLINE/FAIL: **FAIL** (quality GO=False; property GO=False).
26. Adaptive CFG used? **NO**.
27. `DFT_VERIFIED=False`; `SURROGATE_PROPERTY_EVAL=True` (frozen evaluator calibration MAE 0.001449, Pearson 0.9499, tau 0.01).
28. Final one-sentence conclusion: S-on does not establish the preregistered advantage over both B0 and S-off, so multi-field self-conditioning is not retained.
