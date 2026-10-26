# Adaptive Neighborhood GemNet-T P0 — FINAL

1. branch: `experiment/adaptive-neighborhood-p0`.
2. HEAD: `1af301c5e534ff6b9b9f96ea5dab48036d9c46af`.
3. base checkpoint: `/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/hf_mattergen/checkpoints/dft_mag_density/checkpoints/last.ckpt` (Official `dft_mag_density` Full FT).
4. checkpoint SHA-256: `01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e`.
5. exact code modification: one shared normalized-time MLP `1→16→3`; `1+0.5*tanh` gates Near/Mid/Far messages only at each block's `AtomUpdateBlock` aggregation input. Graph topology, 7 Å cutoff, radial basis, atom embedding, and output heads are unchanged.
6. trainable parameters: 83 (gate only).
7. trainable ratio: 0.00017022% of 48760526 total parameters.
8. training steps: 1000 (early stopped: False).
9. runtime: diagnostic generation 3045.27s cumulative + sensitivity 119.89s; training 83.83s; generation mean B0/A0 88.20/87.00s.
10. peak VRAM: diagnostic 354.98 MiB; training 2400.45 MiB; generation mean B0/A0 278.22/278.55 MiB.
11. seeds: diagnostic 94000–94031; independent paired held-out 96000–96031; training/validation 20260923/20260924.
12. Base property metrics: CHGNet surrogate Mag MAE=0.010081, Hit=23/32 (71.88%).
13. New method property metrics: Mag MAE=0.008990, Hit=21/32 (65.62%); paired not-worse=15/32.
14. Base RMSD: mean=0.063057 Å, median=0.003000 Å.
15. New RMSD: mean=0.063385 Å, median=0.002878 Å; relative improvement=-0.52%.
16. Base MaxF: mean=0.116673 eV/Å, median=0.080497 eV/Å.
17. New MaxF: mean=0.139547 eV/Å, median=0.096255 eV/Å; relative improvement=-19.61%.
18. E-hull: B0/A0 mean=0.045783/0.056452 eV/atom.
19. Stable: B0/A0=87.50%/93.75%.
20. Novel: B0/A0=43.75%/25.00%.
21. Unique: B0/A0=75.00%/62.50%.
22. NUS: B0/A0=31.25%/18.75% (delta -12.50 pp).
23. paired result: MaxF wins/ties/losses=17/0/15; RMSD=16/1/15; property not-worse=15/32.
24. mechanism diagnostic: 32 trajectories × 6 timesteps × 4 blocks × 3 bands = 2304 rows; all four motivation conditions passed. Final gates `[near,mid,far]` at t=0.80 are [0.979673, 0.998479, 1.002081], and at t=0.02 are [0.99068, 1.005369, 1.003729]; stage-pattern check=True.
25. GO/BORDERLINE/FAIL: **FAIL** (geometry GO=False; property GO=False).
26. Adaptive CFG used? **NO**.
27. `DFT_VERIFIED=False`; `SURROGATE_PROPERTY_EVAL=True` (frozen evaluator calibration MAE 0.001449, Pearson 0.9499, tau 0.01).
28. Final one-sentence conclusion: Stage-dependent distance sensitivity exists, but the 83-parameter gate does not deliver the preregistered held-out gain, so A is not retained.
