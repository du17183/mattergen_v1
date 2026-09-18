# Frozen method definition

F0 is the exact Formal32/P0 LateMatterSimForceGuidedSampler. Target magnetic density 0.2, constant CFG 2, 1000 steps, one corrector; 20 predictor guidance events at t=0.020..0.001. Clean x0 position forces only; nominal 0.005 Å, hard 0.01 Å, minimum distance 0.5 Å; no weight updates.

A5 alone composes the existing frozen adaptive CFG with F0. F1 is a separate worktree and changes only candidate acceptance/shrinking. F1 epsilon=1e-4 eV/Å; radii 0.005/0.0025/0.00125 Å; up to two retries; high-force threshold 0.2 eV/Å. F1 does not allocate budget across samples or automatically start Formal32.

A4 removes both effective saturation and the redundant hard cap: merely removing 0.01 Å would leave the 0.005 Å saturation intact. The uncapped linear force scale remains 0.005/0.07795149218357911 Å per (eV/Å).

Source hashes and all seed registrations are in config.yaml and experiment_registry.csv. SURROGATE_PROPERTY_EVAL=True; DFT_VERIFIED=False.
