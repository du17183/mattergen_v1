# Innovation 2 — Next-Round Final Summary

| Candidate | Verdict | Key Evidence |
|---|---|---|
| Site-level convergence | FAIL | validation SiteSpread>=0.10: 0.0%; median spread 0.0000; range gate 0.0%; stable relations none; species/noise driven=False/False |
| Cell–Atom Fusion | FAIL | identity max diff 0; best val total improvement -0.0032%; FUSION_UNUSED=True; smoke Mag MAE 0.00940->0.00705, MaxF 0.1062->0.0931, NUS 12.5%->6.2% |

## Recommendation

`Recommended Innovation2 = E3_PCR_FALLBACK`

The frozen selection logic was applied without lowering any historical gate: Cell–Atom Fusion has priority only after GO; otherwise a site-level diagnostic GO advances only to a future, separate Site-level Adaptive Diffusion test; if neither route is GO, the project falls back to the already successful E3-PCR Formal256 evidence.

## A. Site-level Position Convergence Diagnostic

- Existing aggregate trajectories lacked per-site clean estimates, so `REUSED_EXISTING_TRAJECTORIES=False`. The same 64 seeds were deterministically replayed; all final structures matched exactly, and existing final quality/per-atom MatterSim forces were reused.
- Each replay passively saved 1000 clean estimates with no extra forward, while the formal analysis used the original frozen 55-point schedule. Every trajectory retained 2000 score calls.
- Threshold `0.07655689 Å`, K=3, was selected only from calibration32; validation was not used.
- Final A verdict: `SITE_LEVEL_HETEROGENEITY=False`, `SITE_LEVEL_DIAGNOSTIC=FAIL`, `NEXT=STOP_ASYNC_FAMILY`.

Detailed A report: `/mnt/datasets-livsyn/dxl/mattergen_v1_async_freeze/experiments/site_level_convergence_diagnostic/final_report.md`.

## B. Global Cell–Atom Bidirectional Fusion

- Fixed insertion: after GemNet Block3 and before Block4. Row-lattice convention uses the invariant `L L^T` Gram; no timestep gate or alternate insertion sweep was used.
- Official weights: 48,760,443; fusion/trainable parameters: 330,370 (0.6775%). Only fusion parameters were trained for 1000 steps at LR `1e-3`; the official-state digest was unchanged.
- Zero-init full-model identity passed exactly for atomic, position, and cell outputs (all max absolute differences 0, tolerance `1e-5`).
- Best fixed-noise validation step: 100; total improvement -0.0032%; validation GO signal `False`. `FUSION_UNUSED=True` at the selected checkpoint.
- The requested 16 fresh paired seeds were evaluated. Smoke verdict: `FAIL`; `FORMAL32_ELIGIBLE=False`; next `STOP`. No Formal32/Formal256 or ablation was run without its gate.

Detailed B reports: `/mnt/datasets-livsyn/dxl/mattergen_v1_cell_atom_fusion/experiments/cell_atom_fusion_p0/final_report.md`, `/mnt/datasets-livsyn/dxl/mattergen_v1_cell_atom_fusion/experiments/cell_atom_fusion_p0/training_report.md`, and `/mnt/datasets-livsyn/dxl/mattergen_v1_cell_atom_fusion/experiments/cell_atom_fusion_p0/smoke_report.md`.

## Evaluation scope

Baseline settings remained Official `dft_mag_density` Full FT, target 0.2, constant CFG 2, 1000 steps, one corrector step, and Adaptive CFG off. Magnetic density is the frozen CHGNet 0.3.0 surrogate (held-out MAE 0.001449, Pearson 0.9499, tau 0.01); MatterSim-5M supplies energy/force/relaxation proxies. `SURROGATE_PROPERTY_EVAL=True`; `DFT_VERIFIED=False`.
