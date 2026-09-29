# Thesis integration figures: frozen-evidence contract

These three figures were drawn for the final USTB thesis in Python/Matplotlib with the nature-figure workflow. They are vector PDFs with editable SVG sources and 600-dpi PNG previews. No simulation, resampling, or experiment is run by the figure script.

| Figure | Sole claim | Evidence source and statistical unit | Limitation |
|---|---|---|---|
| budget_verifier_framework.pdf | Fixed-K2 is a reference-preserving, finite-budget terminal-selection instance. | Frozen method definition in Chapter 3; 2,000 C0 vs 4,400 Fixed-K2 MatterGen score calls and 1 vs 3 terminal evaluations. | Mechanism only; no performance or wall-time assertion. |
| best2_vs_fixedk2_mechanism.pdf | Independent Best-of-2 uses two full trajectories; Fixed-K2 shares a prefix and expands two suffixes under the same SAFE-A rule. | Frozen Best-of-2 fairness audit and Chapter 3: 4,000/2 terminals vs 4,400/3. | Not equal budget, not evidence of Fixed-K2 superiority. The existing quantitative method comparison remains in Chapter 3. |
| timing_paired_maxf.pdf | Original Late decreases paired MatterSim MaxF versus C0, while Early/Middle intervals cross zero and costs differ. | results/rcnfgd/timing_ablation/timing_paired_bootstrap.csv; 32 fresh paired seeds, candidate minus C0 mean and 20,000-resample percentile 95% CI. | Exploratory, no multiplicity adjustment; timing changes cumulative force calls. Direct Late-minus-Early/Middle CIs cross zero and are in the accompanying prose. |

Design: neutral C0 gray, verifier selection blue, independent sampling purple, force feedback teal; white background, direct labels, editable PDF/SVG text. Body and labels are at least 7.4 pt in source. Chapter captions state the proxy, cohort, interval definition, budget and limitation. All three final PDFs passed the rendered collision audit (0 fail/0 warn); source preflight is ready (0 fail). The two remaining preflight warnings concern lack of a TIFF raster (PDF is the thesis master) and 147-mm width not matching journal 89/183-mm defaults (the target is the USTB thesis text width, not a Nature journal column).

Existing figures reused without redrawing: figures/search_framework/compute_performance_curve.pdf, figures/search_framework/method_comparison.pdf, thesis_release/innovation2/figures_polished/I2_F1_rc_nfgd_pipeline.pdf, and figures/rcnfgd/pareto_frontier.pdf. A6 Pareto points are descriptive method means within one cohort; the Timing figure uses a separate cohort and is never merged into that frontier.

