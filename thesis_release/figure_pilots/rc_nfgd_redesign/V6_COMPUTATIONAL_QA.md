# RC-NFGD computational figure v6 — internal thesis candidate

Design response to author feedback that v5 looked AI-generated: discard the
generic crystal drawings, neural-network glyph and numbered three-module
composition. This version uses a single computational graph built from only
the method's real state, score lanes, force feedback and predictor update.

Claim: at eligible late predictors (`t <= 0.02`, 20 events in the frozen
1,000-step protocol), frozen MatterGen's atomic/cell scores pass through
unchanged. Predicted-clean reconstruction is evaluated by frozen MatterSim-5M;
centered/bounded force-derived displacement is mapped into a position-score
increment and added before the official predictor update. Invalid force, cell
or geometry leaves the original score unchanged; the corrector is unchanged.

The figure abbreviates the true predictor coefficient as `c_pos`; Chapter 4
states the exact `c_t^f` equation. No material structure, measured force,
neural architecture, DFT result, or physical-stability claim is illustrated.
There are no AI-generated pixels in this version.

Source: `thesis_release/scripts/reproduce_rc_nfgd_computational_v6.py`.
Exports: `I2_F1_rc_nfgd_computational_v6.{pdf,svg,png,tiff}` at 138.4 ×
59.0 mm, matching the thesis text width. PDF/SVG text is editable.

QA (2026-09-23): source preflight 20 PASS, 1 non-applicable journal-width WARN,
0 FAIL; minimum rendered PDF glyph 6.25 pt; strict PDF collision audit 0 FAIL,
0 WARN; panel-alignment gate NOT APPLICABLE for one canvas. Visual inspection
at full and roughly 650 px page-view width found the two score lanes,
position-only addition, and fallback readable. Automated checks do not certify
aesthetic approval. Existing thesis figure remains untouched.
