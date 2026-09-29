# RC-NFGD signal-flow v4 — non-overwriting design trial

Core claim: the frozen MatterGen score streams remain intact except that a
reliability-gated MatterSim branch contributes an increment to the position
score of eligible late predictors; invalid feedback contributes nothing.

Design change from v1–v3: use a signal-bus architecture. The three MatterGen
fields are visible as separate rails. The force branch joins only the position
rail at an explicit addition node. Remove illustrative crystals, network icons,
decorative cards, oversized headings and an out-of-scale diffusion timeline.
This is a single method schematic, not a quantitative result panel.

Scientific cross-check: Chapter 4 frozen method (CFG=2.0, 1000 steps, 20 late
predictor events for t<=0.02, predicted-clean evaluation, centered forces,
bounded Cartesian displacement, periodic fractional mapping, actual predictor
score-coefficient map, original-score fallback, atomic/cell/corrector
unchanged). The schematic abbreviates the coefficient as `c_pos`; the thesis
text and caption provide the exact `c_t^f` equation. No DFT or stability claim.

Style reference consulted on 2026-09-23: Nature's official research figure
guide, https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/.
Only general principles (space economy, restrained labels, editable text,
removing superfluous icons) were applied. This USTB thesis graphic does not
claim to follow Nature's submission dimensions or to copy any published art.

Source: `thesis_release/scripts/reproduce_rc_nfgd_signalflow_v4.py`.
Output: `I2_F1_rc_nfgd_signalflow_v4.{pdf,svg,png,tiff}` at 138.4 × 61.0 mm.
PDF text is selectable. Source preflight: 20 PASS, 1 expected width WARN,
0 FAIL. Rendered PDF minimum text size: 6.1 pt. Strict collision audit:
0 FAIL, 0 WARN; 5 contained background-fill overlays are intentional.
Single-axes panel-alignment result: NOT APPLICABLE. Visual QA was repeated
at a 650-pixel reduced preview as well as the full render.

Status: trial only; manuscript figure reference and frozen experimental
conclusions are unchanged. This version has not received author approval.
