# RC-NFGD editorial redraw v5 — internal design candidate

Purpose: show a single eligible late predictor event. The frozen MatterGen
state/scores yield a predicted-clean periodic crystal; frozen MatterSim-5M
evaluates forces; centered forces are scaled and capped; Cartesian displacement
is mapped into periodic fractional displacement and then a position-score
increment. Invalid feedback retains the original score. Atomic/cell scores and
the corrector are unchanged.

Visual source: internal AI concept B in this pilot directory. Adaptation level:
**style/composition only** (three open modules, oblique cells, restrained
teal/blue palette, white gutters). No raster pixels, AI arrows, atom positions,
network edges, text or scientific relationships were copied. All marks in v5
come from the reproducible Python source. The two cells use identical generic
atom positions; force cues are illustrative zero-sum 2D glyphs, not computed
forces or a measured material. They are not evidence of an output crystal.

The graph is only a pictorial cue for the frozen potential, not a claim about
its actual neural architecture. `c_pos` is a short diagram label for the actual
predictor coefficient `c_t^f`; Chapter 4 gives the exact equation
`Delta s_t^f = delta f / c_t^f`. The figure does not claim DFT validation,
physical stability, or superiority of the bound mechanism.

Source: `thesis_release/scripts/reproduce_rc_nfgd_editorial_v5.py`.
Exports: `I2_F1_rc_nfgd_editorial_v5.{pdf,svg,png,tiff}` at 138.4 × 82.0 mm.
The dimensions follow the thesis text width, not Nature journal production
specifications. Editable text and vector marks remain in PDF/SVG.

QA on 2026-09-23: source preflight 20 PASS, 1 width-related WARN, 0 FAIL;
rendered PDF minimum glyph 6.45 pt; strict collision audit 0 FAIL and 0 WARN;
single-axes alignment result NOT APPLICABLE. The preview was inspected both at
high resolution and reduced to about 650 px wide, corresponding to page-scale
reading. The thesis still references the existing figure; no manuscript
replacement or compile was performed.

Status: **internal design draft — author approval and degree/publication policy
eligibility unverified**.
