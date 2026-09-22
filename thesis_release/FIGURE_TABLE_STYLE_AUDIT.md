# Figure and table presentation audit

This audit concerns presentation only.  It does not change any frozen result,
seed, metric definition, statistical decision, or scientific claim.

## Design rules adopted for the polished preview

- Generate quantitative plots from the frozen CSV/JSON files with reproducible
  Python source code; keep PDF/SVG as the primary artwork and PNG only as a
  convenience preview.
- Design at the intended final width (single-column or full text width), rather
  than drawing a very wide image and shrinking its labels in LaTeX.
- Use one typeface, a restrained colour-blind-safe palette, and redundant
  shape/line/texture cues so the figures remain interpretable in grayscale.
- Keep captions outside the artwork.  Remove decorative plot titles when the
  LaTeX caption already identifies the result.
- Show uncertainty where it is available in the frozen evidence; do not add
  significance claims or invent error bars.
- Preserve all observations in skewed distributions.  Use an inset or a
  clearly marked transformed axis for the central mass, while retaining the
  full-range panel and reporting outliers.
- Use `booktabs` rules, grouped headers, explicit units, consistent decimal
  precision, and restrained emphasis in tables.  Bold is reserved for the
  registered primary result or a clearly defined comparison, not every column
  maximum.

These choices follow the official figure guidance from Nature and IEEE: use
consistent readable typography, vector artwork when possible, avoid relying on
colour alone, size figures for the final column width, and define symbols and
error bars in captions.

## Current high-priority changes

| Item | Current issue | Polished-preview treatment |
| --- | --- | --- |
| I1-F2 | 128 paired lines form a dense spaghetti plot | Sort non-zero paired improvements, retain the exact zero-gain seeds as an explicit tie block, and report W/T/L over all 128 pairs |
| I1-F6 | 11-node horizontal diagram shrinks to unreadable text | Two-row evidence map with compact labels and an explicit status legend |
| I2-F1 | Six-step pipeline is too wide for the thesis text width | Two-row numbered pipeline with phase callout |
| I2-F3 | Extreme tails compress the central distribution | Full-range rug plus central-99% histogram; no samples removed |
| I2-F5 | A few large points compress the scatter cloud | Full-range scatter plus an explicitly labelled axis-wise central-core view; all paired points retained |
| I2-F2/F6 | Three-panel bars become too small and contain redundant titles | Vertical panels, panel labels, direct values, and a common legend |
| Chapters 1--2 | Concept diagrams were included as PNG despite having source SVGs | Re-render six diagrams as code-native PDF/SVG/PNG masters with flat panels, consistent typography, and no drop shadows; use PDF masters in the polished preview |
| Main figures | LaTeX currently prefers PNG even though PDF/SVG masters exist | Use PDF masters for release and conceptual figures; retain PNG for preview/fallback |
| Tables 3-5/4-4/5-2/5-3/5-4 | Wide, flat headers and long status strings | Grouped headers in the polished preview, `makecell` wrapping where needed, units in headers, and restrained emphasis |

## Verification status of the polished preview

- Twelve quantitative release figures have reproducible PDF/SVG/PNG masters under
  `innovation1/figures_polished/` and `innovation2/figures_polished/`; six
  conceptual diagrams have matching masters under `concept_figures_polished/`.
- The polished LaTeX preview compiles to 125 pages with 24 figure labels,
  24 table entries, 22 bibliography items, no missing glyphs, and no undefined
  citations. The original 127-page PNG preview remains untouched.
- Grouped headers are applied to the wide result tables 4-4, 5-2, 5-3, and
  5-4 in the preview only. The bold cleanup in Table 5-2 removes misleading
  emphasis from C0 Novel and Random/Linear NUS; no numeric cell was changed.
- A small number of residual TeX overfull/underfull diagnostics remain in
  dense legacy prose/table cells (maximum about 8.2 pt in the current preview).
  They are presentation issues to review visually, not evidence changes; the
  preview is not yet declared the official submission PDF.

The current image pass is a second structural refinement of the quantitative
plots and a vector redraw of the Chapter 1--2 diagrams. It is still a
preview-only artifact: original PNG/SVG artwork, source data, and the first
polished quantitative set remain available for rollback.

## Integrity constraints

The original artwork remains available until the polished preview has been
visually accepted.  The regeneration script reads only the frozen release
artifacts under `innovation1/results`, `innovation2/results`, and the frozen
summary tables.  It must never launch MatterGen, MatterSim, CHGNet, or a GPU
job.  Any changed figure must be checked against its source CSV/JSON and then
recompiled in the thesis before it is promoted to the final artwork directory.

## References

- Nature Research, *Building and exporting figure panels*:
  <https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/>
- Nature, *Formatting guide*:
  <https://www.nature.com/nature/for-authors/formatting-guide>
- IEEE Author Center, *Create graphics for your article*:
  <https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/>
- IEEE Author Center, *Resolution and size*:
  <https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/resolution-and-size/>
