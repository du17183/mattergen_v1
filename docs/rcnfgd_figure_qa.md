# RC-NFGD Pareto figure QA

Figures follow the `nature-figure` scientific-figure contract in `docs/rcnfgd_pareto_figure_contract.md`. This influenced the use of a single explicit claim per figure, 89-mm single-column vector PDF, paired-source provenance, consistent semantic colors/shapes, editable text, and a final rendered collision audit. The plots are not a graphical claim that F0 statistically beats POST.

- Source-data cross-check: A6 original `experiments/equal_budget_online_vs_post/per_structure_metrics.csv` equals all 96 A6 records × 3 plotted metrics in `pareto_per_structure.csv` (absolute difference < 1e−14).
- Figure source preflight: 18 PASS, 3 WARN, 0 FAIL. WARNs are (i) PNG preview without TIFF, acceptable because requested deliverable is vector PDF; (ii) PNG preview 300 dpi rather than recommended 600 dpi, not used as final figure; (iii) `np.random.default_rng` matched a simulated-data heuristic, but it is used only for 20,000 resamples of recorded experimental seeds, not to fabricate plotted data.
- `property_force_pareto.pdf` and `pareto_frontier.pdf`: rendered collision audit 0 FAIL/0 WARN; PDF minimum text run 6 pt, above the 5 pt floor. Each is one panel, so panel-alignment helper returned NOT APPLICABLE; the alignment manifests are retained.
- Both final PNG previews were visually inspected at final layout after moving legends outside the data region. No axis truncation, smoothing, point removal or image manipulation was used.
- Known interpretive risk: error bars for absolute method means are wide due to between-seed heterogeneity. Paired-difference CIs in the report are the appropriate comparison; the dashed mean frontier is descriptive only. A6 and A1 are deliberately not combined in a single front.
