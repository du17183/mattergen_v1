"""Write the required 28-point Adaptive Neighborhood final report."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
BRANCH = "experiment/adaptive-neighborhood-p0"
HEAD = "1af301c5e534ff6b9b9f96ea5dab48036d9c46af"
BASE = "/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/hf_mattergen/checkpoints/dft_mag_density/checkpoints/last.ckpt"
BASE_SHA = "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e"


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> None:
    training = json.loads((ROOT / "training_summary.json").read_text())
    diagnostic = json.loads((ROOT / "diagnostic_summary.json").read_text())
    decision = json.loads((ROOT / "decision_summary.json").read_text())
    prop = json.loads((ROOT / "property_summary.json").read_text())["methods"]
    quality = pd.read_csv(ROOT / "quality_metrics.csv").set_index("method")
    paired = pd.read_csv(ROOT / "paired_results.csv").set_index("metric")
    base = quality.loc["B0"]
    new = quality.loc["A0"]
    maxf_pair = paired.loc["max_force"]
    rmsd_pair = paired.loc["rmsd"]
    verdict = decision["ADAPTIVE_NEIGHBORHOOD_P0"]
    gate = training["final_gate_grid"]
    conclusion = (
        "The 83-parameter radial gate is supported by the trajectory diagnostic and improves a preregistered held-out endpoint, so A is retained."
        if verdict == "GO" else
        "Stage-dependent distance sensitivity exists, but the 83-parameter gate does not deliver the preregistered held-out gain, so A is not retained."
    )
    report = f"""# Adaptive Neighborhood GemNet-T P0 — FINAL

1. branch: `{BRANCH}`.
2. HEAD: `{HEAD}`.
3. base checkpoint: `{BASE}` (Official `dft_mag_density` Full FT).
4. checkpoint SHA-256: `{BASE_SHA}`.
5. exact code modification: one shared normalized-time MLP `1→16→3`; `1+0.5*tanh` gates Near/Mid/Far messages only at each block's `AtomUpdateBlock` aggregation input. Graph topology, 7 Å cutoff, radial basis, atom embedding, and output heads are unchanged.
6. trainable parameters: {training['trainable_params']} (gate only).
7. trainable ratio: {training['trainable_ratio']:.8%} of {training['total_params']} total parameters.
8. training steps: {training['steps']} (early stopped: {training['early_stopped']}).
9. runtime: diagnostic generation {diagnostic['generation_seconds_total']:.2f}s cumulative + sensitivity {diagnostic['diagnostic_seconds_total']:.2f}s; training {training['elapsed_seconds']:.2f}s; generation mean B0/A0 {base['runtime_mean']:.2f}/{new['runtime_mean']:.2f}s.
10. peak VRAM: diagnostic {diagnostic['peak_allocated_bytes_max']/1024**2:.2f} MiB; training {training['peak_cuda_memory_mb']:.2f} MiB; generation mean B0/A0 {base['peak_vram_mean']:.2f}/{new['peak_vram_mean']:.2f} MiB.
11. seeds: diagnostic 94000–94031; independent paired held-out 96000–96031; training/validation 20260923/20260924.
12. Base property metrics: CHGNet surrogate Mag MAE={prop['B0']['mag_mae']:.6f}, Hit={prop['B0']['mag_hits']}/32 ({pct(prop['B0']['mag_hit_rate'])}).
13. New method property metrics: Mag MAE={prop['A0']['mag_mae']:.6f}, Hit={prop['A0']['mag_hits']}/32 ({pct(prop['A0']['mag_hit_rate'])}); paired not-worse={prop['A0']['paired_not_worse_count']}/32.
14. Base RMSD: mean={base['rmsd_mean']:.6f} Å, median={base['rmsd_median']:.6f} Å.
15. New RMSD: mean={new['rmsd_mean']:.6f} Å, median={new['rmsd_median']:.6f} Å; relative improvement={pct(decision['relative_improvement_vs_B0']['rmsd'])}.
16. Base MaxF: mean={base['max_force_mean']:.6f} eV/Å, median={base['max_force_median']:.6f} eV/Å.
17. New MaxF: mean={new['max_force_mean']:.6f} eV/Å, median={new['max_force_median']:.6f} eV/Å; relative improvement={pct(decision['relative_improvement_vs_B0']['max_force'])}.
18. E-hull: B0/A0 mean={base['e_hull_mean']:.6f}/{new['e_hull_mean']:.6f} eV/atom.
19. Stable: B0/A0={pct(base['stable_mean'])}/{pct(new['stable_mean'])}.
20. Novel: B0/A0={pct(base['novel_mean'])}/{pct(new['novel_mean'])}.
21. Unique: B0/A0={pct(base['unique_mean'])}/{pct(new['unique_mean'])}.
22. NUS: B0/A0={pct(base['nus_mean'])}/{pct(new['nus_mean'])} (delta {100*(new['nus_mean']-base['nus_mean']):+.2f} pp).
23. paired result: MaxF wins/ties/losses={int(maxf_pair['paired_wins'])}/{int(maxf_pair['paired_ties'])}/{int(maxf_pair['paired_losses'])}; RMSD={int(rmsd_pair['paired_wins'])}/{int(rmsd_pair['paired_ties'])}/{int(rmsd_pair['paired_losses'])}; property not-worse={decision['property_paired_not_worse_count']}/32.
24. mechanism diagnostic: 32 trajectories × 6 timesteps × 4 blocks × 3 bands = {diagnostic['sensitivity_rows']} rows; all four motivation conditions passed. Final gates `[near,mid,far]` at t=0.80 are {[round(gate[0][key],6) for key in ('g_near','g_mid','g_far')]}, and at t=0.02 are {[round(gate[-1][key],6) for key in ('g_near','g_mid','g_far')]}; stage-pattern check={decision['stage_dependent_gate_pattern']}.
25. GO/BORDERLINE/FAIL: **{verdict}** (geometry GO={decision['geometry_track_go']}; property GO={decision['property_track_go']}).
26. Adaptive CFG used? **NO**.
27. `DFT_VERIFIED=False`; `SURROGATE_PROPERTY_EVAL=True` (frozen evaluator calibration MAE 0.001449, Pearson 0.9499, tau 0.01).
28. Final one-sentence conclusion: {conclusion}
"""
    (ROOT / "final_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(ROOT / 'final_report.md'), "verdict": verdict}))


if __name__ == "__main__":
    main()
