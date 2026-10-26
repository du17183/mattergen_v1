"""Write the required 28-point Multi-field Self-Conditioning final report."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
BRANCH = "experiment/multifield-self-conditioning-p0"
HEAD = "1af301c5e534ff6b9b9f96ea5dab48036d9c46af"
BASE = "/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/hf_mattergen/checkpoints/dft_mag_density/checkpoints/last.ckpt"
BASE_SHA = "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e"


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> None:
    training = json.loads((ROOT / "training_summary.json").read_text())
    decision = json.loads((ROOT / "decision_summary.json").read_text())
    prop = json.loads((ROOT / "property_summary.json").read_text())["methods"]
    quality = pd.read_csv(ROOT / "quality_metrics.csv").set_index("method")
    paired = pd.read_csv(ROOT / "paired_results.csv").set_index(["comparison", "metric"])
    base, off, on = quality.loc["B0"], quality.loc["S-off"], quality.loc["S-on"]
    verdict = decision["MULTIFIELD_SELF_CONDITIONING_P0"]
    conclusion = (
        "S-on beats both B0 and the matched S-off checkpoint on a preregistered track, so explicit multi-field feedback is retained."
        if verdict == "GO" else
        "S-on does not establish the preregistered advantage over both B0 and S-off, so multi-field self-conditioning is not retained."
    )
    gates = pd.read_csv(ROOT / "gate_trace.csv")
    final_gates = gates[gates.step == gates.step.max()].sort_values("timestep", ascending=False)
    maxf_pair = paired.loc[("S-on-vs-B0", "max_force")]
    off_pair = paired.loc[("S-on-vs-S-off", "max_force")]
    report = f"""# Multi-field Self-Conditioned MatterGen P0 — FINAL

1. branch: `{BRANCH}`.
2. HEAD: `{HEAD}`.
3. base checkpoint: `{BASE}` (Official `dft_mag_density` Full FT).
4. checkpoint SHA-256: `{BASE_SHA}`.
5. exact code modification: detached prior clean estimates are encoded as soft atom probabilities × frozen atom table, periodic sin/cos position deltas through `6→64→512`, and atom-count-normalized cell deltas through `9→64→512`; three zero-initialized time gates sum the fields once before the GemNet block stack. Sampling caches the prior score call and adds no forward.
6. trainable parameters: {training['trainable_params']} (SC modules + GemNet block 4 + GemNet output/lattice heads + atomic head).
7. trainable ratio: {training['trainable_ratio']:.4%} of {training['total_params']} total parameters.
8. training steps: {training['steps']} (early stopped: {training['early_stopped']}; mean SC fraction {training['mean_sc_sample_fraction']:.4f}).
9. runtime: training {training['elapsed_seconds']:.2f}s; generation mean B0/S-off/S-on={base['runtime_mean']:.2f}/{off['runtime_mean']:.2f}/{on['runtime_mean']:.2f}s.
10. peak VRAM: training {training['peak_cuda_memory_mb']:.2f} MiB; generation mean B0/S-off/S-on={base['peak_vram_mean']:.2f}/{off['peak_vram_mean']:.2f}/{on['peak_vram_mean']:.2f} MiB.
11. seeds: paired held-out 95000–95031; training/validation 20260921/20260922.
12. Base property metrics: CHGNet surrogate Mag MAE={prop['B0']['mag_mae']:.6f}, Hit={prop['B0']['mag_hits']}/32 ({pct(prop['B0']['mag_hit_rate'])}).
13. New method property metrics: S-off MAE={prop['S-off']['mag_mae']:.6f}, Hit={prop['S-off']['mag_hits']}/32; S-on MAE={prop['S-on']['mag_mae']:.6f}, Hit={prop['S-on']['mag_hits']}/32, paired not-worse vs B0={prop['S-on']['paired_not_worse_count']}/32.
14. Base RMSD: mean={base['rmsd_mean']:.6f} Å, median={base['rmsd_median']:.6f} Å.
15. New RMSD: S-off/S-on mean={off['rmsd_mean']:.6f}/{on['rmsd_mean']:.6f} Å; S-on improvement vs B0={pct(decision['relative_improvement_vs_B0']['rmsd'])}.
16. Base MaxF: mean={base['max_force_mean']:.6f} eV/Å, median={base['max_force_median']:.6f} eV/Å.
17. New MaxF: S-off/S-on mean={off['max_force_mean']:.6f}/{on['max_force_mean']:.6f} eV/Å; S-on improvement vs B0={pct(decision['relative_improvement_vs_B0']['max_force'])}.
18. E-hull: B0/S-off/S-on mean={base['e_hull_mean']:.6f}/{off['e_hull_mean']:.6f}/{on['e_hull_mean']:.6f} eV/atom.
19. Stable: B0/S-off/S-on={pct(base['stable_mean'])}/{pct(off['stable_mean'])}/{pct(on['stable_mean'])}.
20. Novel: B0/S-off/S-on={pct(base['novel_mean'])}/{pct(off['novel_mean'])}/{pct(on['novel_mean'])}.
21. Unique: B0/S-off/S-on={pct(base['unique_mean'])}/{pct(off['unique_mean'])}/{pct(on['unique_mean'])}.
22. NUS: B0/S-off/S-on={pct(base['nus_mean'])}/{pct(off['nus_mean'])}/{pct(on['nus_mean'])}.
23. paired result: S-on vs B0 MaxF wins/ties/losses={int(maxf_pair['paired_wins'])}/{int(maxf_pair['paired_ties'])}/{int(maxf_pair['paired_losses'])}; S-on vs S-off MaxF={int(off_pair['paired_wins'])}/{int(off_pair['paired_ties'])}/{int(off_pair['paired_losses'])}; property not-worse vs B0={decision['property_paired_not_worse_count']}/32.
24. mechanism diagnostic: zero-init maximum output difference={training['zero_init_check']['maximum_output_difference']:.3e}; final field gate max |g|={decision['gate_max_abs']:.6f}, time spans={decision['gate_time_spans']}; final t=0.80 gates={final_gates.iloc[0][['g_atom','g_pos','g_cell']].round(6).tolist()}, t=0.02 gates={final_gates.iloc[-1][['g_atom','g_pos','g_cell']].round(6).tolist()}; S-on superiority={decision['S_on_clearly_better_than_S_off']}.
25. GO/BORDERLINE/FAIL: **{verdict}** (quality GO={decision['quality_track_go']}; property GO={decision['property_track_go']}).
26. Adaptive CFG used? **NO**.
27. `DFT_VERIFIED=False`; `SURROGATE_PROPERTY_EVAL=True` (frozen evaluator calibration MAE 0.001449, Pearson 0.9499, tau 0.01).
28. Final one-sentence conclusion: {conclusion}
"""
    (ROOT / "final_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(ROOT / 'final_report.md'), "verdict": verdict}))


if __name__ == "__main__":
    main()
