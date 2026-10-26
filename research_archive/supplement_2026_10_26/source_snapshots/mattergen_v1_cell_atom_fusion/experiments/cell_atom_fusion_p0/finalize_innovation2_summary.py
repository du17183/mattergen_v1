"""Create the cross-worktree Innovation-2 decision summary after both arms finish."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
B_ROOT = PROJECT_ROOT / "experiments/cell_atom_fusion_p0"
A_ROOT = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1_async_freeze/"
    "experiments/site_level_convergence_diagnostic"
)
OUTPUT = PROJECT_ROOT / "experiments/innovation2_next_round_summary.md"


def normalized_b_verdict(value: str) -> str:
    if value in {"GO", "PASS_FORMAL32"}:
        return "GO"
    if value == "BORDERLINE":
        return "BORDERLINE"
    return "FAIL"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    a = json.loads((A_ROOT / "decision_summary.json").read_text())
    threshold = json.loads((A_ROOT / "site_convergence_threshold.json").read_text())
    structures = pd.read_csv(A_ROOT / "structure_site_summary.csv")
    val_spread = structures.loc[
        structures.subset == "validation", "site_spread_p90_minus_p10"
    ]
    b = json.loads((B_ROOT / "smoke_decision_summary.json").read_text())
    training = json.loads((B_ROOT / "checkpoints/training_summary.json").read_text())
    results = pd.read_csv(B_ROOT / "smoke_results.csv").set_index("method")
    b_verdict = normalized_b_verdict(b["cell_atom_fusion_p0"])
    if b_verdict == "GO":
        recommendation = "CELL_ATOM_FUSION"
    elif a["SITE_LEVEL_DIAGNOSTIC"] == "GO":
        recommendation = "SITE_LEVEL_ADAPTIVE_DIFFUSION"
    else:
        recommendation = "E3_PCR_FALLBACK"

    b0, c0 = results.loc["B0"], results.loc["C0"]
    a_evidence = (
        f"validation SiteSpread>=0.10: {a['fraction_site_spread_ge_0p10']:.1%}; "
        f"median spread {val_spread.median():.4f}; range gate "
        f"{a['fraction_site_range_ge_0p15']:.1%}; stable relations "
        f"{', '.join(a['stable_environment_relations']) or 'none'}; "
        f"species/noise driven={a['SPECIES_DRIVEN_HETEROGENEITY']}/"
        f"{a['MEASUREMENT_NOISE_DRIVEN']}"
    )
    b_evidence = (
        f"identity max diff 0; best val total improvement "
        f"{training['validation_signal']['relative_improvement']['total']:.4%}; "
        f"FUSION_UNUSED={training['fusion_unused']}; smoke Mag MAE "
        f"{b0['mag_absolute_error_mean']:.5f}->{c0['mag_absolute_error_mean']:.5f}, "
        f"MaxF {b0['max_force_ev_a_mean']:.4f}->{c0['max_force_ev_a_mean']:.4f}, "
        f"NUS {b0['nus_rate']:.1%}->{c0['nus_rate']:.1%}"
    )
    report = f"""# Innovation 2 — Next-Round Final Summary

| Candidate | Verdict | Key Evidence |
|---|---|---|
| Site-level convergence | {a['SITE_LEVEL_DIAGNOSTIC']} | {a_evidence} |
| Cell–Atom Fusion | {b_verdict} | {b_evidence} |

## Recommendation

`Recommended Innovation2 = {recommendation}`

The frozen selection logic was applied without lowering any historical gate: Cell–Atom Fusion has priority only after GO; otherwise a site-level diagnostic GO advances only to a future, separate Site-level Adaptive Diffusion test; if neither route is GO, the project falls back to the already successful E3-PCR Formal256 evidence.

## A. Site-level Position Convergence Diagnostic

- Existing aggregate trajectories lacked per-site clean estimates, so `REUSED_EXISTING_TRAJECTORIES=False`. The same 64 seeds were deterministically replayed; all final structures matched exactly, and existing final quality/per-atom MatterSim forces were reused.
- Each replay passively saved 1000 clean estimates with no extra forward, while the formal analysis used the original frozen 55-point schedule. Every trajectory retained 2000 score calls.
- Threshold `{threshold['threshold_site_Angstrom']:.8f} Å`, K=3, was selected only from calibration32; validation was not used.
- Final A verdict: `SITE_LEVEL_HETEROGENEITY={a['SITE_LEVEL_HETEROGENEITY']}`, `SITE_LEVEL_DIAGNOSTIC={a['SITE_LEVEL_DIAGNOSTIC']}`, `NEXT={a['NEXT']}`.

Detailed A report: `{A_ROOT / 'final_report.md'}`.

## B. Global Cell–Atom Bidirectional Fusion

- Fixed insertion: after GemNet Block3 and before Block4. Row-lattice convention uses the invariant `L L^T` Gram; no timestep gate or alternate insertion sweep was used.
- Official weights: 48,760,443; fusion/trainable parameters: 330,370 (0.6775%). Only fusion parameters were trained for 1000 steps at LR `1e-3`; the official-state digest was unchanged.
- Zero-init full-model identity passed exactly for atomic, position, and cell outputs (all max absolute differences 0, tolerance `1e-5`).
- Best fixed-noise validation step: {training['best_step']}; total improvement {training['validation_signal']['relative_improvement']['total']:.4%}; validation GO signal `{training['validation_signal']['validation_go_signal']}`. `FUSION_UNUSED={training['fusion_unused']}` at the selected checkpoint.
- The requested 16 fresh paired seeds were evaluated. Smoke verdict: `{b['cell_atom_fusion_p0']}`; `FORMAL32_ELIGIBLE={b['formal32_eligible']}`; next `{b['next']}`. No Formal32/Formal256 or ablation was run without its gate.

Detailed B reports: `{B_ROOT / 'final_report.md'}`, `{B_ROOT / 'training_report.md'}`, and `{B_ROOT / 'smoke_report.md'}`.

## Evaluation scope

Baseline settings remained Official `dft_mag_density` Full FT, target 0.2, constant CFG 2, 1000 steps, one corrector step, and Adaptive CFG off. Magnetic density is the frozen CHGNet 0.3.0 surrogate (held-out MAE 0.001449, Pearson 0.9499, tau 0.01); MatterSim-5M supplies energy/force/relaxation proxies. `SURROGATE_PROPERTY_EVAL=True`; `DFT_VERIFIED=False`.
"""
    OUTPUT.write_text(report, encoding="utf-8")
    print(json.dumps({
        "site_level": a["SITE_LEVEL_DIAGNOSTIC"],
        "cell_atom_fusion": b_verdict,
        "recommended_innovation2": recommendation,
        "output": str(OUTPUT),
    }, indent=2))


if __name__ == "__main__":
    main()
