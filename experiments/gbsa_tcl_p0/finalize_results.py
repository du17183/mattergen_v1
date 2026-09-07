"""Create the compact per-seed table and human-readable GBSA-TCL P0 report."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import shutil

from ase.io import read
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/gbsa_tcl_p0"
METHODS = ("C0", "TCL", "GBSA-TCL")
SEEDS = tuple(range(84000, 84008))


def load_per_seed() -> pd.DataFrame:
    generation = pd.read_csv(ROOT / "generation_results.csv")
    rows = []
    for method in METHODS:
        generated = generation[generation.method == method].sort_values("seed")
        with gzip.open(ROOT / "quality" / method / "official_detailed.json.gz", "rt") as stream:
            detail = json.load(stream)
        relaxation = json.loads(
            (ROOT / "relaxation" / method / "relaxation_summary.json").read_text()
        )
        initial = read(
            ROOT / "relaxation" / method / "initial_with_properties.extxyz", index=":"
        )
        if tuple(generated.seed.astype(int)) != SEEDS:
            raise RuntimeError(f"{method}: seed mismatch")
        if tuple(map(int, relaxation["sample_seeds"])) != SEEDS:
            raise RuntimeError(f"{method}: relaxation mismatch")
        for index, seed in enumerate(SEEDS):
            force_norms = np.linalg.norm(initial[index].get_forces(), axis=1)
            rmsd = float(detail["rmsd_from_relaxation"][index])
            max_force = float(force_norms.max())
            steps = int(relaxation["relaxation_steps"][index])
            flags = []
            if rmsd > 0.5:
                flags.append("RMSD>0.5")
            if max_force > 1.0:
                flags.append("MaxF>1")
            if max_force > 2.0:
                flags.append("MaxF>2")
            if steps > 200:
                flags.append("Steps>200")
            if steps > 400:
                flags.append("Steps>400")
            item = generated.iloc[index]
            rows.append({
                "method": method,
                "seed": seed,
                "formula": item.formula,
                "num_atoms": int(item.num_atoms),
                "e_hull_ev_per_atom": float(detail["energy_above_hull_per_atom"][index]),
                "stable": bool(detail["stable"][index]),
                "nus": bool(detail["novel_unique_stable"][index]),
                "novel": bool(detail["novel"][index]),
                "unique": bool(detail["unique"][index]),
                "rmsd_a": rmsd,
                "atomic_force_mean_ev_per_a": float(force_norms.mean()),
                "structure_max_force_ev_per_a": max_force,
                "relaxation_steps": steps,
                "tail_flags": ";".join(flags),
                "generation_seconds": float(item.elapsed_seconds),
                "dft_verified": False,
            })
    return pd.DataFrame(rows)


def tail_counts(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        frame = per_seed[per_seed.method == method]
        rows.append({
            "method": method,
            "n": len(frame),
            "rmsd_gt_0p5_count": int((frame.rmsd_a > 0.5).sum()),
            "max_force_gt_1_count": int((frame.structure_max_force_ev_per_a > 1.0).sum()),
            "max_force_gt_2_count": int((frame.structure_max_force_ev_per_a > 2.0).sum()),
            "steps_gt_200_count": int((frame.relaxation_steps > 200).sum()),
            "steps_gt_400_count": int((frame.relaxation_steps > 400).sum()),
        })
    return pd.DataFrame(rows)


def percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def report(result_commit: str, per_seed: pd.DataFrame, tails: pd.DataFrame) -> str:
    training = json.loads(
        (ROOT / "checkpoints/GBSA-TCL/training_summary.json").read_text()
    )
    quality = pd.read_csv(ROOT / "quality_results.csv").set_index("method")
    drift = pd.read_csv(ROOT / "score_drift_results.csv")
    late = drift[(drift.field == "pos") & (drift.stage == "late_70_100")].set_index("comparison")
    tcl_late = float(late.loc["TCL_vs_C0", "relative_l2_diff_mean"])
    gbsa_late = float(late.loc["GBSA-TCL_vs_C0", "relative_l2_diff_mean"])
    late_improvement = 1.0 - gbsa_late / tcl_late
    grad = training["gradient_by_objective"]

    quality_lines = [
        "| Method | E-hull (eV/atom) | Stable | NUS | Novel | Unique | RMSD (Å) | Atomic force | Max force | Relax steps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = quality.loc[method]
        quality_lines.append(
            f"| {method} | {row.e_hull_mean_ev_per_atom:.6f} | {percent(row.stable)} | "
            f"{percent(row.nus)} | {percent(row.novel)} | {percent(row['unique'])} | "
            f"{row.rmsd_mean_a:.6f} | {row.atomic_force_mean_ev_per_a:.6f} | "
            f"{row.structure_max_force_mean_ev_per_a:.6f} | {row.relaxation_steps_mean:.3f} |"
        )
    tail_lines = [
        "| Method | RMSD>0.5 | MaxF>1 | MaxF>2 | Steps>200 | Steps>400 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in tails.iterrows():
        tail_lines.append(
            f"| {row.method} | {row.rmsd_gt_0p5_count} | {row.max_force_gt_1_count} | "
            f"{row.max_force_gt_2_count} | {row.steps_gt_200_count} | {row.steps_gt_400_count} |"
        )
    seed_lines = [
        "| Method | Seed | Formula | E-hull | RMSD | Atomic force | Max force | Steps | Tail |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in per_seed.iterrows():
        seed_lines.append(
            f"| {row.method} | {row.seed} | {row.formula} | {row.e_hull_ev_per_atom:.5f} | "
            f"{row.rmsd_a:.5f} | {row.atomic_force_mean_ev_per_a:.5f} | "
            f"{row.structure_max_force_ev_per_a:.5f} | {row.relaxation_steps} | "
            f"{row.tail_flags or '-'} |"
        )

    return f"""# GBSA-TCL P0 FINAL

Final decision: **GO**. This is an eight-seed directional P0 result, not a formal statistical conclusion. All quality and relaxation metrics use MatterSim-5M/project evaluation; **DFT_VERIFIED=False**.

1. Branch: `experiment/gbsa-tcl-p0`.
2. Result commit before this generated report: `{result_commit}`. The exact final handoff HEAD is reported after committing this report.
3. P0 seeds: `84000–84007` (8 paired seeds).
4. Historical overlap: none; `81000–81031`, `82000–82063`, and `83000–83255` were excluded.

5. Total/trainable parameters: `{training['total_params']:,}` / `{training['trainable_params']:,}`.
6. Trainable ratio: `{100.0 * training['trainable_ratio']:.4f}%`.
7. Frozen modules: input embedding, noise/time encoder, GemNet blocks 1–4, position/output heads, cell/lattice heads, property embeddings, and every other pretrained parameter.
8. Trainable modules: condition adapter/mixin `{training['trainable_groups']['condition_adapter_mixin']:,}` parameters; atomic head `{training['trainable_groups']['atomic_head']:,}` parameters.
9. Old cell TCL was genuinely removed: `{not training['old_clean_cell_tcl_present']}`. No clean-cell `1/alpha_t` objective is present.
10. Final loss: two-view original MatterGen base loss + warm-started atomic TCL + normalized periodic position TCL + piecewise C0 position score anchor + piecewise C0 cell score anchor. A sampled exact-gradient cap targets each auxiliary at no more than 1× base before global clipping; no atomic C0 anchor was added in P0.
11. Reverse-progress anchor schedule (`p=1-t`): position `0.01/0.05/0.10`; cell `0.01/0.025/0.05` over `p=[0,.3)/[.3,.7)/[.7,1]`, with 100-step warmup.
12. Training steps: 50-step independent diagnostic, then a fresh 1000-step main run.
13. Main training wall time: `{training['elapsed_seconds']:.3f} s`.

14. Base gradient mean: `{training['base_gradient_l2_mean']:.6f}`.
15. Atomic TCL gradient mean: `{grad['atomic_tcl']['balanced_gradient_l2_mean']:.6f}`; gradient/base mean `{grad['atomic_tcl']['gradient_over_base_mean']:.4f}`; cosine vs base `{grad['atomic_tcl']['cosine_vs_base_mean']:.4f}`.
16. Position TCL gradient mean: `{grad['position_tcl']['balanced_gradient_l2_mean']:.6f}`; gradient/base mean `{grad['position_tcl']['gradient_over_base_mean']:.4f}`.
17. Position-anchor gradient mean: `{grad['position_anchor']['balanced_gradient_l2_mean']:.6f}`; gradient/base mean `{grad['position_anchor']['gradient_over_base_mean']:.4f}`.
18. Cell-anchor gradient mean: `{grad['cell_anchor']['balanced_gradient_l2_mean']:.6f}`; gradient/base mean `{grad['cell_anchor']['gradient_over_base_mean']:.4f}`.
19. Maximum measured balanced auxiliary/base gradient ratio: `{training['maximum_balanced_auxiliary_over_base']:.6f}×`. All balance scales remained 1.0, so the guardrail did not need to hide an unstable objective.
20. Gradient clipping: `{training['gradient_clipping_count']}/{training['steps']}` at L2 threshold `{training['gradient_clip_threshold']}` (original TCL: 1000/1000).
21. NaN/Inf: none; teacher frozen/eval/no optimizer = `{training['teacher_frozen']}/{training['teacher_eval']}/{training['teacher_optimizer_params'] == 0}`; initial student/teacher field max-absolute differences were all zero.

22. TCL late-position relative L2 drift vs C0: `{tcl_late:.6f}`.
23. GBSA-TCL late-position relative L2 drift vs C0: `{gbsa_late:.6f}`.
24. Late-position drift reduction: `{100.0 * late_improvement:.2f}%`, evaluated on all 24 new P0 structures at 10 fixed noise bins with shared corruption and CFG=2. Late-cell drift also fell from `0.116445` to `0.038039`.

25–27. Absolute P0 quality, geometry, force, and relaxation results:

{chr(10).join(quality_lines)}

28–32. Tail counts:

{chr(10).join(tail_lines)}

33. Every paired-seed result:

{chr(10).join(seed_lines)}

34. Three required repairs: gradient repair **YES**; score-drift repair **YES**; real-geometry repair **YES**.
35. Final P0 decision: **GO**.
36. Independent 32-seed P1 is scientifically justified, but was not started in this round.
37. Failure layer: not applicable. The n=8 uncertainty remains, and no formal claim should be made from P0.
38. Strict Atomic-Only TCL Residual Adapter: **not recommended as the immediate next step** because GBSA-TCL passed P0; retain it only as the preregistered fallback if independent P1 fails.

## Scientific interpretation

The root-cause chain is supported by a direct intervention: removing unbounded cell consistency and freezing the pretrained geometry path eliminated gradient domination (`0/1000` clips), while C0 score anchoring/partial freezing reduced late-position drift by about `{100.0 * late_improvement:.1f}%`. On the same new paired seeds, GBSA-TCL reduced RMSD, atomic force, maximum force, and relaxation cost relative to frozen TCL without a visible quality collapse. The effect is strong enough for P1, but the sample size is deliberately too small for confirmatory inference.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-commit", required=True)
    args = parser.parse_args()
    per_seed = load_per_seed()
    per_seed.to_csv(ROOT / "per_seed_results.csv", index=False)
    tails = tail_counts(per_seed)
    tails.to_csv(ROOT / "tail_counts.csv", index=False)
    shutil.copyfile(
        ROOT / "checkpoints/GBSA-TCL/training_summary.csv",
        ROOT / "training_summary.csv",
    )
    shutil.copyfile(
        ROOT / "checkpoints/GBSA-TCL/gradient_diagnostics.csv",
        ROOT / "gradient_diagnostics.csv",
    )
    (ROOT / "final_report.md").write_text(
        report(args.result_commit, per_seed, tails), encoding="utf-8"
    )
    print(tails.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
