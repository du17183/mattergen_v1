"""Create the two compact P0 result tables from real run artifacts."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ase.io import read


ROOT = Path(__file__).resolve().parent
METHODS = ("C0", "M1", "M2")


def adapter_norms(method: str) -> tuple[float, float]:
    checkpoint = torch.load(
        ROOT / "checkpoints" / method / "adapter.pt",
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint["adapter_state_dict"]
    total = math.sqrt(sum(float(value.float().square().sum()) for value in state.values()))
    up = math.sqrt(
        sum(
            float(value.float().square().sum())
            for name, value in state.items()
            if name.endswith("up.weight") or name.endswith("up.bias")
        )
    )
    return total, up


def write_training_summary() -> None:
    rows = []
    for method in ("M1", "M2"):
        run_root = ROOT / "checkpoints" / method
        summary = json.loads((run_root / "training_summary.json").read_text())
        curve = pd.read_csv(run_root / "training_curve.csv")
        validation = curve[curve["split"] == "validation"].sort_values("step")
        total_l2, up_l2 = adapter_norms(method)
        rows.append(
            {
                "method": method,
                "quality_weighted": method == "M2",
                "steps": summary["steps"],
                "batch_size": summary["batch_size"],
                "learning_rate": summary["learning_rate"],
                "trainable_params": summary["trainable_params"],
                "trainable_parameter_count": len(summary["trainable_parameter_names"]),
                "all_losses_finite": summary["all_losses_finite"],
                "zero_init_passed": summary["zero_init_check"]["passed"],
                "frozen_parameters_unchanged": summary["frozen_parameters_unchanged"],
                "frozen_parameters_with_gradient": len(
                    summary["first_step_gradient_check"]["frozen_parameters_with_gradient"]
                ),
                "adapter_parameters_with_gradient": len(
                    summary["first_step_gradient_check"]["adapter_parameters_with_gradient"]
                ),
                "replay_fraction": summary["replay_fraction"],
                "anchor_lambda": summary["anchor_lambda"],
                "first_20_train_loss_mean": summary["first_20_train_loss_mean"],
                "last_20_train_loss_mean": summary["last_20_train_loss_mean"],
                "validation_loss_step_1": float(validation.iloc[0]["loss_total"]),
                "validation_loss_step_200": float(validation.iloc[-1]["loss_total"]),
                "best_recorded_validation_loss": float(validation["loss_total"].min()),
                "adapter_state_l2": total_l2,
                "trained_up_projection_l2": up_l2,
                "elapsed_seconds": summary["elapsed_seconds"],
                "peak_cuda_memory_mb": summary["peak_cuda_memory_mb"],
            }
        )
    pd.DataFrame(rows).to_csv(ROOT / "training_summary.csv", index=False)


def write_quality_results() -> None:
    generation = pd.read_csv(ROOT / "generation_results.csv")
    summaries = {
        method: json.loads((ROOT / "quality" / method / "quality_summary.json").read_text())
        for method in METHODS
    }
    baseline = summaries["C0"]
    rows = []
    for method in METHODS:
        summary = summaries[method]
        initial_atoms = read(
            ROOT / "relaxation" / method / "initial_with_properties.extxyz", index=":"
        )
        atomic_force_norms = np.concatenate(
            [np.linalg.norm(item.get_forces(), axis=1) for item in initial_atoms]
        )
        official = summary["official_metrics"]
        base_official = baseline["official_metrics"]
        rows.append(
            {
                "method": method,
                "n": summary["n"],
                "generation_success_rate": 1.0,
                "relaxation_success_rate": summary["relaxation_success_rate"],
                "avg_energy_above_hull_per_atom_ev": official["avg_energy_above_hull_per_atom"],
                "frac_stable": official["frac_stable_structures"],
                "frac_novel_unique_stable": official["frac_novel_unique_stable_structures"],
                "frac_novel": official["frac_novel_structures"],
                "frac_unique": official["frac_unique_structures"],
                "avg_rmsd_from_relaxation": official["avg_rmsd_from_relaxation"],
                "pre_relaxation_atomic_force_norm_mean_ev_per_a": float(atomic_force_norms.mean()),
                "pre_relaxation_atomic_force_norm_p95_ev_per_a": float(np.quantile(atomic_force_norms, 0.95)),
                "pre_relaxation_atomic_force_norm_max_ev_per_a": float(atomic_force_norms.max()),
                "pre_relaxation_max_force_mean_ev_per_a": summary["pre_relaxation_max_force_mean"],
                "pre_relaxation_max_force_median_ev_per_a": summary["pre_relaxation_max_force_median"],
                "pre_relaxation_max_force_p95_ev_per_a": summary["pre_relaxation_max_force_p95"],
                "pre_relaxation_max_force_max_ev_per_a": summary["pre_relaxation_max_force_max"],
                "relaxation_steps_mean": summary["relaxation_steps_mean"],
                "relaxation_steps_max": summary["relaxation_steps_max"],
                "generation_seconds_mean": float(
                    generation[generation["method"] == method]["elapsed_seconds"].mean()
                ),
                "delta_e_hull_vs_c0_ev": official["avg_energy_above_hull_per_atom"]
                - base_official["avg_energy_above_hull_per_atom"],
                "delta_stable_vs_c0": official["frac_stable_structures"]
                - base_official["frac_stable_structures"],
                "delta_nus_vs_c0": official["frac_novel_unique_stable_structures"]
                - base_official["frac_novel_unique_stable_structures"],
                "delta_rmsd_vs_c0": official["avg_rmsd_from_relaxation"]
                - base_official["avg_rmsd_from_relaxation"],
                "delta_structure_max_force_mean_vs_c0": summary["pre_relaxation_max_force_mean"]
                - baseline["pre_relaxation_max_force_mean"],
            }
        )
    pd.DataFrame(rows).to_csv(ROOT / "quality_results.csv", index=False)


def main() -> None:
    write_training_summary()
    write_quality_results()
    print((ROOT / "training_summary.csv").read_text())
    print((ROOT / "quality_results.csv").read_text())


if __name__ == "__main__":
    main()
