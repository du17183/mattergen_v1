"""Aggregate paired P0 metrics and apply the preregistered self-conditioning gate."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from ase.io import read
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor

from mattergen.evaluation.metrics.structure import is_smact_valid, structure_validity


ROOT = Path(__file__).resolve().parent
METHODS = ("B0", "S-off", "S-on")
SEEDS = tuple(range(95000, 95032))
LOWER = ("mag_mae", "rmsd", "atomic_force", "max_force", "e_hull", "runtime", "peak_vram")
HIGHER = ("mag_hit", "validity", "stable", "novel", "unique", "nus")
BOOTSTRAP_SEED = 20260925


def load_method(method: str, generation: pd.DataFrame, properties: pd.DataFrame) -> dict[str, np.ndarray]:
    generated_rows = generation[generation.method == method].sort_values("seed")
    property_rows = properties[properties.method == method].sort_values("seed")
    if tuple(generated_rows.seed.astype(int)) != SEEDS or tuple(property_rows.seed.astype(int)) != SEEDS:
        raise RuntimeError(f"{method}: seed contract mismatch")
    with gzip.open(ROOT / "quality" / method / "official_detailed.json.gz", "rt") as stream:
        detail = json.load(stream)
    generated_atoms = read(ROOT / "structures" / f"{method}_generated.extxyz", index=":")
    initial_atoms = read(ROOT / "relaxation" / method / "initial_with_properties.extxyz", index=":")
    relaxation = json.loads((ROOT / "relaxation" / method / "relaxation_summary.json").read_text())
    if len(generated_atoms) != 32 or len(initial_atoms) != 32 or tuple(map(int, relaxation["sample_seeds"])) != SEEDS:
        raise RuntimeError(f"{method}: relaxation/structure contract mismatch")
    pymatgen_structures = [AseAtomsAdaptor.get_structure(item) for item in generated_atoms]
    composition_valid = np.asarray([is_smact_valid(item) for item in pymatgen_structures], dtype=bool)
    structure_valid = np.asarray([structure_validity(item) for item in pymatgen_structures], dtype=bool)
    force_norms = [np.linalg.norm(item.get_forces(), axis=1) for item in initial_atoms]
    return {
        "mag_mae": property_rows.mag_absolute_error.to_numpy(float),
        "mag_hit": property_rows.mag_hit.astype(bool).to_numpy(),
        "rmsd": np.asarray(detail["rmsd_from_relaxation"], dtype=float),
        "atomic_force": np.asarray([value.mean() for value in force_norms]),
        "max_force": np.asarray([value.max() for value in force_norms]),
        "e_hull": np.asarray(detail["energy_above_hull_per_atom"], dtype=float),
        "stable": np.asarray(detail["stable"], dtype=bool),
        "novel": np.asarray(detail["novel"], dtype=bool),
        "unique": np.asarray(detail["unique"], dtype=bool),
        "nus": np.asarray(detail["novel_unique_stable"], dtype=bool),
        "validity": composition_valid & structure_valid,
        "composition_validity": composition_valid,
        "structure_validity": structure_valid,
        "runtime": generated_rows.elapsed_seconds.to_numpy(float),
        "peak_vram": generated_rows.peak_allocated_bytes.to_numpy(float) / 1024**2,
        "score_calls": generated_rows.mattergen_score_calls.to_numpy(int),
        "extra_forwards": generated_rows.extra_sampling_forwards.to_numpy(int),
    }


def summarize(method: str, values: dict[str, np.ndarray]) -> dict:
    row = {"method": method, "n": 32, "dft_verified": False}
    for metric in (*LOWER, *HIGHER, "composition_validity", "structure_validity"):
        data = values[metric].astype(float)
        row[f"{metric}_mean"] = float(data.mean())
        row[f"{metric}_median"] = float(np.median(data))
    row["score_calls_mean"] = float(values["score_calls"].mean())
    row["extra_sampling_forwards_mean"] = float(values["extra_forwards"].mean())
    return row


def paired_rows(data: dict[str, dict[str, np.ndarray]]) -> list[dict]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, 32, size=(20_000, 32))
    rows = []
    for baseline, candidate in (("B0", "S-off"), ("B0", "S-on"), ("S-off", "S-on")):
        for metric in (*LOWER, *HIGHER):
            base = data[baseline][metric].astype(float)
            cand = data[candidate][metric].astype(float)
            delta = cand - base
            preferred = -delta if metric in LOWER else delta
            bootstrap = delta[indices].mean(axis=1)
            base_mean = float(base.mean())
            improvement = (
                (base_mean - float(cand.mean())) / base_mean
                if metric in LOWER and abs(base_mean) > 1.0e-12
                else (float(cand.mean()) - base_mean)
            )
            rows.append({
                "comparison": f"{candidate}-vs-{baseline}",
                "metric": metric,
                "baseline_mean": base_mean,
                "candidate_mean": float(cand.mean()),
                "mean_delta_candidate_minus_baseline": float(delta.mean()),
                "relative_improvement_or_rate_delta": improvement,
                "paired_wins": int((preferred > 1.0e-12).sum()),
                "paired_ties": int((np.abs(preferred) <= 1.0e-12).sum()),
                "paired_losses": int((preferred < -1.0e-12).sum()),
                "bootstrap_ci95_low": float(np.quantile(bootstrap, 0.025)),
                "bootstrap_ci95_high": float(np.quantile(bootstrap, 0.975)),
                "bootstrap_resamples": 20_000,
                "bootstrap_seed": BOOTSTRAP_SEED,
            })
    return rows


def decision(data: dict[str, dict[str, np.ndarray]], pairs: pd.DataFrame) -> dict:
    mean = lambda method, metric: float(data[method][metric].astype(float).mean())
    relative_lower = lambda candidate, baseline, metric: 1.0 - mean(candidate, metric) / mean(baseline, metric)
    rate_delta = lambda candidate, baseline, metric: mean(candidate, metric) - mean(baseline, metric)
    pair = pairs.set_index(["comparison", "metric"])
    sc_vs_base = {metric: relative_lower("S-on", "B0", metric) for metric in ("max_force", "rmsd", "mag_mae")}
    sc_vs_off = {metric: relative_lower("S-on", "S-off", metric) for metric in ("max_force", "rmsd", "mag_mae")}
    clearly_better_off = {
        metric: bool(
            sc_vs_off[metric] >= 0.05
            and int(pair.loc[("S-on-vs-S-off", metric), "paired_wins"]) >= 18
        )
        for metric in sc_vs_off
    }
    mag_guardrail = sc_vs_base["mag_mae"] >= -0.05
    nus_guardrail = rate_delta("S-on", "B0", "nus") >= -0.05
    quality_track_metrics = [
        metric for metric in ("max_force", "rmsd")
        if sc_vs_base[metric] >= 0.10 and clearly_better_off[metric]
    ]
    quality_go = bool(quality_track_metrics and mag_guardrail and nus_guardrail)
    property_not_worse = int(np.sum(data["S-on"]["mag_mae"] <= data["B0"]["mag_mae"]))
    structure_guardrail = (
        sc_vs_base["max_force"] >= -0.05
        and sc_vs_base["rmsd"] >= -0.05
        and nus_guardrail
    )
    property_go = bool(
        sc_vs_base["mag_mae"] >= 0.10
        and property_not_worse >= 22
        and clearly_better_off["mag_mae"]
        and structure_guardrail
    )
    gate_trace = pd.read_csv(ROOT / "gate_trace.csv")
    final_gates = gate_trace[gate_trace.step == gate_trace.step.max()]
    gate_max_abs = float(final_gates[["g_atom", "g_pos", "g_cell"]].abs().to_numpy().max())
    time_spans = {
        field: float(final_gates[field].max() - final_gates[field].min())
        for field in ("g_atom", "g_pos", "g_cell")
    }
    gates_nonzero = gate_max_abs > 0.001
    improvements = max(sc_vs_base.values())
    severe_ehull = mean("S-on", "e_hull") > mean("B0", "e_hull") + 0.01
    forgetting = not (mag_guardrail and nus_guardrail and structure_guardrail) or severe_ehull
    if quality_go or property_go:
        verdict = "GO"
    elif (not gates_nonzero) or improvements < 0.05 or not any(clearly_better_off.values()) or forgetting:
        verdict = "FAIL"
    else:
        verdict = "BORDERLINE"
    return {
        "MULTIFIELD_SELF_CONDITIONING_P0": verdict,
        "quality_track_go": quality_go,
        "quality_track_qualifying_metrics": quality_track_metrics,
        "property_track_go": property_go,
        "relative_improvement_vs_B0": sc_vs_base,
        "relative_improvement_S_on_vs_S_off": sc_vs_off,
        "S_on_clearly_better_than_S_off": clearly_better_off,
        "property_paired_not_worse_count": property_not_worse,
        "mag_mae_guardrail_pass": mag_guardrail,
        "nus_rate_delta": rate_delta("S-on", "B0", "nus"),
        "nus_guardrail_pass": nus_guardrail,
        "structure_guardrail_pass": structure_guardrail,
        "severe_ehull_degradation": severe_ehull,
        "gate_max_abs": gate_max_abs,
        "gate_time_spans": time_spans,
        "gates_nonzero": gates_nonzero,
        "sampling_score_calls_all_2000": bool(all(np.all(values["score_calls"] == 2000) for values in data.values())),
        "extra_sampling_forwards_all_zero": bool(all(np.all(values["extra_forwards"] == 0) for values in data.values())),
        "surrogate_property_eval": True,
        "adaptive_cfg_used": False,
        "dft_verified": False,
    }


def create_plots(data: dict[str, dict[str, np.ndarray]]) -> None:
    plot_dir = ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    gates = pd.read_csv(ROOT / "gate_trace.csv")
    final = gates[gates.step == gates.step.max()].sort_values("timestep", ascending=False)
    figure, axis = plt.subplots(figsize=(6.4, 4.3))
    for field in ("g_atom", "g_pos", "g_cell"):
        axis.plot(final.timestep, final[field], marker="o", label=field)
    axis.invert_xaxis(); axis.set_xlabel("diffusion t"); axis.set_ylabel("gate"); axis.legend(); figure.tight_layout()
    figure.savefig(plot_dir / "field_gate_vs_timestep.png", dpi=180); plt.close(figure)

    trace = pd.read_csv(ROOT / "training_trace.csv")
    final_val = trace[trace.split == "validation"].iloc[-1]
    figure, axis = plt.subplots(figsize=(7.0, 4.3))
    x = np.arange(3); width = 0.35
    axis.bar(x - width/2, [final_val[f"sc_off_{field}"] for field in ("atomic_numbers", "pos", "cell")], width, label="S-off")
    axis.bar(x + width/2, [final_val[f"sc_on_{field}"] for field in ("atomic_numbers", "pos", "cell")], width, label="S-on")
    axis.set_xticks(x, ["atomic", "pos", "cell"]); axis.set_yscale("log"); axis.set_ylabel("validation loss"); axis.legend(); figure.tight_layout()
    figure.savefig(plot_dir / "field_loss_comparison.png", dpi=180); plt.close(figure)

    for metric, filename, ylabel in (("max_force", "maxf_comparison.png", "pre-relax MaxF (eV/Å)"), ("mag_mae", "property_error_comparison.png", "|surrogate magnetic density - 0.2|")):
        figure, axis = plt.subplots(figsize=(6.4, 4.3))
        axis.boxplot([data[method][metric] for method in METHODS], labels=METHODS, showmeans=True)
        axis.set_ylabel(ylabel); figure.tight_layout(); figure.savefig(plot_dir / filename, dpi=180); plt.close(figure)


def main() -> None:
    generation = pd.read_csv(ROOT / "generation_results.csv")
    properties = pd.read_csv(ROOT / "heldout_metrics.csv")
    if len(generation) != 96 or len(properties) != 96:
        raise RuntimeError("B P0 requires exactly 96 method-seed rows")
    data = {method: load_method(method, generation, properties) for method in METHODS}
    quality = pd.DataFrame([summarize(method, data[method]) for method in METHODS])
    quality.to_csv(ROOT / "quality_metrics.csv", index=False)
    efficiency_columns = ["method", "n", "runtime_mean", "runtime_median", "peak_vram_mean", "peak_vram_median", "score_calls_mean", "extra_sampling_forwards_mean", "dft_verified"]
    quality[efficiency_columns].to_csv(ROOT / "efficiency_metrics.csv", index=False)
    pairs = pd.DataFrame(paired_rows(data))
    pairs.to_csv(ROOT / "paired_results.csv", index=False)
    verdict = decision(data, pairs)
    (ROOT / "decision_summary.json").write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    create_plots(data)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    print(quality.to_string(index=False))


if __name__ == "__main__":
    main()
