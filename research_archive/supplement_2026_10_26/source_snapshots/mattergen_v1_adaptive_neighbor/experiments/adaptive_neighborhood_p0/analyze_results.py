"""Aggregate paired P0 metrics and apply the adaptive-neighborhood gate."""
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
METHODS = ("B0", "A0")
SEEDS = tuple(range(96000, 96032))
LOWER = ("mag_mae", "rmsd", "atomic_force", "max_force", "e_hull", "runtime", "peak_vram")
HIGHER = ("mag_hit", "validity", "stable", "novel", "unique", "nus")
BOOTSTRAP_SEED = 20260926


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
    structures = [AseAtomsAdaptor.get_structure(item) for item in generated_atoms]
    composition_valid = np.asarray([is_smact_valid(item) for item in structures], dtype=bool)
    structure_valid = np.asarray([structure_validity(item) for item in structures], dtype=bool)
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
    }


def summarize(method: str, values: dict[str, np.ndarray]) -> dict:
    row = {"method": method, "n": 32, "dft_verified": False}
    for metric in (*LOWER, *HIGHER, "composition_validity", "structure_validity"):
        data = values[metric].astype(float)
        row[f"{metric}_mean"] = float(data.mean())
        row[f"{metric}_median"] = float(np.median(data))
    row["score_calls_mean"] = float(values["score_calls"].mean())
    return row


def paired_rows(data: dict[str, dict[str, np.ndarray]]) -> list[dict]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, 32, size=(20_000, 32))
    rows = []
    for metric in (*LOWER, *HIGHER):
        base = data["B0"][metric].astype(float)
        candidate = data["A0"][metric].astype(float)
        delta = candidate - base
        preferred = -delta if metric in LOWER else delta
        bootstrap = delta[indices].mean(axis=1)
        base_mean = float(base.mean())
        improvement = (
            (base_mean - float(candidate.mean())) / base_mean
            if metric in LOWER and abs(base_mean) > 1.0e-12
            else float(candidate.mean()) - base_mean
        )
        rows.append({
            "comparison": "A0-vs-B0",
            "metric": metric,
            "baseline_mean": base_mean,
            "candidate_mean": float(candidate.mean()),
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


def make_decision(data: dict[str, dict[str, np.ndarray]]) -> dict:
    mean = lambda method, metric: float(data[method][metric].astype(float).mean())
    lower_gain = lambda metric: 1.0 - mean("A0", metric) / mean("B0", metric)
    rate_delta = lambda metric: mean("A0", metric) - mean("B0", metric)
    gains = {metric: lower_gain(metric) for metric in ("max_force", "rmsd", "mag_mae")}
    mag_guardrail = gains["mag_mae"] >= -0.05
    nus_guardrail = rate_delta("nus") >= -0.05
    e_hull_guardrail = mean("A0", "e_hull") <= mean("B0", "e_hull") + 0.01
    property_not_worse = int(np.sum(data["A0"]["mag_mae"] <= data["B0"]["mag_mae"]))
    gate_trace = pd.read_csv(ROOT / "gate_trace.csv")
    final = gate_trace[gate_trace.step == gate_trace.step.max()]
    deviations = np.abs(final[["g_near", "g_mid", "g_far"]].to_numpy() - 1.0)
    spans = {
        field: float(final[field].max() - final[field].min())
        for field in ("g_near", "g_mid", "g_far")
    }
    mechanism = bool(deviations.max() >= 0.01 and max(spans.values()) >= 0.005)
    geometry_metrics = [metric for metric in ("max_force", "rmsd") if gains[metric] >= 0.10]
    geometry_go = bool(geometry_metrics and mag_guardrail and nus_guardrail and mechanism)
    structure_guardrail = gains["max_force"] >= -0.05 and gains["rmsd"] >= -0.05 and e_hull_guardrail
    property_go = bool(
        gains["mag_mae"] >= 0.10
        and property_not_worse >= 22
        and structure_guardrail
        and nus_guardrail
        and mechanism
    )
    maximum_main_gain = max(gains.values())
    if geometry_go or property_go:
        verdict = "GO"
    elif (not mechanism) or maximum_main_gain < 0.05 or not (nus_guardrail and e_hull_guardrail):
        verdict = "FAIL"
    else:
        verdict = "BORDERLINE"
    return {
        "ADAPTIVE_NEIGHBORHOOD_MOTIVATED": True,
        "ADAPTIVE_NEIGHBORHOOD_P0": verdict,
        "geometry_track_go": geometry_go,
        "geometry_track_qualifying_metrics": geometry_metrics,
        "property_track_go": property_go,
        "relative_improvement_vs_B0": gains,
        "property_paired_not_worse_count": property_not_worse,
        "mag_mae_guardrail_pass": mag_guardrail,
        "nus_rate_delta": rate_delta("nus"),
        "nus_guardrail_pass": nus_guardrail,
        "e_hull_guardrail_pass": e_hull_guardrail,
        "structure_guardrail_pass": structure_guardrail,
        "gate_max_identity_deviation": float(deviations.max()),
        "gate_time_spans": spans,
        "stage_dependent_gate_pattern": mechanism,
        "sampling_score_calls_all_2000": bool(all(np.all(values["score_calls"] == 2000) for values in data.values())),
        "graph_topology_modified": False,
        "cutoff_angstrom": 7.0,
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
    for field in ("g_near", "g_mid", "g_far"):
        axis.plot(final.timestep, final[field], marker="o", label=field)
    axis.invert_xaxis(); axis.set_xlabel("diffusion t"); axis.set_ylabel("radial gate"); axis.legend(); figure.tight_layout()
    figure.savefig(plot_dir / "gate_vs_timestep.png", dpi=180); plt.close(figure)
    figure, axis = plt.subplots(figsize=(5.8, 4.3))
    axis.boxplot([data[method]["max_force"] for method in METHODS], labels=METHODS, showmeans=True)
    axis.set_ylabel("pre-relax MaxF (eV/Å)"); figure.tight_layout(); figure.savefig(plot_dir / "maxf_comparison.png", dpi=180); plt.close(figure)


def main() -> None:
    generation = pd.read_csv(ROOT / "generation_results.csv")
    properties = pd.read_csv(ROOT / "heldout_metrics.csv")
    if len(generation) != 64 or len(properties) != 64:
        raise RuntimeError("A P0 requires exactly 64 method-seed rows")
    data = {method: load_method(method, generation, properties) for method in METHODS}
    quality = pd.DataFrame([summarize(method, data[method]) for method in METHODS])
    quality.to_csv(ROOT / "quality_metrics.csv", index=False)
    efficiency_columns = ["method", "n", "runtime_mean", "runtime_median", "peak_vram_mean", "peak_vram_median", "score_calls_mean", "dft_verified"]
    quality[efficiency_columns].to_csv(ROOT / "efficiency_metrics.csv", index=False)
    pairs = pd.DataFrame(paired_rows(data))
    pairs.to_csv(ROOT / "paired_results.csv", index=False)
    verdict = make_decision(data)
    (ROOT / "decision_summary.json").write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    create_plots(data)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    print(quality.to_string(index=False))


if __name__ == "__main__":
    main()
