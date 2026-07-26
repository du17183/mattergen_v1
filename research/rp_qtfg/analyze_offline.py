from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import ase.io
import numpy as np
import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.stats import wilcoxon

from mattergen.evaluation.metrics.evaluator import MetricsEvaluator
from mattergen.evaluation.reference.correction_schemes import TRI110Compatibility2024
from mattergen.evaluation.reference.reference_dataset import ReferenceDataset
from mattergen.evaluation.reference.reference_dataset_serializer import (
    LMDBBackedReferenceDatasetImpl,
)
from mattergen.evaluation.utils.lmdb_utils import lmdb_get
from mattergen.evaluation.utils.structure_matcher import (
    DefaultDisorderedStructureMatcher,
)

from research.rp_qtfg.common import REPORTS, RESULTS, atomic_json, now, set_stage
from research.rp_qtfg.offline_relax import METHODS, SEEDS


RELAX = RESULTS / "offline_relax"
PROBE_MANIFEST = RESULTS / "offline_probe/probe_manifest.csv"
OUT = REPORTS / "offline_direction"
REFERENCE_LMDB = Path(
    "/data/dxl/reference_assets/reference_TRI2024correction.lmdb"
)
FROZEN_A0_METRICS = Path(
    "/data/dxl/reports/formal_256/A0/official_metrics_per_structure.csv"
)


def _bootstrap_ci(
    values: np.ndarray,
    repeats: int = 20000,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan
    random = np.random.default_rng(20260726)
    indices = random.integers(0, len(values), size=(repeats, len(values)))
    low, high = np.percentile(values[indices].mean(axis=1), [2.5, 97.5])
    return float(low), float(high)


def _paired_stats(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    metric: str,
) -> dict[str, Any]:
    pair = baseline[["seed", metric]].merge(
        candidate[["seed", metric]],
        on="seed",
        suffixes=("_a0", "_candidate"),
    )
    difference = pair[f"{metric}_candidate"].to_numpy(float) - pair[
        f"{metric}_a0"
    ].to_numpy(float)
    low, high = _bootstrap_ci(difference)
    try:
        test = (
            wilcoxon(difference, zero_method="pratt")
            if np.any(difference)
            else None
        )
    except ValueError:
        test = None
    return {
        "metric": metric,
        "paired_n": len(difference),
        "difference_definition": "candidate_minus_baseline",
        "mean_difference": float(np.mean(difference)),
        "median_difference": float(np.median(difference)),
        "bootstrap_95_ci": [low, high],
        "wilcoxon_statistic": float(test.statistic) if test else 0.0,
        "wilcoxon_p_value": float(test.pvalue) if test else 1.0,
        "candidate_wins": int(np.sum(difference < -1e-12)),
        "ties": int(np.sum(np.abs(difference) <= 1e-12)),
        "candidate_losses": int(np.sum(difference > 1e-12)),
    }


def _reference() -> ReferenceDataset:
    def build_index_single_env(
        self: Any,
        _lmdb_path: Any,
    ) -> dict[str, Any]:
        result: dict[str, dict[str, Any]] = defaultdict(dict)
        with self.env.begin() as transaction:
            for chemical_system in lmdb_get(
                transaction,
                "chemical_systems",
            ):
                formulas = lmdb_get(
                    transaction,
                    f"{chemical_system}.reduced_formulas",
                )
                for formula in formulas:
                    result[chemical_system][formula] = lmdb_get(
                        transaction,
                        f"{chemical_system}.{formula}.length",
                    )
        return dict(result)

    LMDBBackedReferenceDatasetImpl._build_num_entries_by_chemsys_reduced_formulas = (
        build_index_single_env
    )
    return ReferenceDataset(
        name="TRI2024correction",
        impl=LMDBBackedReferenceDatasetImpl(
            REFERENCE_LMDB,
            cleanup_dir=False,
        ),
    )


def _result_frame(method: str) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        result_path = RELAX / method / str(seed) / "result.json"
        result = json.loads(result_path.read_text())
        if result.get("status") != "success":
            raise RuntimeError(f"unsuccessful Gate 0B result: {result_path}")
        rows.append(result)
    return pd.DataFrame(rows)


def _official_frame(
    method: str,
    raw: pd.DataFrame,
    reference: ReferenceDataset,
) -> pd.DataFrame:
    if method == "baseline":
        official = pd.read_csv(FROZEN_A0_METRICS)
        official = official[
            (official.method == "A0") & official.seed.isin(SEEDS)
        ].copy()
        if len(official) != len(SEEDS):
            raise RuntimeError("frozen A0 official metrics coverage mismatch")
    else:
        relaxed = [ase.io.read(path) for path in raw.output_path]
        originals = [ase.io.read(path) for path in raw.input_path]
        evaluator = MetricsEvaluator.from_structures_and_energies(
            structures=[
                AseAtomsAdaptor.get_structure(atoms) for atoms in relaxed
            ],
            energies=raw.relaxed_energy_ev.tolist(),
            reference=reference,
            original_structures=[
                AseAtomsAdaptor.get_structure(atoms) for atoms in originals
            ],
            stability_threshold=0.1,
            structure_matcher=DefaultDisorderedStructureMatcher(),
            energy_correction_scheme=TRI110Compatibility2024(),
            n_failed_jobs=0,
        )
        method_out = OUT / method
        method_out.mkdir(parents=True, exist_ok=True)
        evaluator.compute_metrics(
            "all",
            save_as=method_out / "official_metrics.json",
        )
        official = evaluator.as_dataframe("all").drop(
            columns=["entry"],
            errors="ignore",
        )
        official = official.reset_index(drop=True)
        official.insert(0, "seed", list(SEEDS))
    keep = [
        "seed",
        "energy_above_hull_per_atom",
        "rmsd_from_relaxation",
        "stable",
        "comp_validity",
        "structure_validity",
        "novel",
        "unique",
    ]
    return raw.merge(
        official[keep],
        on="seed",
        validate="one_to_one",
    )


def run() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    reference = _reference()
    probe = pd.read_csv(PROBE_MANIFEST)
    frames: dict[str, pd.DataFrame] = {}
    summaries = []
    for method in METHODS:
        raw = _result_frame(method)
        frame = _official_frame(method, raw, reference)
        probe_method = probe[probe.method == method][
            [
                "seed",
                "minimum_distance_angstrom",
                "chgnet_energy_per_atom_ev",
            ]
        ]
        frame = frame.merge(
            probe_method,
            on="seed",
            validate="one_to_one",
        )
        frame.to_csv(
            OUT / f"{method}_per_structure.csv",
            index=False,
        )
        frames[method] = frame
        summaries.append(
            {
                "method": method,
                "n": len(frame),
                "initial_energy_per_atom_mean": float(
                    frame.initial_energy_per_atom_ev.mean()
                ),
                "initial_max_force_mean": float(
                    frame.initial_max_force_ev_ang.mean()
                ),
                "relaxation_rmsd_mean": float(
                    frame.rmsd_from_relaxation.mean()
                ),
                "average_ehull": float(
                    frame.energy_above_hull_per_atom.mean()
                ),
                "stable_rate": float(frame.stable.mean()),
                "composition_validity": float(frame.comp_validity.mean()),
                "structure_validity": float(
                    frame.structure_validity.mean()
                ),
                "convergence_rate": float(frame.converged.mean()),
                "severe_short_bond_count": int(
                    (frame.minimum_distance_angstrom < 0.5).sum()
                ),
            }
        )
    try:
        reference.impl.cleanup(cleanup_dir=False)
    except Exception:
        pass

    baseline = frames["baseline"]
    comparisons = []
    paired = []
    for method in METHODS[1:]:
        candidate = frames[method]
        pair = baseline[
            [
                "seed",
                "initial_energy_per_atom_ev",
                "initial_max_force_ev_ang",
                "rmsd_from_relaxation",
                "energy_above_hull_per_atom",
            ]
        ].merge(
            candidate[
                [
                    "seed",
                    "initial_energy_per_atom_ev",
                    "initial_max_force_ev_ang",
                    "rmsd_from_relaxation",
                    "energy_above_hull_per_atom",
                ]
            ],
            on="seed",
            suffixes=("_a0", "_candidate"),
            validate="one_to_one",
        )
        energy_better = (
            pair.initial_energy_per_atom_ev_candidate
            < pair.initial_energy_per_atom_ev_a0 - 1e-8
        )
        force_better = (
            pair.initial_max_force_ev_ang_candidate
            < pair.initial_max_force_ev_ang_a0 - 1e-8
        )
        rmsd_better = (
            pair.rmsd_from_relaxation_candidate
            < pair.rmsd_from_relaxation_a0 - 1e-8
        )
        any_better = energy_better | force_better | rmsd_better
        summary = next(
            item for item in summaries if item["method"] == method
        )
        baseline_summary = summaries[0]
        ehull_change = (
            summary["average_ehull"]
            - baseline_summary["average_ehull"]
        )
        validity_safe = (
            summary["structure_validity"]
            >= baseline_summary["structure_validity"]
            and summary["composition_validity"]
            >= baseline_summary["composition_validity"]
            and summary["severe_short_bond_count"]
            <= baseline_summary["severe_short_bond_count"]
        )
        comparison = {
            "method": method,
            "matterSim_energy_improvement_rate": float(
                energy_better.mean()
            ),
            "matterSim_force_improvement_rate": float(force_better.mean()),
            "relaxation_rmsd_improvement_rate": float(rmsd_better.mean()),
            "any_primary_improvement_rate": float(any_better.mean()),
            "ehull_change_candidate_minus_a0": float(ehull_change),
            "structure_validity_change": float(
                summary["structure_validity"]
                - baseline_summary["structure_validity"]
            ),
            "composition_validity_change": float(
                summary["composition_validity"]
                - baseline_summary["composition_validity"]
            ),
            "validity_safe": bool(validity_safe),
            "variant_gate_go": bool(
                any_better.mean() >= 0.60
                and validity_safe
                and ehull_change <= 0.002
            ),
        }
        comparisons.append(comparison)
        for metric in (
            "initial_energy_per_atom_ev",
            "initial_max_force_ev_ang",
            "rmsd_from_relaxation",
            "energy_above_hull_per_atom",
        ):
            stats = _paired_stats(baseline, candidate, metric)
            stats["method"] = method
            paired.append(stats)

    passing = [row for row in comparisons if row["variant_gate_go"]]
    selected = (
        max(
            passing,
            key=lambda row: (
                row["any_primary_improvement_rate"],
                -row["ehull_change_candidate_minus_a0"],
            ),
        )["method"]
        if passing
        else None
    )
    physics_direction_go = selected is not None
    selected_row = (
        next(row for row in comparisons if row["method"] == selected)
        if selected
        else None
    )
    result = {
        "created_at": now(),
        "OFFLINE_PROBE_STRUCTURES": len(SEEDS),
        "PHYSICS_DIRECTION_GO": physics_direction_go,
        "PHYSICS_DIRECTION_NO_GO": not physics_direction_go,
        "SELECTED_OFFLINE_VARIANT": selected,
        "MATTERSIM_DIRECTION_AGREEMENT": (
            selected_row["matterSim_energy_improvement_rate"]
            if selected_row
            else None
        ),
        "ENERGY_IMPROVEMENT_RATE": (
            selected_row["matterSim_energy_improvement_rate"]
            if selected_row
            else None
        ),
        "FORCE_IMPROVEMENT_RATE": (
            selected_row["matterSim_force_improvement_rate"]
            if selected_row
            else None
        ),
        "RMSD_IMPROVEMENT_RATE": (
            selected_row["relaxation_rmsd_improvement_rate"]
            if selected_row
            else None
        ),
        "summaries": summaries,
        "comparisons": comparisons,
        "paired_statistics": paired,
        "interpretation": {
            "stability_source": "MatterSim-5M surrogate",
            "dft_verified": False,
            "magnetic_property_verified": False,
            "gate_rule": ">=60% structures improve at least one of MatterSim initial energy, initial max force, or relaxation RMSD; validity/composition/short-bond safe; mean E-hull change <=0.002 eV/atom.",
        },
    }
    atomic_json(
        OUT / "offline_direction_report.json",
        result,
    )
    pd.DataFrame(summaries).to_csv(
        OUT / "method_summary.csv",
        index=False,
    )
    pd.DataFrame(comparisons).to_csv(
        OUT / "comparisons.csv",
        index=False,
    )
    atomic_json(
        OUT / "paired_statistics.json",
        paired,
    )
    report = [
        "# RP-QTFG Gate 0B offline physical-direction validation",
        "",
        f"- Structures: {len(SEEDS)} frozen A0 outputs, seeds 20000–20063 (read-only).",
        "- CHGNet 0.3.0 proposes trust-region position or weak position+cell updates.",
        "- MatterSim-5M independently evaluates initial energy/force and full relaxations.",
        "- `DFT_VERIFIED=False`; `MAGNETIC_PROPERTY_VERIFIED=False`.",
        "",
        "## Decision",
        "",
        f"- `PHYSICS_DIRECTION_GO={physics_direction_go}`",
        f"- Selected offline variant: `{selected}`",
        "",
        "## Method summary",
        "",
        pd.DataFrame(summaries).to_markdown(index=False),
        "",
        "## Paired Gate comparisons",
        "",
        pd.DataFrame(comparisons).to_markdown(index=False),
    ]
    (OUT / "offline_direction_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    set_stage(
        "direction_go_no_go",
        "success" if physics_direction_go else "stop_for_review",
        (
            f"Gate 0B passed with {selected}."
            if physics_direction_go
            else "Gate 0B No-Go: no fixed trust-region variant met the frozen direction and safety gates."
        ),
        {
            "PHYSICS_DIRECTION_GO": physics_direction_go,
            "selected": selected,
            "structures": len(SEEDS),
        },
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
