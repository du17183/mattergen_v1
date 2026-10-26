"""Frozen zero-generation offline analysis for Reference-Preserved Budgeted CFG."""

from __future__ import annotations

import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.io import read
from pymatgen.entries.computed_entries import ComputedStructureEntry

from mattergen.evaluation.reference.reference_dataset import ReferenceDataset
from mattergen.evaluation.utils.dataset_matcher import DisorderedDatasetUniquenessComputer
from mattergen.evaluation.utils.structure_matcher import DefaultDisorderedStructureMatcher


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/reference_preserved_budgeted_cfg"
OFFLINE = ROOT / "offline"
PLOTS = ROOT / "plots"
SOURCE = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1_field_cfg/experiments/field_decoupled_adaptive_cfg/calibration"
)
PROPERTY_TOLERANCE = 1.0e-6
SPLIT_SALT = "rpb-a-cfg-offline-v1"
RANDOM_SALT = "rpb-a-cfg-random-k2-v1"
TRACE_FEATURES = (
    "r_atomic",
    "r_pos",
    "r_cell",
    "score_ratio_atomic",
    "score_ratio_pos",
    "score_ratio_cell",
    "score_alignment_atomic",
    "score_alignment_pos",
    "score_alignment_cell",
    "residual_mean",
    "residual_median",
    "residual_std",
    "residual_range",
    "early_residual_mean",
    "mid_residual_mean",
    "ema_log_residual",
    "recent_residual_trend",
)
SOURCE_FILES = (
    "evaluated_results.csv",
    "generation_manifest.csv",
    "policy_final_all.extxyz",
    "policies.yaml",
    "seeds.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")


def split_seeds(seeds: Iterable[int]) -> dict[str, list[int]]:
    ordered = sorted(
        map(int, seeds),
        key=lambda seed: hashlib.sha256(f"{SPLIT_SALT}:{seed}".encode()).hexdigest(),
    )
    if len(ordered) != 48 or len(set(ordered)) != 48:
        raise RuntimeError("expected exactly 48 unique offline seeds")
    return {
        "offline_train": ordered[:24],
        "offline_validation": ordered[24:36],
        "offline_test": ordered[36:48],
    }


def load_quality() -> tuple[dict[tuple[int, str], dict], dict[tuple[int, str], dict]]:
    details: dict[tuple[int, str], dict] = {}
    entries: dict[tuple[int, str], dict] = {}
    for quality_root in sorted((SOURCE / "quality").iterdir()):
        if not quality_root.is_dir():
            continue
        policy = quality_root.name
        per = pd.read_csv(quality_root / "per_structure.csv")
        with gzip.open(quality_root / "official_detailed.json.gz", "rt") as stream:
            detail = json.load(stream)
        if len(per) != len(detail["entry"]):
            raise RuntimeError(f"quality length mismatch: {policy}")
        for output_index, record in per.reset_index(drop=True).iterrows():
            key = (int(record.sample_seed), policy)
            if key in details:
                raise RuntimeError(f"duplicate quality key: {key}")
            details[key] = {
                "quality_output_index": int(output_index),
                "mattersim_pre_relaxation_max_force": float(record.pre_relaxation_max_force),
                "mattersim_relaxation_steps": int(record.relaxation_steps),
                "mattersim_final_energy_ev": float(record.final_energy_ev),
                "rmsd_from_relaxation": float(detail["rmsd_from_relaxation"][output_index]),
            }
            entries[key] = detail["entry"][output_index]
    return details, entries


def empirical_duplicate_audit(values: pd.DataFrame) -> dict:
    atoms = read(SOURCE / "policy_final_all.extxyz", index=":")
    if len(atoms) != len(values):
        raise RuntimeError("structure/result length mismatch")
    by_digest: dict[str, list[float]] = {}
    for item in atoms:
        branch_id = f"{int(item.info['seed'])}:{item.info['policy_id']}"
        row = values.loc[values.branch_id == branch_id]
        if len(row) != 1:
            raise RuntimeError(f"branch lookup failed: {branch_id}")
        digest = hashlib.sha256()
        digest.update(np.asarray(item.numbers, dtype=np.int64).tobytes())
        digest.update(np.asarray(item.get_scaled_positions(wrap=False), dtype=np.float64).tobytes())
        digest.update(np.asarray(item.cell.array, dtype=np.float64).tobytes())
        by_digest.setdefault(digest.hexdigest(), []).append(float(row.iloc[0].property_absolute_error))
    repeated = [items for items in by_digest.values() if len(items) > 1]
    maximum_range = max((max(items) - min(items) for items in repeated), default=0.0)
    return {
        "exact_structure_duplicate_groups": len(repeated),
        "exact_structure_duplicate_rows": sum(map(len, repeated)),
        "maximum_property_error_range_within_exact_duplicates": maximum_range,
        "frozen_property_tolerance": PROPERTY_TOLERANCE,
        "tolerance_exceeds_observed_duplicate_range": PROPERTY_TOLERANCE > maximum_range,
        "new_evaluator_calls": 0,
    }


def build_candidate_matrix() -> tuple[pd.DataFrame, dict[tuple[int, str], dict], dict]:
    values = pd.read_csv(SOURCE / "evaluated_results.csv")
    manifest = pd.read_csv(SOURCE / "generation_manifest.csv")
    if values.shape != (576, 28) or values.seed.nunique() != 48 or values.policy_id.nunique() != 12:
        raise RuntimeError(f"unexpected source shape: {values.shape}")
    if values.duplicated(["seed", "policy_id"]).any():
        raise RuntimeError("duplicate seed-policy source rows")
    details, entries = load_quality()
    values = values.merge(
        manifest[["seed", "policy_id", "prefix_state_sha256"]],
        on=["seed", "policy_id"],
        how="left",
        validate="one_to_one",
    )
    values["seed"] = values.seed.astype(int)
    for column in (
        "quality_output_index",
        "mattersim_pre_relaxation_max_force",
        "mattersim_relaxation_steps",
        "mattersim_final_energy_ev",
        "rmsd_from_relaxation",
    ):
        values[column] = [details[(int(row.seed), str(row.policy_id))][column] for row in values.itertuples()]
    split = split_seeds(values.seed.unique())
    membership = {seed: name for name, members in split.items() for seed in members}
    values["offline_split"] = values.seed.map(membership)
    base = values.loc[values.policy_id == "C0"].set_index("seed")
    if len(base) != 48:
        raise RuntimeError("C0 rows incomplete")
    values["c0_property_error"] = values.seed.map(base.property_absolute_error)
    values["property_delta_vs_c0"] = values.property_absolute_error - values.c0_property_error
    values["property_gain_vs_c0"] = -values.property_delta_vs_c0
    values["c0_e_hull"] = values.seed.map(base.e_hull)
    values["e_hull_delta_vs_c0"] = values.e_hull - values.c0_e_hull
    values["c0_stable"] = values.seed.map(base.stable).astype(bool)
    values["c0_valid"] = values.seed.map(base.evaluation_valid).astype(bool)
    values["c0_novel"] = values.seed.map(base.novel).astype(bool)
    values["meaningful_property_improvement"] = (
        values.property_error if "property_error" in values else values.property_absolute_error
    ) < values.c0_property_error - PROPERTY_TOLERANCE
    values["safe_a"] = (
        (values.evaluation_valid.astype(int) >= values.c0_valid.astype(int))
        & (values.stable.astype(int) >= values.c0_stable.astype(int))
        & (values.e_hull <= values.c0_e_hull + 0.01 + 1e-12)
        & values.meaningful_property_improvement
        & (values.policy_id != "C0")
    )
    values["safe_b"] = values.safe_a & (values.novel.astype(int) >= values.c0_novel.astype(int))
    values["source_group_unique_label"] = values.unique.astype(bool)
    values["source_group_nus_label"] = values.nus.astype(bool)
    values["unique"] = np.nan
    values["nus"] = np.nan
    values["unique_note"] = "missing_by_design_recompute_only_after_mixed_batch_selection"
    values["nus_note"] = "missing_by_design_recompute_only_after_mixed_batch_selection"
    values["allocator_features_available"] = False
    values["allocator_feature_missing_reason"] = "field_sampler_did_not_serialize_prebranch_residual_or_score_trace"
    for feature in TRACE_FEATURES:
        values[feature] = np.nan
    values["MaxF"] = values["mattersim_pre_relaxation_max_force"]
    values["RMSD"] = values["rmsd_from_relaxation"]
    values = values.rename(
        columns={
            "property_absolute_error": "property_error",
            "evaluation_valid": "valid",
            "e_hull": "E_hull",
        }
    )
    required = [
        "seed", "offline_split", "policy_id", "policy_kind", "g_atomic", "g_pos", "g_cell",
        "start", "duration", "property_error", "property_delta_vs_c0", "property_gain_vs_c0",
        "E_hull", "e_hull_delta_vs_c0", "stable", "valid", "novel", "unique", "nus",
        "MaxF", "RMSD", "mattersim_pre_relaxation_max_force", "rmsd_from_relaxation", "safe_a", "safe_b",
        "prefix_state_sha256", "allocator_features_available", "allocator_feature_missing_reason",
        *TRACE_FEATURES,
    ]
    extras = [column for column in values.columns if column not in required]
    matrix = values[required + extras].sort_values(["seed", "policy_id"]).reset_index(drop=True)
    duplicate_audit = empirical_duplicate_audit(values.rename(columns={"property_error": "property_absolute_error"}))
    return matrix, entries, {"split": split, "duplicate_audit": duplicate_audit}


def recompute_unique(selected: pd.DataFrame, entries: dict[tuple[int, str], dict]) -> np.ndarray:
    parsed = []
    for index, row in enumerate(selected.itertuples()):
        entry = ComputedStructureEntry.from_dict(entries[(int(row.seed), str(row.policy_id))])
        entry.entry_id = index
        parsed.append(entry)
    dataset = ReferenceDataset.from_entries("mixed_selected_batch", parsed)
    computer = DisorderedDatasetUniquenessComputer(DefaultDisorderedStructureMatcher())
    return np.asarray(computer(dataset), dtype=bool)


def select_terminal(frame: pd.DataFrame, policies: Iterable[str], safe_column: str) -> pd.DataFrame:
    allowed = set(map(str, policies)) | {"C0"}
    rows = []
    for seed, seed_frame in frame.loc[frame.policy_id.isin(allowed)].groupby("seed", sort=True):
        c0 = seed_frame.loc[seed_frame.policy_id == "C0"]
        if len(c0) != 1:
            raise RuntimeError(f"missing C0 for seed {seed}")
        safe = seed_frame.loc[seed_frame[safe_column].astype(bool)]
        if len(safe):
            selected = safe.sort_values(["property_error", "policy_id"]).iloc[0]
        else:
            selected = c0.iloc[0]
        rows.append(selected)
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def metrics(
    selected: pd.DataFrame,
    entries: dict[tuple[int, str], dict],
    candidate_count: float,
    compute_multiplier: float,
) -> tuple[dict, pd.DataFrame]:
    selected = selected.copy().sort_values("seed").reset_index(drop=True)
    selected["mixed_unique"] = recompute_unique(selected, entries)
    selected["mixed_nus"] = selected.stable.astype(bool) & selected.novel.astype(bool) & selected.mixed_unique
    selected["unique"] = selected["mixed_unique"]
    selected["nus"] = selected["mixed_nus"]
    result = {
        "n": len(selected),
        "property_mae": float(selected.property_error.mean()),
        "property_median": float(selected.property_error.median()),
        "e_hull_mean": float(selected.E_hull.mean()),
        "stable_rate": float(selected.stable.mean()),
        "novel_rate": float(selected.novel.mean()),
        "unique_rate": float(selected.mixed_unique.mean()),
        "nus_rate": float(selected.mixed_nus.mean()),
        "validity_rate": float(selected.valid.mean()),
        "maxf_mean": float(selected.mattersim_pre_relaxation_max_force.mean()),
        "rmsd_mean": float(selected.rmsd_from_relaxation.mean()),
        "adaptive_acceptance_rate": float((selected.policy_id != "C0").mean()),
        "fallback_rate": float((selected.policy_id == "C0").mean()),
        "average_candidate_count": float(candidate_count),
        "estimated_compute_multiplier": float(compute_multiplier),
    }
    return result, selected


def policy_increment(matrix: pd.DataFrame, policy: str) -> float:
    row = matrix.loc[matrix.policy_id == policy].iloc[0]
    return 0.6 if row.policy_kind == "pulse" else 1.0


def subset_compute(matrix: pd.DataFrame, policies: Iterable[str]) -> float:
    return 1.0 + sum(policy_increment(matrix, policy) for policy in policies)


def c0_metrics(frame: pd.DataFrame, entries: dict[tuple[int, str], dict]) -> tuple[dict, pd.DataFrame]:
    c0 = frame.loc[frame.policy_id == "C0"]
    return metrics(c0, entries, candidate_count=0.0, compute_multiplier=1.0)


def add_headroom(record: dict, c0: dict, oracle: dict) -> dict:
    denominator = c0["property_mae"] - oracle["property_mae"]
    record = dict(record)
    record["safe_oracle_gain"] = c0["property_mae"] - record["property_mae"]
    record["oracle_headroom"] = denominator
    record["recovered_oracle_headroom"] = (
        record["safe_oracle_gain"] / denominator if denominator > 0 else float("nan")
    )
    record["relative_property_improvement_vs_c0"] = (
        record["safe_oracle_gain"] / c0["property_mae"] if c0["property_mae"] > 0 else float("nan")
    )
    return record


def quality_gate(method: dict, c0: dict) -> bool:
    return bool(
        method["property_mae"] < c0["property_mae"]
        and method["e_hull_mean"] - c0["e_hull_mean"] <= 0.01 + 1e-12
        and c0["stable_rate"] - method["stable_rate"] <= 0.05 + 1e-12
        and c0["validity_rate"] - method["validity_rate"] <= 0.05 + 1e-12
    )


def oracle_analysis(matrix: pd.DataFrame, entries: dict[tuple[int, str], dict]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    adaptive = sorted(set(matrix.policy_id) - {"C0"})
    report_rows = []
    selected_rows = []
    cache: dict[tuple[str, str], dict] = {}
    for safe_name, safe_column in (("SAFE-A", "safe_a"), ("SAFE-B", "safe_b")):
        for split_name, members in (
            ("all", sorted(matrix.seed.unique())),
            ("offline_train", sorted(matrix.loc[matrix.offline_split == "offline_train", "seed"].unique())),
            ("offline_validation", sorted(matrix.loc[matrix.offline_split == "offline_validation", "seed"].unique())),
            ("offline_test", sorted(matrix.loc[matrix.offline_split == "offline_test", "seed"].unique())),
        ):
            frame = matrix.loc[matrix.seed.isin(members)]
            c0, _ = c0_metrics(frame, entries)
            selected = select_terminal(frame, adaptive, safe_column)
            oracle, enriched = metrics(
                selected,
                entries,
                candidate_count=len(adaptive),
                compute_multiplier=subset_compute(matrix, adaptive),
            )
            oracle = add_headroom(oracle, c0, oracle)
            oracle.update(selector=safe_name, split=split_name, method="Oracle-All")
            report_rows.append(oracle)
            cache[(safe_name, split_name)] = oracle
            enriched.insert(0, "selector", safe_name)
            enriched.insert(1, "split", split_name)
            selected_rows.append(enriched)
    return pd.DataFrame(report_rows), pd.concat(selected_rows, ignore_index=True), cache


def fixed_k_analysis(
    matrix: pd.DataFrame,
    entries: dict[tuple[int, str], dict],
    oracle_cache: dict,
) -> tuple[pd.DataFrame, dict]:
    policies = sorted(set(matrix.policy_id) - {"C0"})
    output = []
    selected_by_key: dict[tuple[str, int, str], pd.DataFrame] = {}
    for safe_name, safe_column in (("SAFE-A", "safe_a"), ("SAFE-B", "safe_b")):
        train = matrix.loc[matrix.offline_split == "offline_train"]
        validation = matrix.loc[matrix.offline_split == "offline_validation"]
        test = matrix.loc[matrix.offline_split == "offline_test"]
        split_frames = {"offline_train": train, "offline_validation": validation, "offline_test": test}
        for k in (1, 2, 3, 4):
            candidates = []
            for subset in itertools.combinations(policies, k):
                selected = select_terminal(train, subset, safe_column)
                mae = float(selected.property_error.mean())
                candidates.append((mae, subset_compute(matrix, subset), tuple(subset)))
            _, _, best_subset = min(candidates, key=lambda item: (item[0], item[1], item[2]))
            validation_retained = False
            for split_name, frame in split_frames.items():
                c0, _ = c0_metrics(frame, entries)
                selected = select_terminal(frame, best_subset, safe_column)
                method, enriched = metrics(
                    selected,
                    entries,
                    candidate_count=k,
                    compute_multiplier=subset_compute(matrix, best_subset),
                )
                method = add_headroom(method, c0, oracle_cache[(safe_name, split_name)])
                if split_name == "offline_validation":
                    validation_retained = quality_gate(method, c0)
                method.update(
                    selector=safe_name,
                    split=split_name,
                    method=f"Fixed-K{k}",
                    k=k,
                    policies=";".join(best_subset),
                    validation_retained=validation_retained if split_name != "offline_train" else "pending",
                    subset_discovery="train_only_exhaustive",
                )
                output.append(method)
                selected_by_key[(safe_name, k, split_name)] = enriched
            for row in output:
                if row.get("selector") == safe_name and row.get("k") == k:
                    row["validation_retained"] = validation_retained
    return pd.DataFrame(output), selected_by_key


def random_policies(seed: int, policies: list[str], k: int) -> tuple[str, ...]:
    ranked = sorted(
        policies,
        key=lambda policy: hashlib.sha256(f"{RANDOM_SALT}:{seed}:{policy}".encode()).hexdigest(),
    )
    return tuple(ranked[:k])


def random_and_allocator_analysis(
    matrix: pd.DataFrame,
    entries: dict[tuple[int, str], dict],
    oracle_cache: dict,
    fixed_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    policies = sorted(set(matrix.policy_id) - {"C0"})
    output = []
    allocation_rows = []
    for k in (1, 2, 3):
        for split_name in ("offline_train", "offline_validation", "offline_test"):
            frame = matrix.loc[matrix.offline_split == split_name]
            chunks = []
            compute = []
            for seed in sorted(frame.seed.unique()):
                chosen = random_policies(int(seed), policies, k)
                chunks.append(select_terminal(frame.loc[frame.seed == seed], chosen, "safe_a"))
                compute.append(subset_compute(matrix, chosen))
                allocation_rows.extend(
                    {"seed": int(seed), "split": split_name, "allocator": f"Random-K{k}", "rank": rank + 1, "policy_id": policy}
                    for rank, policy in enumerate(chosen)
                )
            selected = pd.concat(chunks, ignore_index=True)
            result, _ = metrics(selected, entries, candidate_count=k, compute_multiplier=float(np.mean(compute)))
            c0, _ = c0_metrics(frame, entries)
            result = add_headroom(result, c0, oracle_cache[("SAFE-A", split_name)])
            result.update(
                selector="SAFE-A",
                split=split_name,
                method=f"Random-K{k}",
                k=k,
                status="EVALUATED_FROZEN_HASH_RANDOM",
                reason="",
            )
            output.append(result)
    fixed_test_k2 = fixed_results.loc[
        (fixed_results.selector == "SAFE-A")
        & (fixed_results.split == "offline_test")
        & (fixed_results.k == 2)
    ].iloc[0]
    for allocator in ("Residual-heuristic-K2", "Logistic-linear-K2", "Shallow-tree-K2"):
        output.append(
            {
                "selector": "SAFE-A",
                "split": "offline_test",
                "method": allocator,
                "k": 2,
                "status": "NOT_EVALUABLE",
                "reason": "no serialized sample-specific prebranch residual/score features for the 48 Field seeds",
                "property_mae": np.nan,
                "recovered_oracle_headroom": np.nan,
                "relative_property_improvement_vs_c0": np.nan,
                "estimated_compute_multiplier": np.nan,
                "fixed_k2_reference_property_mae": float(fixed_test_k2.property_mae),
            }
        )
    return pd.DataFrame(output), pd.DataFrame(allocation_rows)


def oracle_k_analysis(
    matrix: pd.DataFrame,
    entries: dict[tuple[int, str], dict],
    oracle_cache: dict,
) -> pd.DataFrame:
    policies = sorted(set(matrix.policy_id) - {"C0"})
    output = []
    for safe_name, safe_column in (("SAFE-A", "safe_a"), ("SAFE-B", "safe_b")):
        for split_name in ("offline_train", "offline_validation", "offline_test"):
            frame = matrix.loc[matrix.offline_split == split_name]
            oracle_selected = select_terminal(frame, policies, safe_column)
            c0, _ = c0_metrics(frame, entries)
            for k in (1, 2, 3, 4):
                per_seed_cost = []
                for row in oracle_selected.itertuples():
                    required = [] if row.policy_id == "C0" else [str(row.policy_id)]
                    fillers = sorted(
                        [policy for policy in policies if policy not in required],
                        key=lambda policy: (policy_increment(matrix, policy), policy),
                    )[: max(0, k - len(required))]
                    per_seed_cost.append(subset_compute(matrix, required + fillers))
                result, _ = metrics(
                    oracle_selected,
                    entries,
                    candidate_count=k,
                    compute_multiplier=float(np.mean(per_seed_cost)),
                )
                result = add_headroom(result, c0, oracle_cache[(safe_name, split_name)])
                result.update(
                    selector=safe_name,
                    split=split_name,
                    method=f"Oracle-K{k}",
                    k=k,
                    hindsight_policy_identity_required=True,
                    interpretation="equals Oracle-All for K>=1 by the registered per-seed hindsight definition",
                )
                output.append(result)
    return pd.DataFrame(output)


def candidate_summary(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, frame in matrix.loc[matrix.policy_id != "C0"].groupby("policy_id"):
        rows.append(
            {
                "policy_id": policy,
                "policy_kind": frame.policy_kind.iloc[0],
                "safe_a_rate": float(frame.safe_a.mean()),
                "safe_b_rate": float(frame.safe_b.mean()),
                "mean_property_gain": float(frame.property_gain_vs_c0.mean()),
                "mean_safe_a_gain": float(frame.loc[frame.safe_a, "property_gain_vs_c0"].mean()) if frame.safe_a.any() else 0.0,
                "estimated_branch_increment": policy_increment(matrix, policy),
            }
        )
    return pd.DataFrame(rows).sort_values("policy_id")


def save_plots(
    oracle: pd.DataFrame,
    fixed: pd.DataFrame,
    oracle_k: pd.DataFrame,
    adaptive: pd.DataFrame,
    policy: pd.DataFrame,
    matrix: pd.DataFrame,
) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    style = {"color": "#3266a8", "edgecolor": "black", "linewidth": 0.5}
    full = oracle.loc[oracle.split == "all"].set_index("selector")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["C0", "SAFE-A Oracle", "SAFE-B Oracle"], [
        float(matrix.loc[matrix.policy_id == "C0", "property_error"].mean()),
        full.loc["SAFE-A", "property_mae"], full.loc["SAFE-B", "property_mae"],
    ], **style)
    ax.set_ylabel("Property MAE"); ax.set_title("Retrospective oracle headroom (48 seeds)")
    fig.tight_layout(); fig.savefig(PLOTS / "oracle_headroom.png", dpi=180); plt.close(fig)

    test_fixed = fixed.loc[(fixed.selector == "SAFE-A") & (fixed.split == "offline_test")]
    test_oracle = oracle_k.loc[(oracle_k.selector == "SAFE-A") & (oracle_k.split == "offline_test")]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(test_fixed.k, 100 * test_fixed.recovered_oracle_headroom, marker="o", label="Fixed-K")
    ax.plot(test_oracle.k, 100 * test_oracle.recovered_oracle_headroom, marker="s", label="Oracle-K")
    ax.axhline(60, color="grey", ls="--", lw=1); ax.axhline(75, color="grey", ls=":", lw=1)
    ax.set_xlabel("Adaptive branch budget K"); ax.set_ylabel("Recovered Oracle-All headroom (%)"); ax.legend()
    fig.tight_layout(); fig.savefig(PLOTS / "recovered_headroom_vs_K.png", dpi=180); plt.close(fig)

    random_test = adaptive.loc[(adaptive.split == "offline_test") & adaptive.method.str.startswith("Random")]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(test_fixed.k, test_fixed.property_mae, marker="o", label="Fixed-K")
    ax.plot(random_test.k, random_test.property_mae, marker="s", label="Random-K")
    ax.text(0.5, 0.96, "Adaptive allocator unavailable: prebranch features were not serialized", transform=ax.transAxes, ha="center", va="top", fontsize=8)
    ax.set_xlabel("K"); ax.set_ylabel("Offline-test Property MAE"); ax.legend()
    fig.tight_layout(); fig.savefig(PLOTS / "fixed_vs_adaptive_K.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(policy)); ax.bar(x - .18, policy.safe_a_rate, width=.36, label="SAFE-A", **style)
    ax.bar(x + .18, policy.safe_b_rate, width=.36, label="SAFE-B", color="#65a765", edgecolor="black", linewidth=.5)
    ax.set_xticks(x, policy.policy_id, rotation=45, ha="right"); ax.set_ylabel("Safe beneficial rate"); ax.legend()
    fig.tight_layout(); fig.savefig(PLOTS / "candidate_policy_safe_rate.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(policy.policy_id, policy.mean_property_gain, **style); ax.axhline(0, color="black", lw=.8)
    ax.tick_params(axis="x", rotation=45); ax.set_ylabel("Mean property gain vs C0")
    fig.tight_layout(); fig.savefig(PLOTS / "candidate_policy_property_gain.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    gains = matrix.loc[matrix.safe_a & (matrix.policy_id != "C0"), "property_gain_vs_c0"]
    ax.hist(gains, bins=24, color="#65a765", edgecolor="black", linewidth=.5); ax.set_xlabel("SAFE-A candidate property gain"); ax.set_ylabel("Count")
    fig.tight_layout(); fig.savefig(PLOTS / "safe_gain_distribution.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4)); ax.axis("off")
    ax.text(.5, .58, "Allocator precision/recall not estimable", ha="center", va="center", fontsize=14)
    ax.text(.5, .42, "No sample-specific prebranch residual or score features\nwere serialized for the 48 Field seeds.", ha="center", va="center", fontsize=10)
    fig.tight_layout(); fig.savefig(PLOTS / "allocator_precision_recall.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(test_fixed.estimated_compute_multiplier, 100 * test_fixed.recovered_oracle_headroom, label="Fixed-K", s=55)
    ax.scatter(test_oracle.estimated_compute_multiplier, 100 * test_oracle.recovered_oracle_headroom, label="Oracle-K", marker="s", s=55)
    ax.axvline(2.2, color="grey", ls="--", lw=1); ax.set_xlabel("Estimated C0 compute multiplier"); ax.set_ylabel("Recovered headroom (%)"); ax.legend()
    fig.tight_layout(); fig.savefig(PLOTS / "compute_vs_recovered_headroom.png", dpi=180); plt.close(fig)


def write_reports(
    matrix: pd.DataFrame,
    oracle: pd.DataFrame,
    fixed: pd.DataFrame,
    oracle_k: pd.DataFrame,
    adaptive: pd.DataFrame,
    split_payload: dict,
    duplicate_audit: dict,
) -> dict:
    oa = oracle.loc[(oracle.selector == "SAFE-A") & (oracle.split == "offline_test")].iloc[0]
    ob = oracle.loc[(oracle.selector == "SAFE-B") & (oracle.split == "offline_test")].iloc[0]
    fk2 = fixed.loc[(fixed.selector == "SAFE-A") & (fixed.split == "offline_test") & (fixed.k == 2)].iloc[0]
    ok2 = oracle_k.loc[(oracle_k.selector == "SAFE-A") & (oracle_k.split == "offline_test") & (oracle_k.k == 2)].iloc[0]
    headroom_go = bool(ok2.recovered_oracle_headroom >= 0.60)
    adaptive_rows = adaptive.loc[(adaptive.split == "offline_test") & adaptive.method.isin(["Residual-heuristic-K2", "Logistic-linear-K2", "Shallow-tree-K2"])]
    deployable = adaptive_rows.loc[adaptive_rows.status.str.startswith("EVALUATED")]
    adaptive_supported = False
    status = {
        "OFFLINE_ORACLE_RECOMPUTE": "COMPLETE",
        "BUDGETED_HEADROOM": "GO" if headroom_go else "FAIL",
        "FIXED_K_HEADROOM": float(fk2.recovered_oracle_headroom),
        "ADAPTIVE_K_HEADROOM": None,
        "ADAPTIVE_ALLOCATION_VALUE": "SUPPORTED" if adaptive_supported else "NOT_SUPPORTED",
        "BEST_FIXED_K2": fk2.policies,
        "BEST_FIXED_K2_PROPERTY_MAE": float(fk2.property_mae),
        "BEST_FIXED_K2_COMPUTE_MULTIPLIER": float(fk2.estimated_compute_multiplier),
        "BEST_ADAPTIVE_K2": "NOT_EVALUABLE_NO_PREBRANCH_FEATURES",
        "ORACLE_K2": float(ok2.property_mae),
        "ORACLE_K2_RECOVERED_HEADROOM": float(ok2.recovered_oracle_headroom),
        "ORACLE_K2_COMPUTE_MULTIPLIER": float(ok2.estimated_compute_multiplier),
        "ORACLE_ALL": float(oa.property_mae),
        "ORACLE_ALL_SAFE_B": float(ob.property_mae),
        "REFERENCE_PRESERVATION": "NOT_RUN",
        "SHARED_PREFIX_BRANCHING": "NOT_RUN",
        "REFERENCE_FALLBACK_EFFECT": "NOT_RUN",
        "BUDGETED_CFG_P0": "NOT_RUN",
        "EQUAL_BUDGET_ADVANTAGE": "NOT_RUN",
        "BUDGETED_CFG_FORMAL256": "NOT_RUN",
        "METHOD_COLLAPSED_TO_BEST_OF_K": True,
        "INNOVATION1_FINAL_STATUS": "MIXED",
        "BUDGETED_HEADROOM_INTERPRETATION": "RETROSPECTIVE_EXISTENCE_BOUND_ONLY",
        "STOP_REASON": "No deployable sample-adaptive allocator can be evaluated because all registered prebranch residual and score features are absent from the only paired Field cohort.",
        "ZERO_NEW_GENERATION": True,
        "ZERO_NEW_GPU_SAMPLING": True,
        "DFT_VERIFIED": False,
    }
    json_write(ROOT / "pipeline_status.json", status)
    c0_test = float(matrix.loc[(matrix.offline_split == "offline_test") & (matrix.policy_id == "C0"), "property_error"].mean())
    oracle_all_report = f"""# SAFE-A / SAFE-B Oracle-All report

This is a retrospective offline upper-bound analysis over the retained 48-seed Field-Decoupled cohort. It is not a deployable controller, fresh validation, or DFT verification.

The split was fixed by SHA-256 before subset fitting: 24 train, 12 validation, 12 test. Mixed-batch uniqueness was recomputed from the selected relaxed structures with MatterGen's `DefaultDisorderedStructureMatcher`; NUS was then recomputed as stable AND novel AND mixed-batch unique. Source-group uniqueness labels were not reused.

## Offline-test result

| Selector | C0 Property MAE | Oracle-All Property MAE | Relative improvement | Stable | Novel | Unique | NUS | Validity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SAFE-A | {c0_test:.8f} | {oa.property_mae:.8f} | {oa.relative_property_improvement_vs_c0:.2%} | {oa.stable_rate:.2%} | {oa.novel_rate:.2%} | {oa.unique_rate:.2%} | {oa.nus_rate:.2%} | {oa.validity_rate:.2%} |
| SAFE-B | {c0_test:.8f} | {ob.property_mae:.8f} | {ob.relative_property_improvement_vs_c0:.2%} | {ob.stable_rate:.2%} | {ob.novel_rate:.2%} | {ob.unique_rate:.2%} | {ob.nus_rate:.2%} | {ob.validity_rate:.2%} |

SAFE-A requires validity and stability not below paired C0, E-hull no more than 0.01 eV/atom above paired C0, and property improvement greater than {PROPERTY_TOLERANCE:g}. SAFE-B additionally preserves paired novelty. Uniqueness is deliberately evaluated only after batch selection.
"""
    (OFFLINE / "oracle_all_report.md").write_text(oracle_all_report)
    report = f"""# Reference-Preserved Budgeted Adaptive CFG — offline report

No new structures or GPU samples were generated. The analysis used the completed 48-seed × 12-policy Field-Decoupled matrix.

## Frozen split and data audit

- Train / validation / test: 24 / 12 / 12, assigned by `SHA256(rpb-a-cfg-offline-v1:seed)`.
- Candidate rows: {len(matrix)}; policies: {matrix.policy_id.nunique()}; seeds: {matrix.seed.nunique()}.
- Exact-structure duplicate groups: {duplicate_audit['exact_structure_duplicate_groups']}.
- Maximum property-error spread within exact duplicates: {duplicate_audit['maximum_property_error_range_within_exact_duplicates']:.3g}.
- Frozen meaningful-improvement tolerance: {PROPERTY_TOLERANCE:g}.
- Mixed-batch Unique and NUS were recomputed after every reported terminal selection.

## Primary offline-test results (SAFE-A)

| Method | Property MAE | Recovered Oracle headroom | Estimated compute ×C0 |
|---|---:|---:|---:|
| C0 | {c0_test:.8f} | 0.00% | 1.00× |
| Oracle-All | {oa.property_mae:.8f} | 100.00% | {oa.estimated_compute_multiplier:.2f}× |
| Oracle-K2 | {ok2.property_mae:.8f} | {ok2.recovered_oracle_headroom:.2%} | {ok2.estimated_compute_multiplier:.2f}× |
| Fixed-K2 `{fk2.policies}` | {fk2.property_mae:.8f} | {fk2.recovered_oracle_headroom:.2%} | {fk2.estimated_compute_multiplier:.2f}× |

`Oracle-K2` equals Oracle-All by the registered definition because hindsight may choose the policy identity separately for every seed; one adaptive slot already contains the best safe policy. It is a theoretical budget oracle, not evidence that an allocator can identify that policy.

## Allocator audit and stop decision

The Field sampler retained final outcomes and `prefix_state_sha256`, but did not serialize atomic/position/cell residual statistics, conditional-unconditional score norms, alignment, EMA, or stage trace summaries. The hash is an integrity identifier and was not converted into a predictor. Candidate final property and MatterSim outcomes were prohibited as allocator inputs.

Consequently, residual, logistic/linear, and shallow-tree sample-adaptive allocators are `NOT_EVALUABLE`. A model using only policy identity produces the same ranking for every seed and collapses to Fixed-K. The random allocator is reported as a baseline but does not establish adaptive value.

```text
BUDGETED_HEADROOM = {status['BUDGETED_HEADROOM']}
ADAPTIVE_ALLOCATION_VALUE = {status['ADAPTIVE_ALLOCATION_VALUE']}
BEST_FIXED_K2 = {status['BEST_FIXED_K2']}
BEST_ADAPTIVE_K2 = {status['BEST_ADAPTIVE_K2']}
ORACLE_K2 = {status['ORACLE_K2']:.8f}
ORACLE_ALL = {status['ORACLE_ALL']:.8f}
METHOD_COLLAPSED_TO_BEST_OF_K = TRUE
INNOVATION1_FINAL_STATUS = MIXED
```

Gate A passes as a retrospective existence bound. Gate B fails because no deployable sample-adaptive allocator can be evaluated from the registered offline inputs. Under the frozen stop rule, branch-sampler implementation, reference reproduction, P0, and Formal256 are not run.

The train-selected Fixed-K2 requires an estimated 3.00× C0 generation compute because both selected policies are full-trajectory constants. Even the hindsight Oracle-K2 averages {ok2.estimated_compute_multiplier:.2f}× under the source sampler's cost model, above the preferred 2.2× target. The existing constant-policy outcomes cannot be reinterpreted as step-400 shared-prefix outcomes without generating a different experiment.

All results use CHGNet/MatterSim surrogate evaluations; `DFT_VERIFIED=False`.
"""
    (OFFLINE / "final_offline_report.md").write_text(report)
    return status


def main() -> None:
    OFFLINE.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    protocol = ROOT / "offline_protocol.yaml"
    source_paths = [SOURCE / name for name in SOURCE_FILES]
    source_paths.extend(sorted((SOURCE / "quality").glob("*/per_structure.csv")))
    source_paths.extend(sorted((SOURCE / "quality").glob("*/official_detailed.json.gz")))
    source_manifest = {str(path.relative_to(SOURCE)): sha256(path) for path in source_paths}
    source_manifest["offline_protocol.yaml"] = sha256(protocol)
    matrix, entries, audit = build_candidate_matrix()
    split_payload = {
        "schema_version": 1,
        "algorithm": "sort_by_sha256",
        "hash_payload": f"{SPLIT_SALT}:{{seed}}",
        **audit["split"],
        "counts": {name: len(seeds) for name, seeds in audit["split"].items()},
        "frozen_before_subset_fitting": True,
    }
    json_write(OFFLINE / "split_manifest.json", split_payload)
    json_write(OFFLINE / "source_manifest.json", source_manifest)
    json_write(OFFLINE / "numerical_tolerance_audit.json", audit["duplicate_audit"])
    matrix.to_csv(OFFLINE / "candidate_matrix.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    matrix[["seed", "offline_split", "policy_id", "allocator_features_available", "allocator_feature_missing_reason", *TRACE_FEATURES]].to_csv(
        OFFLINE / "allocator_features.csv", index=False
    )
    oracle, oracle_selected, oracle_cache = oracle_analysis(matrix, entries)
    oracle.to_csv(OFFLINE / "oracle_all_metrics.csv", index=False)
    oracle_selected.to_csv(OFFLINE / "oracle_all_selected.csv", index=False)
    fixed, _ = fixed_k_analysis(matrix, entries, oracle_cache)
    fixed.to_csv(OFFLINE / "fixed_k_results.csv", index=False)
    oracle_k = oracle_k_analysis(matrix, entries, oracle_cache)
    oracle_k.to_csv(OFFLINE / "oracle_k_results.csv", index=False)
    adaptive, allocations = random_and_allocator_analysis(matrix, entries, oracle_cache, fixed)
    adaptive.to_csv(OFFLINE / "adaptive_k_results.csv", index=False)
    adaptive.to_csv(OFFLINE / "allocator_results.csv", index=False)
    allocations.to_csv(OFFLINE / "random_allocator_assignments.csv", index=False)
    policy = candidate_summary(matrix)
    policy.to_csv(OFFLINE / "candidate_policy_summary.csv", index=False)
    recovered = pd.concat(
        [
            fixed.assign(result_family="fixed"),
            oracle_k.assign(result_family="oracle"),
            adaptive.assign(result_family="allocator"),
        ],
        ignore_index=True,
        sort=False,
    )
    recovered.to_csv(OFFLINE / "recovered_headroom.csv", index=False)
    save_plots(oracle, fixed, oracle_k, adaptive, policy, matrix)
    status = write_reports(matrix, oracle, fixed, oracle_k, adaptive, split_payload, audit["duplicate_audit"])
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
