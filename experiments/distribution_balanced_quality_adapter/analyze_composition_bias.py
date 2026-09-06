"""Analyze composition/CHGNet bias and construct one train-only KMeans scheme."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element
from scipy.stats import kruskal, spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/distribution_balanced_quality_adapter"
P0_ROOT = PROJECT_ROOT / "experiments/distribution_constrained_quality_adapter_p0"
SOURCE_LABELS = P0_ROOT / "quality_labels.csv"
DATA_ROOT = P0_ROOT / "data/csv"
RUNTIME_ROOT = ROOT / "runtime_tmp"
DESCRIPTORS = (
    "num_atoms",
    "num_unique_elements",
    "mean_atomic_number",
    "std_atomic_number",
    "mean_atomic_mass",
    "std_atomic_mass",
    "oxygen_fraction",
    "transition_metal_fraction",
    "alkali_fraction",
    "alkaline_earth_fraction",
    "halogen_fraction",
    "mean_electronegativity",
    "std_electronegativity",
    "mean_atomic_radius",
    "std_atomic_radius",
)
TARGETS = (
    "chgnet_force_max_ev_ang",
    "chgnet_force_mean_ev_ang",
    "chgnet_force_rms_ev_ang",
    "chgnet_energy_per_atom_ev",
)


def mean_std(values: np.ndarray, counts: np.ndarray) -> tuple[float, float]:
    mean = float(np.average(values, weights=counts))
    std = float(np.sqrt(np.average((values - mean) ** 2, weights=counts)))
    return mean, std


def radius(element: Element) -> float:
    value = element.atomic_radius
    if value is None:
        value = element.atomic_radius_calculated
    if value is None:
        raise ValueError(f"no atomic radius for {element.symbol}")
    return float(value)


def describe_formula(formula: str) -> dict[str, float]:
    composition = Composition(formula)
    items = [(Element(str(key)), float(value)) for key, value in composition.items()]
    elements = [item[0] for item in items]
    counts = np.asarray([item[1] for item in items])
    total = float(counts.sum())
    z = np.asarray([element.Z for element in elements], dtype=float)
    mass = np.asarray([float(element.atomic_mass) for element in elements])
    x = np.asarray(
        [float(element.X) if element.X is not None else np.nan for element in elements]
    )
    if np.isnan(x).any():
        raise ValueError(f"missing electronegativity in {formula}")
    radii = np.asarray([radius(element) for element in elements])
    mean_z, std_z = mean_std(z, counts)
    mean_mass, std_mass = mean_std(mass, counts)
    mean_x, std_x = mean_std(x, counts)
    mean_radius, std_radius = mean_std(radii, counts)

    def fraction(predicate) -> float:
        return float(
            sum(count for element, count in items if predicate(element)) / total
        )

    return {
        "num_atoms": total,
        "num_unique_elements": float(len(elements)),
        "mean_atomic_number": mean_z,
        "std_atomic_number": std_z,
        "mean_atomic_mass": mean_mass,
        "std_atomic_mass": std_mass,
        "oxygen_fraction": fraction(lambda element: element.symbol == "O"),
        "transition_metal_fraction": fraction(
            lambda element: element.is_transition_metal
        ),
        "alkali_fraction": fraction(lambda element: element.is_alkali),
        "alkaline_earth_fraction": fraction(lambda element: element.is_alkaline),
        "halogen_fraction": fraction(lambda element: element.is_halogen),
        "mean_electronegativity": mean_x,
        "std_electronegativity": std_x,
        "mean_atomic_radius": mean_radius,
        "std_atomic_radius": std_radius,
    }


def load_clean_x0() -> pd.DataFrame:
    labels = pd.read_csv(SOURCE_LABELS)
    pieces = []
    for split in ("train", "val"):
        source = pd.read_csv(DATA_ROOT / f"{split}.csv")
        part = labels[labels["split"] == split].sort_values("split_index").copy()
        expected = list(range(len(source)))
        if part["split_index"].astype(int).tolist() != expected:
            raise RuntimeError(f"{split} labels are misaligned")
        if (
            part["structure_id"].astype(str).tolist()
            != source["material_id"].astype(str).tolist()
        ):
            raise RuntimeError(f"{split} ids are misaligned")
        descriptors = pd.DataFrame(
            [describe_formula(formula) for formula in part["formula"]]
        )
        # Keep the actual clean-x0 unit-cell count. A reduced formula may divide
        # every stoichiometric count by a common factor.
        descriptors = descriptors.drop(columns=["num_atoms"])
        pieces.append(pd.concat([part.reset_index(drop=True), descriptors], axis=1))
    frame = pd.concat(pieces, ignore_index=True)
    if len(frame) != 128 or int((frame["split"] == "train").sum()) != 96:
        raise RuntimeError("expected exactly 96 train and 32 validation samples")
    values = frame[list(DESCRIPTORS) + list(TARGETS)].to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite descriptor or CHGNet label")
    return frame


def balanced_weights(part: pd.DataFrame) -> pd.DataFrame:
    part = part.copy()
    part["db_quality_weight"] = 1.0
    part["db_quality_bin"] = "middle"
    for cluster, group in part.groupby("cluster"):
        ordered = group.sort_values(
            ["chgnet_force_max_ev_ang", "split_index"], kind="mergesort"
        )
        tail = max(1, int(round(0.30 * len(ordered))))
        if 2 * tail >= len(ordered):
            raise RuntimeError(f"cluster {cluster} too small for 30/40/30 weights")
        part.loc[
            ordered.index[:tail], ["db_quality_weight", "db_quality_bin"]
        ] = [1.25, "low_force"]
        part.loc[
            ordered.index[-tail:], ["db_quality_weight", "db_quality_bin"]
        ] = [0.75, "high_force"]
    return part


def high_weight_histogram(frame: pd.DataFrame) -> tuple[str, str]:
    elements: Counter[str] = Counter()
    for system in frame["chemical_system"]:
        elements.update(str(system).split("-"))
    element_text = ";".join(
        f"{key}:{value}" for key, value in elements.most_common(10)
    )
    family_text = ";".join(
        f"{key}:{value}"
        for key, value in frame["chemical_system"].value_counts().head(10).items()
    )
    return element_text, family_text


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    frame = load_clean_x0()
    train = frame[frame["split"] == "train"].copy()
    validation = frame[frame["split"] == "val"].copy()

    correlation_rows = []
    for descriptor in DESCRIPTORS:
        for target in TARGETS:
            result = spearmanr(train[descriptor], train[target])
            correlation_rows.append(
                {
                    "descriptor": descriptor,
                    "quality_target": target,
                    "spearman_rho": float(result.statistic),
                    "p_value": float(result.pvalue),
                    "n_train": len(train),
                    "abs_rho_ge_0_25_and_p_lt_0_05": bool(
                        abs(result.statistic) >= 0.25 and result.pvalue < 0.05
                    ),
                }
            )
    correlations = pd.DataFrame(correlation_rows)
    correlations.to_csv(ROOT / "composition_bias_analysis.csv", index=False)

    scaler = StandardScaler().fit(train[list(DESCRIPTORS)])
    train_scaled = scaler.transform(train[list(DESCRIPTORS)])
    selected_k = 4
    kmeans = KMeans(
        n_clusters=selected_k, random_state=20260906, n_init=50
    ).fit(train_scaled)
    requested_k_sizes = np.bincount(kmeans.labels_, minlength=selected_k)
    if requested_k_sizes.min() < 10:
        selected_k = 3
        kmeans = KMeans(
            n_clusters=selected_k, random_state=20260906, n_init=50
        ).fit(train_scaled)
    sizes = np.bincount(kmeans.labels_, minlength=selected_k)
    if sizes.min() < 10:
        raise RuntimeError(f"K={selected_k} produced small train cluster: {sizes}")

    old_labels = kmeans.labels_
    cluster_mean_z = {
        old: float(train.loc[old_labels == old, "mean_atomic_number"].mean())
        for old in range(selected_k)
    }
    remap = {
        old: new
        for new, old in enumerate(
            sorted(cluster_mean_z, key=lambda old: cluster_mean_z[old])
        )
    }
    train["cluster"] = [remap[int(label)] for label in old_labels]
    val_labels = kmeans.predict(scaler.transform(validation[list(DESCRIPTORS)]))
    validation["cluster"] = [remap[int(label)] for label in val_labels]
    frame = pd.concat([train, validation], ignore_index=True)
    frame = pd.concat(
        [
            balanced_weights(part)
            for _, part in frame.groupby("split", sort=False)
        ],
        ignore_index=True,
    ).sort_values(["split", "split_index"], kind="mergesort")
    train = frame[frame["split"] == "train"]
    force_groups = [
        group["chgnet_force_max_ev_ang"].to_numpy(float)
        for _, group in train.groupby("cluster")
    ]
    force_cluster_test = kruskal(*force_groups)

    cluster_rows = []
    for cluster, group in train.groupby("cluster"):
        original_high = group[group["quality_weight"] == 1.25]
        top_elements, top_families = high_weight_histogram(original_high)
        row = {
            "cluster": int(cluster),
            "train_size": len(group),
            "validation_size": int(
                ((frame["split"] == "val") & (frame["cluster"] == cluster)).sum()
            ),
            "force_max_mean": float(group["chgnet_force_max_ev_ang"].mean()),
            "force_max_median": float(group["chgnet_force_max_ev_ang"].median()),
            "force_mean_mean": float(group["chgnet_force_mean_ev_ang"].mean()),
            "force_rms_mean": float(group["chgnet_force_rms_ev_ang"].mean()),
            "energy_per_atom_mean": float(
                group["chgnet_energy_per_atom_ev"].mean()
            ),
            "global_m2_mean_weight": float(group["quality_weight"].mean()),
            "global_m2_high_weight_proportion": float(
                (group["quality_weight"] == 1.25).mean()
            ),
            "global_m2_middle_weight_proportion": float(
                (group["quality_weight"] == 1.0).mean()
            ),
            "global_m2_low_weight_proportion": float(
                (group["quality_weight"] == 0.75).mean()
            ),
            "m2_db_mean_weight": float(group["db_quality_weight"].mean()),
            "m2_db_high_weight_proportion": float(
                (group["db_quality_weight"] == 1.25).mean()
            ),
            "m2_db_middle_weight_proportion": float(
                (group["db_quality_weight"] == 1.0).mean()
            ),
            "m2_db_low_weight_proportion": float(
                (group["db_quality_weight"] == 0.75).mean()
            ),
            "global_high_weight_top_elements": top_elements,
            "global_high_weight_top_chemical_systems": top_families,
        }
        for descriptor in DESCRIPTORS:
            row[f"{descriptor}_mean"] = float(group[descriptor].mean())
            row[f"{descriptor}_median"] = float(group[descriptor].median())
        cluster_rows.append(row)
    clusters = pd.DataFrame(cluster_rows).sort_values("cluster")
    clusters.to_csv(ROOT / "cluster_summary.csv", index=False)

    descriptor_signal = correlations[
        (correlations["quality_target"] == "chgnet_force_max_ev_ang")
        & correlations["abs_rho_ge_0_25_and_p_lt_0_05"]
    ]
    mean_weight_range = float(
        clusters["global_m2_mean_weight"].max()
        - clusters["global_m2_mean_weight"].min()
    )
    high_share_range = float(
        clusters["global_m2_high_weight_proportion"].max()
        - clusters["global_m2_high_weight_proportion"].min()
    )
    cluster_weight_signal = mean_weight_range >= 0.10 or high_share_range >= 0.20
    supported = bool(len(descriptor_signal) > 0 and cluster_weight_signal)

    source_columns = list(pd.read_csv(SOURCE_LABELS).columns)
    db_labels = frame[source_columns].copy()
    db_labels["quality_weight"] = frame["db_quality_weight"].to_numpy(float)
    db_labels["quality_bin"] = frame["db_quality_bin"].astype(str).to_numpy()
    db_labels["composition_cluster"] = frame["cluster"].to_numpy(int)
    db_labels.to_csv(RUNTIME_ROOT / "m2_db_quality_labels.csv", index=False)
    centers = np.stack(
        [
            kmeans.cluster_centers_[old]
            for old, _new in sorted(remap.items(), key=lambda pair: pair[1])
        ]
    )
    np.savez(
        RUNTIME_ROOT / "composition_clustering.npz",
        descriptor_names=np.asarray(DESCRIPTORS),
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        standardized_cluster_centers=centers,
    )
    membership_columns = [
        "split",
        "split_index",
        "structure_id",
        "formula",
        "chemical_system",
        "cluster",
        "chgnet_force_max_ev_ang",
        "quality_weight",
        "quality_bin",
        "db_quality_weight",
        "db_quality_bin",
    ] + list(DESCRIPTORS)
    frame[membership_columns].to_csv(
        RUNTIME_ROOT / "cluster_membership.csv", index=False
    )
    summary = {
        "source": "frozen original-M2 clean-x0 CHGNet labels",
        "train_n": 96,
        "validation_n": 32,
        "clustering_fit_split": "train only",
        "descriptors": list(DESCRIPTORS),
        "requested_k": 4,
        "requested_k_train_cluster_sizes": requested_k_sizes.astype(int).tolist(),
        "selected_k": selected_k,
        "train_cluster_sizes": clusters["train_size"].astype(int).tolist(),
        "descriptor_signal_rule": "force-max |rho| >= 0.25 and p < 0.05",
        "significant_force_max_descriptors": descriptor_signal[
            ["descriptor", "spearman_rho", "p_value"]
        ].to_dict("records"),
        "kruskal_force_max_statistic": float(force_cluster_test.statistic),
        "kruskal_force_max_p_value": float(force_cluster_test.pvalue),
        "kruskal_role": "descriptive only; not an additional training gate",
        "global_m2_cluster_mean_weight_range": mean_weight_range,
        "global_m2_cluster_high_weight_proportion_range": high_share_range,
        "cluster_weight_signal_rule": (
            "mean-weight range >= 0.10 OR high-weight share range >= 0.20"
        ),
        "cluster_weight_signal": bool(cluster_weight_signal),
        "composition_bias_supported": supported,
        "may_train_m2_db": supported,
        "uses_generated_or_formal_data": False,
        "uses_mattersim_e_hull_or_novelty": False,
    }
    (RUNTIME_ROOT / "mechanism_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(clusters.to_string(index=False))


if __name__ == "__main__":
    main()
