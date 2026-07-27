from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from mattergen.common.data.collate import collate
from research.spg_static_mvp.common import (
    REPORTS,
    RESULTS,
    atomic_json,
    atomic_text,
    now,
    set_stage,
)
from research.spg_static_mvp.generation import (
    build_c0_generator,
    build_recording_sampler,
    configure_determinism,
    find_gemnet,
    singleton_condition,
)
from research.spg_static_mvp.static_builder import (
    StaticBucketConfig,
    StaticPeriodicGraphBuilder,
    install_static_builder,
)


STATE_COUNT = 64
FIELDS = ("atomic_numbers", "pos", "cell")
ATOL = 1e-6
RTOL = 1e-5


def flatten_tensors(value, prefix: str = "") -> dict[str, torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return {prefix or "tensor": value.detach().clone()}
    if isinstance(value, Mapping):
        output = {}
        for key, child in value.items():
            output.update(flatten_tensors(child, f"{prefix}.{key}".strip(".")))
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        output = {}
        for index, child in enumerate(value):
            output.update(flatten_tensors(child, f"{prefix}.{index}".strip(".")))
        return output
    return {}


def tensor_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict:
    if reference.shape != actual.shape:
        return {
            "shape_match": False,
            "finite": False,
            "bitwise_equal": False,
            "allclose": False,
            "max_absolute_error": float("inf"),
            "relative_l2_error": float("inf"),
            "cosine_similarity": -1.0,
        }
    difference = (reference - actual).float()
    reference_float = reference.float()
    actual_float = actual.float()
    max_absolute_error = (
        float(difference.abs().max()) if difference.numel() else 0.0
    )
    reference_norm = float(torch.linalg.vector_norm(reference_float))
    difference_norm = float(torch.linalg.vector_norm(difference))
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(actual).all())
    if reference.numel() == 0 or (reference_norm == 0.0 and difference_norm == 0.0):
        cosine_similarity = 1.0
    else:
        cosine_similarity = float(
            torch.nn.functional.cosine_similarity(
                reference_float.reshape(1, -1),
                actual_float.reshape(1, -1),
            )
        )
    return {
        "shape_match": True,
        "finite": finite,
        "bitwise_equal": bool(torch.equal(reference, actual)),
        "allclose": finite and bool(
            torch.allclose(reference, actual, atol=ATOL, rtol=RTOL)
        ),
        "max_absolute_error": max_absolute_error,
        "relative_l2_error": difference_norm / max(reference_norm, 1e-12),
        "cosine_similarity": cosine_similarity,
    }


def select_states() -> tuple[list[dict], dict]:
    manifest = pd.DataFrame(
        json.loads(
            (RESULTS / "equivalence/manifest.json").read_text(encoding="utf-8")
        )
    )
    frames = [
        pd.read_csv(path)
        for path in sorted(
            (RESULTS / "shape_states").glob("seed_*/shape_statistics.csv")
        )
    ]
    statistics = pd.concat(frames, ignore_index=True)
    merged = manifest.merge(statistics, on=["seed", "state_index"], how="left")
    if merged.isna().any().any():
        raise RuntimeError("numerical manifest does not map to shape statistics")
    merged["noise_bin"] = pd.cut(
        merged["sampling_step"],
        bins=[-1, 332, 665, 1000],
        labels=["high", "mid", "low"],
    )
    picks = []
    grouped = merged.groupby(
        ["num_atoms", "phase", "noise_bin"],
        observed=True,
        sort=True,
    )
    for _, group in grouped:
        count = min(2, len(group))
        indices = np.linspace(0, len(group) - 1, count, dtype=np.int64)
        picks.extend(group.iloc[indices].to_dict("records"))
    extremes = pd.concat(
        [
            merged.nlargest(4, "edge_count"),
            merged.nlargest(4, "triplet_count"),
            merged.nlargest(4, "cell_condition_number"),
            merged.nsmallest(4, "cell_volume"),
        ],
        ignore_index=True,
    ).to_dict("records")
    picks.extend(extremes)
    unique = {}
    for row in picks:
        unique[(int(row["seed"]), int(row["state_index"]))] = row
    if len(unique) < STATE_COUNT:
        remaining = merged.loc[
            ~merged.set_index(["seed", "state_index"]).index.isin(unique)
        ]
        needed = STATE_COUNT - len(unique)
        indices = np.linspace(0, len(remaining) - 1, needed, dtype=np.int64)
        for row in remaining.iloc[indices].to_dict("records"):
            unique[(int(row["seed"]), int(row["state_index"]))] = row
    selected_rows = list(unique.values())[:STATE_COUNT]
    coverage = {
        "states": len(selected_rows),
        "num_atoms": sorted({int(row["num_atoms"]) for row in selected_rows}),
        "phases": sorted({str(row["phase"]) for row in selected_rows}),
        "noise_bins": sorted({str(row["noise_bin"]) for row in selected_rows}),
        "sampling_step_min": min(int(row["sampling_step"]) for row in selected_rows),
        "sampling_step_max": max(int(row["sampling_step"]) for row in selected_rows),
        "edge_count_min": min(int(row["edge_count"]) for row in selected_rows),
        "edge_count_max": max(int(row["edge_count"]) for row in selected_rows),
        "triplet_count_min": min(int(row["triplet_count"]) for row in selected_rows),
        "triplet_count_max": max(int(row["triplet_count"]) for row in selected_rows),
    }
    return selected_rows, coverage


def joint_batch(sampler, batch):
    unconditional = sampler._remove_conditioning_fn(batch)
    conditional = sampler._keep_conditioning_fn(batch)
    joint = collate([unconditional, conditional])
    for attr, value in unconditional.items():
        if isinstance(value, list):
            joint[attr] = unconditional[attr] + conditional[attr]
    return joint


def run_joint_score(diffusion_module, sampler, batch, timestep):
    joint = joint_batch(sampler, batch)
    combined = diffusion_module.score_fn(
        joint,
        torch.cat([timestep, timestep], dim=0),
    )
    unconditional = combined[0]
    conditional = combined[1]
    final = unconditional.replace(
        **{
            field: torch.lerp(
                unconditional[field],
                conditional[field],
                sampler._guidance_scale,
            )
            for field in sampler._multi_corruption.corrupted_fields
        }
    )
    return {
        "unconditional": unconditional,
        "conditional": conditional,
        "final_cfg": final,
    }


def main() -> int:
    selected, coverage = select_states()
    set_stage(
        "gemnet_numerical_equivalence",
        "running",
        "Comparing 64 stratified real joint-CFG states and GemNet blocks.",
        {"atol": ATOL, "rtol": RTOL, "coverage": coverage},
    )
    configure_determinism()
    generator = build_c0_generator(sampling_steps=1000)
    sampler = build_recording_sampler(generator, int(selected[0]["seed"]))
    gemnet = find_gemnet(generator.model)
    diffusion_module = generator.model.diffusion_module
    device = next(gemnet.parameters()).device
    builder = StaticPeriodicGraphBuilder(
        StaticBucketConfig.from_json(RESULTS / "selected_bucket.json"), device
    )
    capture = {"target": None}

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            if capture["target"] is not None:
                capture["target"][name] = flatten_tensors(output)

        return hook

    hooks = [
        gemnet.atom_latent_emb.register_forward_hook(make_hook("input_node")),
        gemnet.angle_edge_emb.register_forward_hook(make_hook("input_edge")),
    ]
    hooks.extend(
        block.register_forward_hook(make_hook(f"block_{index + 1}"))
        for index, block in enumerate(gemnet.int_blocks)
    )
    state_cache: dict[int, list[dict]] = {}
    rows = []
    try:
        with torch.inference_mode():
            for item in selected:
                seed = int(item["seed"])
                if seed not in state_cache:
                    state_cache[seed] = torch.load(
                        RESULTS / f"shape_states/seed_{seed}/states.pt",
                        map_location="cpu",
                        weights_only=False,
                    )
                state = state_cache[seed][int(item["state_index"])]
                batch = singleton_condition(seed).replace(
                    pos=state["pos"],
                    cell=state["cell"],
                    atomic_numbers=state["atomic_numbers"],
                    num_atoms=state["num_atoms"],
                ).to(device)
                timestep = torch.tensor(
                    [state["t"]], dtype=torch.float32, device=device
                )
                dynamic_features = {}
                capture["target"] = dynamic_features
                dynamic = run_joint_score(
                    diffusion_module, sampler, batch.clone(), timestep
                )
                static_features = {}
                capture["target"] = static_features
                counters = {}
                with install_static_builder(gemnet, builder, counters):
                    static = run_joint_score(
                        diffusion_module, sampler, batch.clone(), timestep
                    )
                capture["target"] = None
                metrics = {}
                for location in sorted(
                    set(dynamic_features) | set(static_features)
                ):
                    reference_tensors = dynamic_features.get(location, {})
                    actual_tensors = static_features.get(location, {})
                    for name in sorted(
                        set(reference_tensors) | set(actual_tensors)
                    ):
                        key = f"{location}.{name}"
                        if name in reference_tensors and name in actual_tensors:
                            metrics[key] = tensor_metrics(
                                reference_tensors[name], actual_tensors[name]
                            )
                        else:
                            metrics[key] = tensor_metrics(
                                torch.empty(0, device=device),
                                torch.empty(1, device=device),
                            )
                for branch in ("unconditional", "conditional", "final_cfg"):
                    for field in FIELDS:
                        metrics[f"{branch}.{field}"] = tensor_metrics(
                            dynamic[branch][field], static[branch][field]
                        )
                rows.append(
                    {
                        "seed": seed,
                        "state_index": int(state["state_index"]),
                        "sampling_step": int(state["sampling_step"]),
                        "phase": str(state["phase"]),
                        "num_atoms": int(state["num_atoms"].sum()),
                        "static_calls": int(counters.get("static", 0)),
                        "fallback_calls": int(counters.get("fallback", 0)),
                        "metrics": metrics,
                    }
                )
    finally:
        for hook in hooks:
            hook.remove()
    values = [metric for row in rows for metric in row["metrics"].values()]
    first_nonzero = next(
        (
            {
                "seed": row["seed"],
                "state_index": row["state_index"],
                "location": name,
                "metrics": metric,
            }
            for row in rows
            for name, metric in row["metrics"].items()
            if metric["max_absolute_error"] > 0.0
        ),
        None,
    )
    first_failure = next(
        (
            {
                "seed": row["seed"],
                "state_index": row["state_index"],
                "location": name,
                "metrics": metric,
            }
            for row in rows
            for name, metric in row["metrics"].items()
            if not metric["allclose"]
        ),
        None,
    )
    per_location = {}
    for name in sorted({name for row in rows for name in row["metrics"]}):
        location_values = [row["metrics"][name] for row in rows]
        per_location[name] = {
            "bitwise_rate": sum(v["bitwise_equal"] for v in location_values)
            / len(location_values),
            "max_absolute_error": max(v["max_absolute_error"] for v in location_values),
            "max_relative_l2_error": max(v["relative_l2_error"] for v in location_values),
            "min_cosine_similarity": min(v["cosine_similarity"] for v in location_values),
        }
    passed = (
        len(rows) == STATE_COUNT
        and first_failure is None
        and all(row["static_calls"] == 1 for row in rows)
        and all(row["fallback_calls"] == 0 for row in rows)
    )
    summary = {
        "completed_at": now(),
        "states": len(rows),
        "coverage": coverage,
        "atol": ATOL,
        "rtol": RTOL,
        "static_calls": sum(row["static_calls"] for row in rows),
        "fallback_calls": sum(row["fallback_calls"] for row in rows),
        "bitwise_rate": sum(value["bitwise_equal"] for value in values) / len(values),
        "max_absolute_error": max(value["max_absolute_error"] for value in values),
        "max_relative_l2_error": max(value["relative_l2_error"] for value in values),
        "min_cosine_similarity": min(value["cosine_similarity"] for value in values),
        "first_nonzero": first_nonzero,
        "first_failure": first_failure,
        "per_location": per_location,
        "GEMNET_NUMERICAL_EQUIVALENT": passed,
        "JOINT_CFG_NUMERICAL_EQUIVALENT": passed,
        "passed": passed,
        "rows": rows,
    }
    output = RESULTS / "numerical/numerical_equivalence.json"
    atomic_json(output, summary)
    report_lines = [
        "# GemNet joint-CFG numerical equivalence",
        "",
        f"- States: `{summary['states']}`",
        f"- Coverage: `{coverage}`",
        f"- Static/fallback calls: `{summary['static_calls']}/{summary['fallback_calls']}`",
        f"- Bitwise tensor rate: `{summary['bitwise_rate']:.6%}`",
        f"- Maximum absolute error: `{summary['max_absolute_error']}`",
        f"- Maximum relative L2 error: `{summary['max_relative_l2_error']}`",
        f"- Minimum cosine similarity: `{summary['min_cosine_similarity']}`",
        f"- First nonzero location: `{summary['first_nonzero']}`",
        f"- First tolerance failure: `{summary['first_failure']}`",
        f"- GEMNET_NUMERICAL_EQUIVALENT: `{passed}`",
        f"- JOINT_CFG_NUMERICAL_EQUIVALENT: `{passed}`",
        "",
        "| Location | Max abs error | Relative L2 | Min cosine | Bitwise rate |",
        "|---|---:|---:|---:|---:|",
    ]
    report_lines.extend(
        "| {name} | {max_absolute_error:.6g} | {max_relative_l2_error:.6g} | "
        "{min_cosine_similarity:.9f} | {bitwise_rate:.3%} |".format(
            name=name, **metrics
        )
        for name, metrics in per_location.items()
    )
    atomic_text(REPORTS / "numerical_equivalence.md", "\n".join(report_lines) + "\n")
    compact = {key: value for key, value in summary.items() if key != "rows"}
    set_stage(
        "gemnet_numerical_equivalence",
        "success" if passed else "failed",
        "Completed 64-state joint-CFG and GemNet block numerical equivalence.",
        compact,
    )
    print(json.dumps(compact, indent=2))
    if not passed:
        raise RuntimeError(f"numerical equivalence failed: {first_failure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
