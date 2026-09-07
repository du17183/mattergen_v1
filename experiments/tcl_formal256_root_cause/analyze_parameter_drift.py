"""Module-level parameter drift for the frozen C0/FT0/TCL checkpoints.

This is a read-only analysis of existing checkpoints.  It does not train or
modify a model.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import torch

from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.common.utils.eval_utils import load_model_diffusion


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT / "checkpoints/official/hf_mattergen/checkpoints/dft_mag_density/checkpoints/last.ckpt"
TRAINED = ROOT / "experiments/tcl_dml_p0/checkpoints"


def category(name: str) -> str:
    prefix = "diffusion_module.model."
    short = name[len(prefix):] if name.startswith(prefix) else name
    if short.startswith("fc_atom."):
        return "atomic_head"
    if short.startswith("gemnet.lattice_out_blocks.") or short.startswith("gemnet.mlp_rbf_lattice."):
        return "cell_head"
    if short.startswith("gemnet.out_blocks."):
        return "position_head"
    if (
        short.startswith("gemnet.cond_adapt_layers.")
        or short.startswith("gemnet.cond_mixin_layers.")
        or short.startswith("property_embeddings")
    ):
        return "condition_modules"
    for index in range(4):
        if short.startswith(f"gemnet.int_blocks.{index}."):
            return f"gemnet_block_{index + 1}"
    if short.startswith((
        "gemnet.atom_emb.", "gemnet.atom_latent_emb.", "gemnet.edge_emb.",
        "gemnet.angle_edge_emb.", "noise_level_encoding.",
    )):
        return "embedding_input"
    return "other_shared"


def load(path: Path, key: str) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload[key]


def accumulate(
    base: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    parameter_names: set[str],
    method: str,
):
    groups: dict[str, dict[str, float]] = {}
    for name, base_value in base.items():
        if name not in candidate or name not in parameter_names:
            continue
        value = candidate[name]
        if value.shape != base_value.shape:
            raise RuntimeError(f"shape mismatch for {name}")
        group = category(name)
        stats = groups.setdefault(group, {
            "parameter_count": 0,
            "tensor_count": 0,
            "base_sq": 0.0,
            "candidate_sq": 0.0,
            "delta_sq": 0.0,
            "dot": 0.0,
            "abs_delta": 0.0,
        })
        a = base_value.double().reshape(-1)
        b = value.double().reshape(-1)
        delta = b - a
        stats["parameter_count"] += a.numel()
        stats["tensor_count"] += 1
        stats["base_sq"] += float(torch.dot(a, a))
        stats["candidate_sq"] += float(torch.dot(b, b))
        stats["delta_sq"] += float(torch.dot(delta, delta))
        stats["dot"] += float(torch.dot(a, b))
        stats["abs_delta"] += float(delta.abs().sum())

    total = {key: sum(item[key] for item in groups.values()) for key in next(iter(groups.values()))}
    groups["ALL_TRAINABLE_PARAMETERS"] = total
    rows = []
    total_delta_sq = total["delta_sq"]
    for group, stats in groups.items():
        n = int(stats["parameter_count"])
        base_norm = math.sqrt(stats["base_sq"])
        cand_norm = math.sqrt(stats["candidate_sq"])
        delta_norm = math.sqrt(stats["delta_sq"])
        rows.append({
            "method": method,
            "category": group,
            "parameter_count": n,
            "tensor_count": int(stats["tensor_count"]),
            "l2_drift": delta_norm,
            "relative_l2_drift": delta_norm / base_norm if base_norm else float("nan"),
            "cosine_to_c0": stats["dot"] / (base_norm * cand_norm) if base_norm and cand_norm else float("nan"),
            "rms_parameter_drift": math.sqrt(stats["delta_sq"] / n) if n else float("nan"),
            "mean_abs_parameter_drift": stats["abs_delta"] / n if n else float("nan"),
            "fraction_of_total_drift_sq": stats["delta_sq"] / total_delta_sq if total_delta_sq else 0.0,
        })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint_root = BASE.parents[1]
    model = load_model_diffusion(
        MatterGenCheckpointInfo(model_path=str(checkpoint_root.resolve()))
    )
    parameter_names = {name for name, _ in model.named_parameters()}
    del model
    base = load(BASE, "state_dict")
    ft0 = load(TRAINED / "FT0/model.pt", "model_state_dict")
    tcl = load(TRAINED / "TCL/model.pt", "model_state_dict")
    if base.keys() != ft0.keys() or base.keys() != tcl.keys():
        raise RuntimeError("checkpoint state-dict namespaces differ")
    rows = (
        accumulate(base, ft0, parameter_names, "FT0_vs_C0")
        + accumulate(base, tcl, parameter_names, "TCL_vs_C0")
    )
    fieldnames = list(rows[0])
    with (OUT / "parameter_drift.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
