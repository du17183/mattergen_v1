"""Small real-checkpoint backward diagnostic for TCL objective attribution.

Four existing validation batches are evaluated without an optimizer or any
parameter update.  The script measures gradient norms from the original loss
and each weighted TCL field term at the frozen TCL checkpoint.
"""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from mattergen.common.data.collate import collate
from mattergen.common.data.dataset import CrystalDatasetBuilder
from mattergen.common.data.transform import symmetrize_lattice
from mattergen.diffusion.training.cross_timestep_consistency import consistency_losses
from mattergen.diffusion.training.two_view import (
    corrupt_two_views,
    fixed_weight_loss,
    per_sample_field_losses,
    sample_timestep_pair,
)
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.common.utils.eval_utils import load_model_diffusion


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "experiments/tcl_dml_p0/checkpoints/TCL/model.pt"
DATA = ROOT / "experiments/global_transformer_adapter_p0/data/cache/val"
MODEL_ROOT = ROOT / "checkpoints/official/hf_mattergen/checkpoints/dft_mag_density"
LAMBDA = 0.1


class CleanDataset(Dataset):
    def __init__(self, cache_path: Path) -> None:
        self.base = CrystalDatasetBuilder.from_cache_path(
            str(cache_path), transforms=[symmetrize_lattice], properties=["dft_mag_density"]
        ).build()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        return self.base[index]


def category(name):
    short = name.removeprefix("diffusion_module.model.")
    if short.startswith("fc_atom."):
        return "atomic_head"
    if short.startswith(("gemnet.lattice_out_blocks.", "gemnet.mlp_rbf_lattice.")):
        return "cell_head"
    if short.startswith("gemnet.out_blocks."):
        return "position_head"
    if short.startswith(("gemnet.cond_adapt_layers.", "gemnet.cond_mixin_layers.", "property_embeddings")):
        return "condition_modules"
    for i in range(4):
        if short.startswith(f"gemnet.int_blocks.{i}."):
            return f"gemnet_block_{i + 1}"
    if short.startswith(("gemnet.atom_emb.", "gemnet.atom_latent_emb.", "gemnet.edge_emb.", "gemnet.angle_edge_emb.", "noise_level_encoding.")):
        return "embedding_input"
    return "other_shared"


def grad_stats(grads, named_parameters, base_grads=None):
    groups = {}
    total_sq = total_dot = base_sq = 0.0
    for grad, base, (name, _parameter) in zip(grads, base_grads, named_parameters):
        if grad is None or base is None:
            continue
        g = grad.detach()
        b = base.detach()
        group = category(name)
        item = groups.setdefault(group, {"sq": 0.0, "dot": 0.0, "base_sq": 0.0})
        square = float(g.double().square().sum())
        dot = float((g.double() * b.double()).sum())
        bsq = float(b.double().square().sum())
        item["sq"] += square
        item["dot"] += dot
        item["base_sq"] += bsq
        total_sq += square
        total_dot += dot
        base_sq += bsq
    groups["ALL_PARAMETERS"] = {"sq": total_sq, "dot": total_dot, "base_sq": base_sq}
    return groups


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260916)
    torch.cuda.manual_seed_all(20260916)
    device = torch.device("cuda")
    model = load_model_diffusion(MatterGenCheckpointInfo(model_path=str(Path(MODEL_ROOT).resolve()))).to(device)
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    del payload
    model.train()
    named_parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    parameters = [p for _, p in named_parameters]
    loader = DataLoader(CleanDataset(DATA), batch_size=8, shuffle=False, num_workers=0, collate_fn=collate)
    rows = []
    for batch_index, batch in enumerate(loader):
        if batch_index == 4:
            break
        batch = batch.to(device)
        diffusion = model.diffusion_module
        clean = diffusion.pre_corruption_fn(batch)
        t_low, t_high = sample_timestep_pair(diffusion, clean, min_normalized_gap=0.2)
        views = corrupt_two_views(diffusion, clean, t_low, t_high)
        low = diffusion.model(views.low, views.t_low)
        high = diffusion.model(views.high, views.t_high)
        fields_low = per_sample_field_losses(diffusion.loss_fn, diffusion.corruption, clean, views.low, low, views.t_low)
        fields_high = per_sample_field_losses(diffusion.loss_fn, diffusion.corruption, clean, views.high, high, views.t_high)
        base_loss = 0.5 * (fixed_weight_loss(diffusion.loss_fn, fields_low) + fixed_weight_loss(diffusion.loss_fn, fields_high))
        consistency = consistency_losses(
            corruption=diffusion.corruption, clean=clean, low=views.low, high=views.high,
            output_low=low, output_high=high, t_low=views.t_low, t_high=views.t_high,
        )
        objectives = {
            "base_original": base_loss,
            "tcl_atomic_weighted": LAMBDA * consistency["atomic_consistency"] / 3.0,
            "tcl_position_weighted": LAMBDA * consistency["position_consistency"] / 3.0,
            "tcl_cell_weighted": LAMBDA * consistency["cell_consistency"] / 3.0,
        }
        objectives["total"] = sum(objectives.values())
        base_grads = torch.autograd.grad(objectives["base_original"], parameters, retain_graph=True, allow_unused=True)
        for objective, loss in objectives.items():
            grads = base_grads if objective == "base_original" else torch.autograd.grad(loss, parameters, retain_graph=objective != "total", allow_unused=True)
            grouped = grad_stats(grads, named_parameters, base_grads)
            total_norm = np.sqrt(grouped["ALL_PARAMETERS"]["sq"])
            for group, values in grouped.items():
                norm = np.sqrt(values["sq"])
                denom = np.sqrt(values["sq"] * values["base_sq"])
                rows.append({
                    "batch": batch_index,
                    "objective": objective,
                    "category": group,
                    "loss_value": float(loss.detach()),
                    "gradient_l2_norm": norm,
                    "fraction_total_gradient_sq": values["sq"] / grouped["ALL_PARAMETERS"]["sq"] if grouped["ALL_PARAMETERS"]["sq"] else np.nan,
                    "cosine_with_base_gradient": values["dot"] / denom if denom else np.nan,
                    "t_low_mean": float(views.t_low.mean()),
                    "t_high_mean": float(views.t_high.mean()),
                })
            print("batch", batch_index, objective, "loss", float(loss.detach()), "grad", total_norm, flush=True)
        del low, high, fields_low, fields_high, base_grads
        gc.collect()
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "gradient_attribution.csv", index=False)
    summary = frame.groupby(["objective", "category"], as_index=False).agg(
        n=("batch", "count"),
        loss_mean=("loss_value", "mean"),
        gradient_l2_norm_mean=("gradient_l2_norm", "mean"),
        gradient_l2_norm_median=("gradient_l2_norm", "median"),
        gradient_sq_fraction_mean=("fraction_total_gradient_sq", "mean"),
        cosine_with_base_mean=("cosine_with_base_gradient", "mean"),
    )
    summary.to_csv(OUT / "gradient_attribution_summary.csv", index=False)
    print(summary[summary.category == "ALL_PARAMETERS"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
