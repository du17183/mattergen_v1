"""Evaluate C0, FT0, TCL, and DML with identical original-loss corruptions."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mattergen.common.data.collate import collate
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.common.utils.eval_utils import load_model_diffusion
from train_method import CleanDataset, disable_historical_modules, seed_everything


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/tcl_dml_p0"
DATA_ROOT = PROJECT_ROOT / "experiments/global_transformer_adapter_p0/data/cache"
MODEL_ROOT = PROJECT_ROOT / "checkpoints/official/hf_mattergen/checkpoints/dft_mag_density"
METHODS = ("C0", "FT0", "TCL", "DML")
VALIDATION_SEED = 20260929


@torch.no_grad()
def evaluate(pl_module, loader, method: str, device: torch.device) -> dict:
    pl_module.eval()
    seed_everything(VALIDATION_SEED)
    totals = []
    fields = {name: [] for name in ("atomic_numbers", "pos", "cell")}
    for batch in loader:
        total, metrics = pl_module.diffusion_module.calc_loss(batch.to(device))
        totals.append(float(total.item()))
        for field in fields:
            fields[field].append(float(metrics[field].item()))
    return {
        "method": method,
        "n_validation": len(loader.dataset),
        "corruption_seed": VALIDATION_SEED,
        "loss_total": float(np.mean(totals)),
        **{f"loss_{field}": float(np.mean(values)) for field, values in fields.items()},
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("real-checkpoint validation requires CUDA")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    dataset = CleanDataset(DATA_ROOT / "val")
    loader = DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate
    )
    pl_module = load_model_diffusion(
        MatterGenCheckpointInfo(model_path=str(MODEL_ROOT.resolve()))
    ).to(device)
    disable_historical_modules(pl_module)
    base_state = {
        key: value.detach().cpu().clone() for key, value in pl_module.state_dict().items()
    }
    rows = []
    for method in METHODS:
        if method == "C0":
            pl_module.load_state_dict(base_state, strict=True)
        else:
            checkpoint = torch.load(
                ROOT / "checkpoints" / method / "model.pt",
                map_location="cpu", weights_only=False,
            )
            if checkpoint["method"] != method:
                raise RuntimeError(f"checkpoint method mismatch: {method}")
            pl_module.load_state_dict(checkpoint["model_state_dict"], strict=True)
        disable_historical_modules(pl_module)
        rows.append(evaluate(pl_module, loader, method, device))
    references = {row["method"]: row for row in rows}
    for row in rows:
        for metric in ("loss_total", "loss_atomic_numbers", "loss_pos", "loss_cell"):
            row[f"delta_{metric}_vs_C0"] = row[metric] - references["C0"][metric]
            row[f"delta_{metric}_vs_FT0"] = row[metric] - references["FT0"][metric]
    with (ROOT / "validation_summary.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    import pandas as pd
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
