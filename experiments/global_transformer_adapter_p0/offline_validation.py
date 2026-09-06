"""Compare C0, matched MLP, and Transformer on identical validation corruptions."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mattergen.common.data.collate import collate
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.common.utils.eval_utils import load_model_diffusion
from mattergen.global_transformer_adapter import build_global_adapter
from train_adapters import (
    CleanDataset,
    calculate_loss,
    seed_everything,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/global_transformer_adapter_p0"
MODEL_ROOT = PROJECT_ROOT / "checkpoints/official/hf_mattergen/checkpoints/dft_mag_density"
METHODS = ("C0", "MLP", "Transformer")
VALIDATION_SEED = 20260927


@torch.no_grad()
def evaluate(pl_module, loader, method: str, device: torch.device) -> dict[str, float | int | str]:
    if method == "C0":
        pl_module.diffusion_module.model.gemnet.global_adapter = None
    else:
        checkpoint = torch.load(
            ROOT / "checkpoints" / method / "adapter.pt",
            map_location="cpu",
            weights_only=False,
        )
        if checkpoint["architecture"] != method:
            raise RuntimeError(f"checkpoint architecture mismatch for {method}")
        adapter = build_global_adapter(method).to(device).eval()
        adapter.load_state_dict(checkpoint["adapter_state_dict"], strict=True)
        pl_module.diffusion_module.model.gemnet.global_adapter = adapter
    pl_module.eval()
    seed_everything(VALIDATION_SEED)
    totals: list[float] = []
    fields = {name: [] for name in ("atomic_numbers", "pos", "cell")}
    for batch in loader:
        total, batch_fields = calculate_loss(pl_module, batch.to(device))
        totals.append(float(total.item()))
        for name in fields:
            fields[name].append(float(batch_fields[name].item()))
    return {
        "method": method,
        "n_validation": len(loader.dataset),
        "corruption_seed": VALIDATION_SEED,
        "loss_total": float(np.mean(totals)),
        **{f"loss_{name}": float(np.mean(values)) for name, values in fields.items()},
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("offline real-checkpoint validation requires CUDA")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    dataset = CleanDataset(ROOT / "data/cache/val")
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate)
    pl_module = load_model_diffusion(
        MatterGenCheckpointInfo(model_path=str(MODEL_ROOT.resolve()))
    ).to(device)
    for parameter in pl_module.parameters():
        parameter.requires_grad_(False)
    rows = [evaluate(pl_module, loader, method, device) for method in METHODS]
    baseline = rows[0]
    mlp = rows[1]
    for row in rows:
        for metric in ("loss_total", "loss_atomic_numbers", "loss_pos", "loss_cell"):
            row[f"delta_{metric}_vs_C0"] = float(row[metric]) - float(baseline[metric])
            row[f"delta_{metric}_vs_MLP"] = float(row[metric]) - float(mlp[metric])
    output = ROOT / "validation_summary.csv"
    if output.exists():
        raise FileExistsError(output)
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    import pandas as pd
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
