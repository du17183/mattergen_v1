"""Train the frozen-budget Cross-field Feature Interaction adapter."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from mattergen.common.data.collate import collate
from mattergen.common.data.dataset import CrystalDatasetBuilder
from mattergen.common.data.transform import symmetrize_lattice
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.common.utils.eval_utils import load_model_diffusion
from mattergen.cross_field_interaction import CrossFieldAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/cross_field_interaction_p0"
PREVIOUS_ROOT = PROJECT_ROOT / "experiments/global_transformer_adapter_p0"
MODEL_ROOT = PROJECT_ROOT / "checkpoints/official/hf_mattergen/checkpoints/dft_mag_density"
DATA_ROOT = PREVIOUS_ROOT / "data/cache"
TRAINING_SEED = 20260907


class CleanDataset(Dataset):
    def __init__(self, cache_path: Path) -> None:
        self.base = CrystalDatasetBuilder.from_cache_path(
            str(cache_path),
            transforms=[symmetrize_lattice],
            properties=["dft_mag_density"],
        ).build()

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        return self.base[index]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def digest_frozen(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if ".cross_field_adapter." in name:
            continue
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def parameter_norm(module: torch.nn.Module) -> float:
    return float(torch.sqrt(sum(
        parameter.detach().float().square().sum()
        for parameter in module.parameters()
    )).item())


def field_differences(left, right) -> dict[str, float]:
    return {
        field: float((left[field].float() - right[field].float()).abs().max().item())
        for field in ("atomic_numbers", "pos", "cell")
    }


def calculate_loss(pl_module, batch):
    diffusion = pl_module.diffusion_module
    clean = diffusion.pre_corruption_fn(batch)
    noisy, t = diffusion._corrupt_batch(clean)
    output = diffusion.model(noisy, t)
    return diffusion.loss_fn(
        multi_corruption=diffusion.corruption,
        batch=clean,
        noisy_batch=noisy,
        score_model_output=output,
        t=t,
    )


@torch.no_grad()
def validate(pl_module, loader, device) -> dict[str, float]:
    pl_module.eval()
    totals = []
    fields = {name: [] for name in ("atomic_numbers", "pos", "cell")}
    for batch in loader:
        total, batch_fields = calculate_loss(pl_module, batch.to(device))
        totals.append(float(total.item()))
        for name in fields:
            fields[name].append(float(batch_fields[name].item()))
    return {
        "loss_total": float(np.mean(totals)),
        **{f"loss_{name}": float(np.mean(values)) for name, values in fields.items()},
    }


@torch.no_grad()
def zero_init_check(pl_module, adapter, batch, seed: int) -> dict:
    diffusion = pl_module.diffusion_module
    pl_module.eval()
    seed_everything(seed)
    clean = diffusion.pre_corruption_fn(batch)
    noisy, t = diffusion._corrupt_batch(clean)
    with adapter.disabled():
        baseline_first = diffusion.model(noisy, t)
        baseline_repeat = diffusion.model(noisy, t)
    enabled = diffusion.model(noisy, t)
    repeat = field_differences(baseline_first, baseline_repeat)
    enabled_difference = field_differences(baseline_first, enabled)
    projection_max = {
        name: float(parameter.detach().abs().max().item())
        for name, parameter in adapter.named_parameters()
        if name.startswith("output_projection")
    }
    tolerances = {"atomic_numbers": 0.025, "pos": 0.005, "cell": 0.002}
    passed = all(value == 0.0 for value in projection_max.values()) and all(
        enabled_difference[field] <= max(tolerances[field], 2.0 * repeat[field])
        for field in tolerances
    )
    return {
        "projection_max_abs": projection_max,
        "disabled_repeat_max_abs_difference": repeat,
        "enabled_max_abs_difference": enabled_difference,
        "absolute_tolerances": tolerances,
        "passed": passed,
    }


def load_previous_mlp_row() -> dict:
    summary = json.loads(
        (PREVIOUS_ROOT / "checkpoints/MLP/training_summary.json").read_text()
    )
    curve = pd.read_csv(PREVIOUS_ROOT / "checkpoints/MLP/training_curve.csv")
    final_train = curve[curve["split"] == "train"].iloc[-1]
    final_validation = summary["final_validation"]
    return {
        "method": "MLP",
        "training_origin": "reused_global_transformer_p0",
        "trainable_params": summary["trainable_params"],
        "steps": summary["steps"],
        "batch_size": summary["batch_size"],
        "learning_rate": summary["learning_rate"],
        "first_50_train_loss_mean": summary["first_50_train_loss_mean"],
        "last_50_train_loss_mean": summary["last_50_train_loss_mean"],
        "final_train_loss": float(final_train["loss_total"]),
        "final_train_atomic_loss": float(final_train["loss_atomic_numbers"]),
        "final_train_pos_loss": float(final_train["loss_pos"]),
        "final_train_cell_loss": float(final_train["loss_cell"]),
        "final_validation_loss": final_validation["loss_total"],
        "final_validation_atomic_loss": final_validation["loss_atomic_numbers"],
        "final_validation_pos_loss": final_validation["loss_pos"],
        "final_validation_cell_loss": final_validation["loss_cell"],
        "all_losses_finite": summary["all_losses_finite"],
        "zero_init_passed": summary["zero_init_check"]["passed"],
        "frozen_parameters_unchanged": summary["frozen_parameters_unchanged"],
        "all_trainable_parameters_changed": (
            summary["changed_adapter_state_entry_count"]
            == summary["adapter_state_entry_count"]
        ),
        "elapsed_seconds": summary["elapsed_seconds"],
        "peak_cuda_memory_mb": summary["peak_cuda_memory_mb"],
        "checkpoint_sha256": summary.get("checkpoint_sha256", ""),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("real CFI training requires CUDA")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("high")
    seed_everything(args.seed)
    device = torch.device("cuda")
    train_data = CleanDataset(DATA_ROOT / "train")
    val_data = CleanDataset(DATA_ROOT / "val")
    if (len(train_data), len(val_data)) != (1024, 128):
        raise RuntimeError("expected reused 1024 train / 128 validation split")
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    pl_module = load_model_diffusion(
        MatterGenCheckpointInfo(model_path=str(MODEL_ROOT.resolve()))
    ).to(device)
    adapter = CrossFieldAdapter().to(device)
    pl_module.diffusion_module.model.gemnet.cross_field_adapter = adapter
    for parameter in pl_module.parameters():
        parameter.requires_grad_(False)
    for parameter in adapter.parameters():
        parameter.requires_grad_(True)
    trainable_names = [
        name for name, parameter in pl_module.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_names or not all(
        ".cross_field_adapter." in name for name in trainable_names
    ):
        raise RuntimeError(f"unexpected trainable parameters: {trainable_names[:5]}")
    trainable_params = sum(parameter.numel() for parameter in adapter.parameters())
    total_params = sum(parameter.numel() for parameter in pl_module.parameters())
    print(json.dumps({
        "method": "CFI",
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_names": trainable_names,
    }, indent=2), flush=True)

    first_val = next(iter(val_loader)).to(device)
    zero_check = zero_init_check(pl_module, adapter, first_val, args.seed + 10)
    if not zero_check["passed"]:
        raise RuntimeError(f"zero-init baseline recovery failed: {zero_check}")
    print(json.dumps({"zero_init": zero_check}, indent=2), flush=True)

    frozen_before = digest_frozen(pl_module)
    initial_parameters = {
        name: value.detach().cpu().clone()
        for name, value in adapter.named_parameters()
    }
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    iterator = iter(train_loader)
    rows = []
    first_gradient_check = None
    torch.cuda.reset_peak_memory_stats()
    seed_everything(args.seed + 100)
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = batch.to(device)
        pl_module.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        total, fields = calculate_loss(pl_module, batch)
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite loss at step {step}: {total.item()}")
        total.backward()
        adapter_gradients = {
            name: parameter.grad
            for name, parameter in adapter.named_parameters()
            if parameter.grad is not None
        }
        frozen_gradients = [
            name for name, parameter in pl_module.named_parameters()
            if ".cross_field_adapter." not in name and parameter.grad is not None
        ]
        if step == 1:
            first_gradient_check = {
                "parameters_with_gradient": sorted(adapter_gradients),
                "nonzero_gradient_parameters": sorted(
                    name for name, gradient in adapter_gradients.items()
                    if float(gradient.detach().float().norm().item()) > 0
                ),
                "all_gradients_finite": all(
                    bool(torch.isfinite(gradient).all())
                    for gradient in adapter_gradients.values()
                ),
                "frozen_parameters_with_gradient": frozen_gradients,
            }
        if frozen_gradients:
            raise RuntimeError(f"frozen parameter received gradient: {frozen_gradients[:3]}")
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0).item()
        )
        optimizer.step()
        torch.cuda.synchronize()
        row = {
            "method": "CFI",
            "split": "train",
            "step": step,
            "loss_total": float(total.item()),
            "loss_atomic_numbers": float(fields["atomic_numbers"].item()),
            "loss_pos": float(fields["pos"].item()),
            "loss_cell": float(fields["cell"].item()),
            "gradient_norm": gradient_norm,
            "parameter_norm": parameter_norm(adapter),
            "step_seconds": time.perf_counter() - step_started,
        }
        rows.append(row)
        if step == 1 or step % args.validation_every == 0 or step == args.steps:
            validation = validate(pl_module, val_loader, device)
            rows.append({
                "method": "CFI",
                "split": "validation",
                "step": step,
                **validation,
            })
            print(json.dumps({
                "method": "CFI",
                "step": step,
                "train_total": row["loss_total"],
                "validation": validation,
                "gradient_norm": gradient_norm,
            }), flush=True)

    elapsed = time.perf_counter() - started
    frozen_after = digest_frozen(pl_module)
    if frozen_before != frozen_after:
        raise RuntimeError("a frozen checkpoint parameter changed")
    changed_parameters = [
        name for name, parameter in adapter.named_parameters()
        if not torch.equal(parameter.detach().cpu(), initial_parameters[name])
    ]
    parameter_names = [name for name, _ in adapter.named_parameters()]
    if sorted(changed_parameters) != sorted(parameter_names):
        missing = sorted(set(parameter_names) - set(changed_parameters))
        raise RuntimeError(f"CFI parameters did not update: {missing}")

    output = ROOT / "checkpoints/CFI"
    output.mkdir(parents=True, exist_ok=False)
    curve = pd.DataFrame(rows)
    curve.to_csv(output / "training_curve.csv", index=False)
    checkpoint = {
        "architecture": "CFI",
        "adapter_state_dict": {
            key: value.detach().cpu() for key, value in adapter.state_dict().items()
        },
        "base_model_root": str(MODEL_ROOT.resolve()),
        "block_index": 1,
        "config": {
            "hidden_dim": 512,
            "time_dim": 512,
            "field_dim": 128,
            "fusion_dim": 256,
            "num_rbf": 32,
            "cutoff": 7.0,
        },
        "training": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": 1e-4,
            "seed": args.seed,
            "optimizer": "AdamW",
            "objective": "original MatterGen mixed-field diffusion loss",
            "data_root": str(DATA_ROOT.resolve()),
        },
    }
    torch.save(checkpoint, output / "adapter.pt")
    train_rows = curve[curve["split"] == "train"]
    validation_rows = curve[curve["split"] == "validation"]
    final_train = train_rows.iloc[-1]
    final_validation = {
        key: (int(value) if key == "step" else float(value))
        for key, value in validation_rows.iloc[-1].dropna().to_dict().items()
        if key not in {"method", "split"}
    }
    summary = {
        "method": "CFI",
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_parameter_names": trainable_names,
        "zero_init_check": zero_check,
        "first_gradient_check": first_gradient_check,
        "frozen_digest_before": frozen_before,
        "frozen_digest_after": frozen_after,
        "frozen_parameters_unchanged": frozen_before == frozen_after,
        "changed_parameter_count": len(changed_parameters),
        "trainable_parameter_tensor_count": len(parameter_names),
        "all_trainable_parameters_changed": True,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "first_50_train_loss_mean": float(train_rows.head(50)["loss_total"].mean()),
        "last_50_train_loss_mean": float(train_rows.tail(50)["loss_total"].mean()),
        "final_train": {
            "loss_total": float(final_train["loss_total"]),
            "loss_atomic_numbers": float(final_train["loss_atomic_numbers"]),
            "loss_pos": float(final_train["loss_pos"]),
            "loss_cell": float(final_train["loss_cell"]),
        },
        "final_validation": final_validation,
        "all_losses_finite": bool(np.isfinite(train_rows["loss_total"]).all()),
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    cfi_row = {
        "method": "CFI",
        "training_origin": "cross_field_interaction_p0",
        "trainable_params": trainable_params,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "first_50_train_loss_mean": summary["first_50_train_loss_mean"],
        "last_50_train_loss_mean": summary["last_50_train_loss_mean"],
        "final_train_loss": summary["final_train"]["loss_total"],
        "final_train_atomic_loss": summary["final_train"]["loss_atomic_numbers"],
        "final_train_pos_loss": summary["final_train"]["loss_pos"],
        "final_train_cell_loss": summary["final_train"]["loss_cell"],
        "final_validation_loss": final_validation["loss_total"],
        "final_validation_atomic_loss": final_validation["loss_atomic_numbers"],
        "final_validation_pos_loss": final_validation["loss_pos"],
        "final_validation_cell_loss": final_validation["loss_cell"],
        "all_losses_finite": summary["all_losses_finite"],
        "zero_init_passed": zero_check["passed"],
        "frozen_parameters_unchanged": summary["frozen_parameters_unchanged"],
        "all_trainable_parameters_changed": True,
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_mb": summary["peak_cuda_memory_mb"],
        "checkpoint_sha256": sha256(output / "adapter.pt"),
    }
    mlp_row = load_previous_mlp_row()
    mlp_row["checkpoint_sha256"] = sha256(
        PREVIOUS_ROOT / "checkpoints/MLP/adapter.pt"
    )
    result_path = ROOT / "training_summary.csv"
    with result_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(mlp_row), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows((mlp_row, cfi_row))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(pd.DataFrame((mlp_row, cfi_row)).to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=TRAINING_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
