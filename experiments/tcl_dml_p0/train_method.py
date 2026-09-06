"""Train one independent full-finetuning FT0, TCL, or DML model."""
from __future__ import annotations

import argparse
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
from mattergen.diffusion.training.cross_timestep_consistency import (
    consistency_losses,
    consistency_weight,
)
from mattergen.diffusion.training.dynamic_multifield_loss import (
    FIELD_ORDER,
    NoiseAwareFieldScheduler,
    dynamic_weighted_loss,
    multiplier_anchor,
)
from mattergen.diffusion.training.two_view import (
    corrupt_two_views,
    fixed_weight_loss,
    mean_field_metrics,
    per_sample_field_losses,
    sample_timestep_pair,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/tcl_dml_p0"
DATA_ROOT = PROJECT_ROOT / "experiments/global_transformer_adapter_p0/data/cache"
MODEL_ROOT = PROJECT_ROOT / "checkpoints/official/hf_mattergen/checkpoints/dft_mag_density"
METHODS = ("FT0", "TCL", "DML")
TRAINING_SEED = 20260908
VALIDATION_SEED = 20260928


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def names_hash(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def disable_historical_modules(pl_module) -> None:
    gemnet = pl_module.diffusion_module.model.gemnet
    for attribute in ("global_adapter", "cross_field_adapter", "quality_adapter"):
        setattr(gemnet, attribute, None)


def build_loaders(batch_size: int, seed: int):
    train_data = CleanDataset(DATA_ROOT / "train")
    val_data = CleanDataset(DATA_ROOT / "val")
    if (len(train_data), len(val_data)) != (1024, 128):
        raise RuntimeError("expected reused 1024 train / 128 validation split")
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        generator=generator,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    return train_loader, val_loader


@torch.no_grad()
def validate_original(pl_module, loader, device: torch.device) -> dict[str, float]:
    pl_module.eval()
    totals = []
    fields = {field: [] for field in FIELD_ORDER}
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(VALIDATION_SEED)
        torch.cuda.manual_seed_all(VALIDATION_SEED)
        for batch in loader:
            total, metrics = pl_module.diffusion_module.calc_loss(batch.to(device))
            totals.append(float(total.item()))
            for field in fields:
                fields[field].append(float(metrics[field].item()))
    return {
        "loss_total": float(np.mean(totals)),
        **{f"loss_{field}": float(np.mean(values)) for field, values in fields.items()},
    }


def make_routing_summary(timesteps: list[np.ndarray], multipliers: list[np.ndarray]) -> dict:
    if not timesteps:
        return {}
    t = np.concatenate(timesteps)
    r = np.concatenate(multipliers)
    bins = np.linspace(0.0, 1.0, 6)
    fields = {}
    for index, field in enumerate(FIELD_ORDER):
        values = r[:, index]
        by_bin = []
        for lower, upper in zip(bins[:-1], bins[1:]):
            mask = (t >= lower) & (t < upper if upper < 1.0 else t <= upper)
            by_bin.append({
                "range": [float(lower), float(upper)],
                "count": int(mask.sum()),
                "mean": float(values[mask].mean()) if mask.any() else None,
            })
        boundary_fraction = float(((values <= 0.51) | (values >= 1.49)).mean())
        fields[field] = {
            "min": float(values.min()),
            "mean": float(values.mean()),
            "max": float(values.max()),
            "boundary_fraction": boundary_fraction,
            "timestep_bins": by_bin,
        }
    return {
        "fields": fields,
        "weight_routing_collapse": any(
            value["boundary_fraction"] >= 0.9 for value in fields.values()
        ),
    }


def train(args: argparse.Namespace) -> None:
    if args.method not in METHODS:
        raise ValueError(args.method)
    if not torch.cuda.is_available():
        raise RuntimeError("full-finetuning P0 requires CUDA")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("high")
    seed_everything(args.seed)
    device = torch.device("cuda")
    train_loader, val_loader = build_loaders(args.batch_size, args.seed)

    pl_module = load_model_diffusion(
        MatterGenCheckpointInfo(model_path=str(MODEL_ROOT.resolve()))
    ).to(device)
    disable_historical_modules(pl_module)
    scheduler = NoiseAwareFieldScheduler(hidden_dim=16).to(device)
    for parameter in pl_module.parameters():
        parameter.requires_grad_(True)
    for parameter in scheduler.parameters():
        parameter.requires_grad_(True)

    model_names = sorted(
        f"model.{name}" for name, parameter in pl_module.named_parameters()
        if parameter.requires_grad
    )
    scheduler_names = sorted(
        f"objective.scheduler.{name}" for name, parameter in scheduler.named_parameters()
        if parameter.requires_grad
    )
    trainable_names = model_names + scheduler_names
    model_parameter_count = sum(parameter.numel() for parameter in pl_module.parameters())
    scheduler_parameter_count = sum(parameter.numel() for parameter in scheduler.parameters())
    trainable_parameter_count = model_parameter_count + scheduler_parameter_count
    base_weights = dict(pl_module.diffusion_module.loss_fn.loss_weights)
    if base_weights != {"pos": 0.1, "cell": 1.0, "atomic_numbers": 1.0}:
        raise RuntimeError(f"unexpected official field weights: {base_weights}")

    parameters = list(pl_module.parameters()) + list(scheduler.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    iterator = iter(train_loader)
    curve_rows = []
    routing_t = []
    routing_r = []
    all_finite = True
    first_gradient = None
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = batch.to(device)
        pl_module.train()
        scheduler.train()
        optimizer.zero_grad(set_to_none=True)
        step_started = time.perf_counter()
        diffusion = pl_module.diffusion_module
        clean = diffusion.pre_corruption_fn(batch)
        t_low, t_high = sample_timestep_pair(diffusion, clean, min_normalized_gap=0.2)
        views = corrupt_two_views(diffusion, clean, t_low, t_high)
        output_low = diffusion.model(views.low, views.t_low)
        output_high = diffusion.model(views.high, views.t_high)
        fields_low = per_sample_field_losses(
            diffusion.loss_fn, diffusion.corruption, clean,
            views.low, output_low, views.t_low,
        )
        fields_high = per_sample_field_losses(
            diffusion.loss_fn, diffusion.corruption, clean,
            views.high, output_high, views.t_high,
        )
        base_loss = 0.5 * (
            fixed_weight_loss(diffusion.loss_fn, fields_low)
            + fixed_weight_loss(diffusion.loss_fn, fields_high)
        )
        lambda_cons = 0.0
        consistency = {
            "consistency": base_loss.new_zeros(()),
            "atomic_consistency": base_loss.new_zeros(()),
            "position_consistency": base_loss.new_zeros(()),
            "cell_consistency": base_loss.new_zeros(()),
        }
        dynamic_loss = base_loss
        anchor_loss = base_loss.new_zeros(())
        if args.method == "TCL":
            consistency = consistency_losses(
                corruption=diffusion.corruption,
                clean=clean,
                low=views.low,
                high=views.high,
                output_low=output_low,
                output_high=output_high,
                t_low=views.t_low,
                t_high=views.t_high,
            )
            lambda_cons = consistency_weight(step, maximum=0.1, warmup_steps=100)
            total_loss = base_loss + lambda_cons * consistency["consistency"]
        elif args.method == "DML":
            dynamic_low, multiplier_low, _ = dynamic_weighted_loss(
                scheduler, fields_low, views.t_low, base_weights
            )
            dynamic_high, multiplier_high, _ = dynamic_weighted_loss(
                scheduler, fields_high, views.t_high, base_weights
            )
            dynamic_loss = 0.5 * (dynamic_low + dynamic_high)
            anchor_loss = multiplier_anchor(multiplier_low, multiplier_high, coefficient=0.01)
            total_loss = dynamic_loss + anchor_loss
            routing_t.append(torch.cat((views.t_low, views.t_high)).detach().cpu().numpy())
            routing_r.append(torch.cat((multiplier_low, multiplier_high)).detach().cpu().numpy())
        else:
            total_loss = base_loss

        if not bool(torch.isfinite(total_loss)):
            raise RuntimeError(f"non-finite loss at step {step}: {total_loss.item()}")
        total_loss.backward()
        if step == 1:
            model_gradients = [
                parameter.grad for parameter in pl_module.parameters()
                if parameter.grad is not None
            ]
            scheduler_gradients = [
                parameter.grad for parameter in scheduler.parameters()
                if parameter.grad is not None
            ]
            first_gradient = {
                "model_tensors_with_gradient": len(model_gradients),
                "model_gradients_finite": all(
                    bool(torch.isfinite(gradient).all()) for gradient in model_gradients
                ),
                "scheduler_tensors_with_gradient": len(scheduler_gradients),
                "scheduler_gradients_finite": all(
                    bool(torch.isfinite(gradient).all()) for gradient in scheduler_gradients
                ),
            }
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0).item())
        if not np.isfinite(gradient_norm):
            raise RuntimeError(f"non-finite gradient norm at step {step}")
        optimizer.step()
        field_metrics = mean_field_metrics(fields_low, fields_high)
        gap = views.t_high - views.t_low
        row = {
            "method": args.method,
            "split": "train",
            "step": step,
            "loss_total": float(total_loss.detach().item()),
            "loss_base": float(base_loss.detach().item()),
            "loss_dynamic": float(dynamic_loss.detach().item()),
            "loss_atomic_numbers": float(field_metrics["atomic_numbers"].detach().item()),
            "loss_pos": float(field_metrics["pos"].detach().item()),
            "loss_cell": float(field_metrics["cell"].detach().item()),
            "consistency": float(consistency["consistency"].detach().item()),
            "atomic_consistency": float(consistency["atomic_consistency"].detach().item()),
            "position_consistency": float(consistency["position_consistency"].detach().item()),
            "cell_consistency": float(consistency["cell_consistency"].detach().item()),
            "lambda_cons": lambda_cons,
            "weight_anchor": float(anchor_loss.detach().item()),
            "t_low_mean": float(views.t_low.mean().item()),
            "t_high_mean": float(views.t_high.mean().item()),
            "t_gap_min": float(gap.min().item()),
            "t_gap_mean": float(gap.mean().item()),
            "gradient_norm": gradient_norm,
            "step_seconds": time.perf_counter() - step_started,
        }
        curve_rows.append(row)
        all_finite = all_finite and all(np.isfinite(value) for value in (
            row["loss_total"], row["loss_base"], row["gradient_norm"]
        ))
        if step == 1 or step % args.validation_every == 0 or step == args.steps:
            validation = validate_original(pl_module, val_loader, device)
            curve_rows.append({
                "method": args.method,
                "split": "validation",
                "step": step,
                **validation,
            })
            print(json.dumps({
                "method": args.method,
                "step": step,
                "train_total": row["loss_total"],
                "base_loss": row["loss_base"],
                "consistency": row["consistency"],
                "validation": validation,
                "gradient_norm": gradient_norm,
                "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
            }), flush=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    output = ROOT / "checkpoints" / args.method
    output.mkdir(parents=True, exist_ok=False)
    curve = pd.DataFrame(curve_rows)
    curve.to_csv(output / "training_curve.csv", index=False)
    checkpoint = {
        "method": args.method,
        "model_state_dict": {
            key: value.detach().cpu() for key, value in pl_module.state_dict().items()
        },
        "scheduler_state_dict": {
            key: value.detach().cpu() for key, value in scheduler.state_dict().items()
        },
        "base_model_root": str(MODEL_ROOT.resolve()),
        "training": {
            "steps": args.steps,
            "effective_batch_size": args.batch_size,
            "two_views_per_structure": 2,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "optimizer": "AdamW",
            "seed": args.seed,
        },
    }
    checkpoint_path = output / "model.pt"
    torch.save(checkpoint, checkpoint_path)
    train_rows = curve[curve["split"] == "train"]
    validation_rows = curve[curve["split"] == "validation"]
    final_train = train_rows.iloc[-1]
    final_validation = validation_rows.iloc[-1]
    summary = {
        "method": args.method,
        "base_model_root": str(MODEL_ROOT.resolve()),
        "data_root": str(DATA_ROOT.resolve()),
        "train_structures": 1024,
        "validation_structures": 128,
        "selection_seed": 20260907,
        "training_seed": args.seed,
        "steps": args.steps,
        "effective_batch_size": args.batch_size,
        "two_views_per_structure": 2,
        "minimum_normalized_timestep_gap": 0.2,
        "continuous_noise_coupled": True,
        "atomic_noise_coupled": False,
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "model_trainable_params": model_parameter_count,
        "scheduler_trainable_params_in_scope": scheduler_parameter_count,
        "total_trainable_params_in_scope": trainable_parameter_count,
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_names_sha256": names_hash(trainable_names),
        "base_field_weights": base_weights,
        "first_gradient_check": first_gradient,
        "all_losses_finite": all_finite,
        "first_50_base_loss_mean": float(train_rows.head(50)["loss_base"].mean()),
        "last_50_base_loss_mean": float(train_rows.tail(50)["loss_base"].mean()),
        "final_train": {
            key: float(final_train[key]) for key in (
                "loss_total", "loss_base", "loss_dynamic", "loss_atomic_numbers",
                "loss_pos", "loss_cell", "consistency", "atomic_consistency",
                "position_consistency", "cell_consistency", "lambda_cons", "weight_anchor",
            )
        },
        "final_validation_original_fixed_loss": {
            key: float(final_validation[key]) for key in (
                "loss_total", "loss_atomic_numbers", "loss_pos", "loss_cell"
            )
        },
        "routing": make_routing_summary(routing_t, routing_r),
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary["checkpoint_sha256"] = sha256(checkpoint_path)
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=TRAINING_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
