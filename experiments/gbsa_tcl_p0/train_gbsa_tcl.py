"""Train one real-checkpoint GBSA-TCL student against a frozen C0 teacher."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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
from mattergen.diffusion.training.two_view import (
    corrupt_two_views,
    fixed_weight_loss,
    mean_field_metrics,
    per_sample_field_losses,
    sample_timestep_pair,
)

from objectives import gbsa_objectives


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/gbsa_tcl_p0"
DATA_ROOT = PROJECT_ROOT / "experiments/global_transformer_adapter_p0/data/cache"
MODEL_ROOT = PROJECT_ROOT / "checkpoints/official/hf_mattergen/checkpoints/dft_mag_density"
METHOD = "GBSA-TCL"
EXPECTED_BASE_HASH = "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e"
AUXILIARIES = ("atomic_tcl", "position_tcl", "position_anchor", "cell_anchor")
TRAINING_SEED = 20260917
VALIDATION_SEED = 20260927
DIAGNOSTIC_INTERVAL = 25
BALANCE_TARGET_RATIO = 1.0
HARD_GUARDRAIL_RATIO = 3.0
CLIP_NORM = 1.0


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
        train_data, batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=collate, generator=generator,
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=collate,
    )
    return train_loader, val_loader


def configure_partial_training(student):
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    named = dict(student.named_parameters())
    selected = []
    for name, parameter in named.items():
        is_condition = (
            "diffusion_module.model.gemnet.cond_adapt_layers." in name
            or "diffusion_module.model.gemnet.cond_mixin_layers." in name
        )
        is_atomic_head = "diffusion_module.model.fc_atom." in name
        if is_condition or is_atomic_head:
            parameter.requires_grad_(True)
            selected.append(name)
    if not selected:
        raise RuntimeError("no trainable parameters selected")
    forbidden = [
        name for name, parameter in student.named_parameters()
        if parameter.requires_grad and not (
            "diffusion_module.model.gemnet.cond_adapt_layers." in name
            or "diffusion_module.model.gemnet.cond_mixin_layers." in name
            or "diffusion_module.model.fc_atom." in name
        )
    ]
    if forbidden:
        raise RuntimeError(f"unexpected trainable parameters: {forbidden}")
    return sorted(selected)


def parameter_summary(student, trainable_names: list[str]) -> dict:
    total = sum(parameter.numel() for parameter in student.parameters())
    trainable = sum(
        parameter.numel() for parameter in student.parameters()
        if parameter.requires_grad
    )
    groups = {"condition_adapter_mixin": 0, "atomic_head": 0}
    for name, parameter in student.named_parameters():
        if not parameter.requires_grad:
            continue
        if "cond_adapt_layers." in name or "cond_mixin_layers." in name:
            groups["condition_adapter_mixin"] += parameter.numel()
        elif "fc_atom." in name:
            groups["atomic_head"] += parameter.numel()
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_ratio": trainable / total,
        "frozen_params": total - trainable,
        "trainable_tensor_count": len(trainable_names),
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_names_sha256": names_hash(trainable_names),
        "trainable_groups": groups,
    }


@torch.no_grad()
def validate_original(student, loader, device: torch.device) -> dict[str, float]:
    student.eval()
    totals = []
    fields = {field: [] for field in ("atomic_numbers", "pos", "cell")}
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(VALIDATION_SEED)
        torch.cuda.manual_seed_all(VALIDATION_SEED)
        for batch in loader:
            total, metrics = student.diffusion_module.calc_loss(batch.to(device))
            totals.append(float(total.item()))
            for field in fields:
                fields[field].append(float(metrics[field].item()))
    return {
        "loss_total": float(np.mean(totals)),
        **{f"loss_{field}": float(np.mean(values)) for field, values in fields.items()},
    }


def grad_norm(grads) -> float:
    square = 0.0
    for gradient in grads:
        if gradient is not None:
            square += float(gradient.detach().double().square().sum())
    return float(np.sqrt(square))


def grad_cosine(grads, base_grads) -> float:
    dot = left = right = 0.0
    for gradient, base in zip(grads, base_grads):
        if gradient is None or base is None:
            continue
        g = gradient.detach().double()
        b = base.detach().double()
        dot += float((g * b).sum())
        left += float(g.square().sum())
        right += float(b.square().sum())
    if left == 0.0 or right == 0.0:
        return float("nan")
    return dot / np.sqrt(left * right)


def gradient_diagnostic(
    *, step: int, base_loss, configured, parameters, balance_scales,
    raw_losses, t_low_mean: float, t_high_mean: float,
) -> list[dict]:
    base_grads = torch.autograd.grad(
        base_loss, parameters, retain_graph=True, allow_unused=True
    )
    base_norm = grad_norm(base_grads)
    if not np.isfinite(base_norm) or base_norm <= 0:
        raise RuntimeError(f"invalid base gradient norm at step {step}: {base_norm}")
    rows = [{
        "step": step,
        "objective": "base",
        "raw_loss": float(base_loss.detach()),
        "configured_weighted_loss": float(base_loss.detach()),
        "balance_scale": 1.0,
        "balanced_weighted_loss": float(base_loss.detach()),
        "raw_configured_gradient_l2": base_norm,
        "balanced_gradient_l2": base_norm,
        "gradient_over_base": 1.0,
        "cosine_vs_base": 1.0,
        "t_low_mean": t_low_mean,
        "t_high_mean": t_high_mean,
    }]
    for name in AUXILIARIES:
        loss = configured[name]
        grads = torch.autograd.grad(
            loss, parameters, retain_graph=True, allow_unused=True
        )
        raw_norm = grad_norm(grads)
        if not np.isfinite(raw_norm):
            raise RuntimeError(f"non-finite {name} gradient at step {step}")
        scale = min(1.0, BALANCE_TARGET_RATIO * base_norm / max(raw_norm, 1e-12))
        balance_scales[name] = scale
        balanced_norm = raw_norm * scale
        ratio = balanced_norm / base_norm
        if ratio > HARD_GUARDRAIL_RATIO + 1e-6:
            raise RuntimeError(f"{name} gradient guardrail violated: {ratio}")
        rows.append({
            "step": step,
            "objective": name,
            "raw_loss": float(raw_losses[name].detach()),
            "configured_weighted_loss": float(loss.detach()),
            "balance_scale": scale,
            "balanced_weighted_loss": float((scale * loss).detach()),
            "raw_configured_gradient_l2": raw_norm,
            "balanced_gradient_l2": balanced_norm,
            "gradient_over_base": ratio,
            "cosine_vs_base": grad_cosine(grads, base_grads),
            "t_low_mean": t_low_mean,
            "t_high_mean": t_high_mean,
        })
    return rows


def train(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("real GBSA-TCL training requires CUDA")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("high")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed_everything(args.seed)
    device = torch.device("cuda")
    train_loader, val_loader = build_loaders(args.batch_size, args.seed)
    base_checkpoint = MODEL_ROOT / "checkpoints/last.ckpt"
    if sha256(base_checkpoint) != EXPECTED_BASE_HASH:
        raise RuntimeError("official C0 checkpoint hash changed")

    student = load_model_diffusion(
        MatterGenCheckpointInfo(model_path=str(MODEL_ROOT.resolve()))
    ).to(device)
    teacher = load_model_diffusion(
        MatterGenCheckpointInfo(model_path=str(MODEL_ROOT.resolve()))
    ).to(device)
    disable_historical_modules(student)
    disable_historical_modules(teacher)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    trainable_names = configure_partial_training(student)
    params = parameter_summary(student, trainable_names)
    parameters = [
        parameter for parameter in student.parameters() if parameter.requires_grad
    ]
    if params["trainable_params"] != 4_250_213:
        raise RuntimeError(f"unexpected trainable count: {params['trainable_params']}")
    if params["total_params"] != 48_760_443:
        raise RuntimeError(f"unexpected total count: {params['total_params']}")
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    base_weights = dict(student.diffusion_module.loss_fn.loss_weights)
    if base_weights != {"pos": 0.1, "cell": 1.0, "atomic_numbers": 1.0}:
        raise RuntimeError(f"unexpected official field weights: {base_weights}")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    iterator = iter(train_loader)
    curve_rows = []
    diagnostic_rows = []
    balance_scales = {name: 1.0 for name in AUXILIARIES}
    clipping_count = 0
    nonfinite_count = 0
    initial_alignment = None
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = batch.to(device)
        student.train()
        teacher.eval()
        optimizer.zero_grad(set_to_none=True)
        step_started = time.perf_counter()
        diffusion = student.diffusion_module
        clean = diffusion.pre_corruption_fn(batch)
        t_low, t_high = sample_timestep_pair(
            diffusion, clean, min_normalized_gap=0.2
        )
        views = corrupt_two_views(diffusion, clean, t_low, t_high)

        with torch.no_grad():
            teacher_low = teacher.diffusion_module.model(views.low, views.t_low)
            teacher_high = teacher.diffusion_module.model(views.high, views.t_high)
        student_low = diffusion.model(views.low, views.t_low)
        student_high = diffusion.model(views.high, views.t_high)
        if step == 1:
            initial_alignment = {
                field: max(
                    float((student_low[field] - teacher_low[field]).abs().max()),
                    float((student_high[field] - teacher_high[field]).abs().max()),
                )
                for field in ("atomic_numbers", "pos", "cell")
            }
            if max(initial_alignment.values()) > 1e-5:
                raise RuntimeError(f"C0 student/teacher misalignment: {initial_alignment}")

        fields_low = per_sample_field_losses(
            diffusion.loss_fn, diffusion.corruption, clean,
            views.low, student_low, views.t_low,
        )
        fields_high = per_sample_field_losses(
            diffusion.loss_fn, diffusion.corruption, clean,
            views.high, student_high, views.t_high,
        )
        base_loss = 0.5 * (
            fixed_weight_loss(diffusion.loss_fn, fields_low)
            + fixed_weight_loss(diffusion.loss_fn, fields_high)
        )
        objectives = gbsa_objectives(
            diffusion=diffusion,
            clean=clean,
            views=views,
            student_low=student_low,
            student_high=student_high,
            teacher_low=teacher_low,
            teacher_high=teacher_high,
            step=step,
        )
        raw_losses = {name: objectives[f"raw_{name}"] for name in AUXILIARIES}
        configured = {
            name: objectives[f"weighted_{name}"] for name in AUXILIARIES
        }
        is_diagnostic = step == 1 or step % DIAGNOSTIC_INTERVAL == 0
        if is_diagnostic:
            rows = gradient_diagnostic(
                step=step,
                base_loss=base_loss,
                configured=configured,
                parameters=parameters,
                balance_scales=balance_scales,
                raw_losses=raw_losses,
                t_low_mean=float(views.t_low.mean()),
                t_high_mean=float(views.t_high.mean()),
            )
            diagnostic_rows.extend(rows)

        balanced = {
            name: balance_scales[name] * configured[name] for name in AUXILIARIES
        }
        total_loss = base_loss + sum(balanced.values())
        finite_losses = all(bool(torch.isfinite(value)) for value in (
            [base_loss, total_loss] + list(raw_losses.values()) + list(balanced.values())
        ))
        if not finite_losses:
            nonfinite_count += 1
            raise RuntimeError(f"non-finite loss at step {step}")
        total_loss.backward()
        all_gradients_finite = all(
            bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters if parameter.grad is not None
        )
        if not all_gradients_finite:
            nonfinite_count += 1
            raise RuntimeError(f"non-finite gradient at step {step}")
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, CLIP_NORM))
        clipped = gradient_norm > CLIP_NORM
        clipping_count += int(clipped)
        optimizer.step()

        field_metrics = mean_field_metrics(fields_low, fields_high)
        row = {
            "step": step,
            "loss_total": float(total_loss.detach()),
            "loss_base": float(base_loss.detach()),
            "loss_atomic_numbers": float(field_metrics["atomic_numbers"].detach()),
            "loss_pos": float(field_metrics["pos"].detach()),
            "loss_cell": float(field_metrics["cell"].detach()),
            "raw_atomic_tcl": float(raw_losses["atomic_tcl"].detach()),
            "raw_position_tcl": float(raw_losses["position_tcl"].detach()),
            "raw_position_anchor": float(raw_losses["position_anchor"].detach()),
            "raw_cell_anchor": float(raw_losses["cell_anchor"].detach()),
            "weighted_atomic_tcl": float(balanced["atomic_tcl"].detach()),
            "weighted_position_tcl": float(balanced["position_tcl"].detach()),
            "weighted_position_anchor": float(balanced["position_anchor"].detach()),
            "weighted_cell_anchor": float(balanced["cell_anchor"].detach()),
            "balance_atomic_tcl": balance_scales["atomic_tcl"],
            "balance_position_tcl": balance_scales["position_tcl"],
            "balance_position_anchor": balance_scales["position_anchor"],
            "balance_cell_anchor": balance_scales["cell_anchor"],
            "warmup_fraction": float(objectives["warmup_fraction"]),
            "preclip_gradient_l2": gradient_norm,
            "clip_triggered": clipped,
            "t_low_mean": float(views.t_low.mean()),
            "t_high_mean": float(views.t_high.mean()),
            "step_seconds": time.perf_counter() - step_started,
        }
        curve_rows.append(row)
        if step == 1 or step % args.validation_every == 0 or step == args.steps:
            validation = validate_original(student, val_loader, device)
            print(json.dumps({
                "method": METHOD,
                "step": step,
                "steps": args.steps,
                "train_total": row["loss_total"],
                "base_loss": row["loss_base"],
                "preclip_gradient_l2": gradient_norm,
                "clip_triggered": clipped,
                "clipping_count": clipping_count,
                "balance_scales": balance_scales,
                "validation": validation,
                "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
            }), flush=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    curve = pd.DataFrame(curve_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    curve.to_csv(output / "training_curve.csv", index=False)
    diagnostics.to_csv(output / "gradient_diagnostics.csv", index=False)
    checkpoint = {
        "method": METHOD,
        "model_state_dict": {
            key: value.detach().cpu() for key, value in student.state_dict().items()
        },
        "base_model_root": str(MODEL_ROOT.resolve()),
        "training": {
            "steps": args.steps,
            "effective_batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "optimizer": "AdamW",
            "seed": args.seed,
        },
    }
    checkpoint_path = output / "model.pt"
    torch.save(checkpoint, checkpoint_path)

    aux_diag = diagnostics[diagnostics.objective != "base"]
    base_diag = diagnostics[diagnostics.objective == "base"]
    summary = {
        "method": METHOD,
        "run_kind": args.run_kind,
        "base_model_root": str(MODEL_ROOT.resolve()),
        "base_checkpoint_sha256": EXPECTED_BASE_HASH,
        "data_root": str(DATA_ROOT.resolve()),
        "train_structures": 1024,
        "validation_structures": 128,
        "training_seed": args.seed,
        "steps": args.steps,
        "effective_batch_size": args.batch_size,
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        **params,
        "old_clean_cell_tcl_present": False,
        "atomic_c0_anchor_present": False,
        "teacher_frozen": all(not p.requires_grad for p in teacher.parameters()),
        "teacher_eval": not teacher.training,
        "teacher_optimizer_params": 0,
        "initial_student_teacher_max_abs": initial_alignment,
        "all_losses_and_gradients_finite": nonfinite_count == 0,
        "nonfinite_count": nonfinite_count,
        "gradient_diagnostic_interval": DIAGNOSTIC_INTERVAL,
        "base_gradient_l2_mean": float(base_diag.balanced_gradient_l2.mean()),
        "maximum_balanced_auxiliary_over_base": float(aux_diag.gradient_over_base.max()),
        "maximum_raw_configured_auxiliary_over_base": float(
            (aux_diag.raw_configured_gradient_l2.to_numpy()
             / np.repeat(base_diag.balanced_gradient_l2.to_numpy(), len(AUXILIARIES))).max()
        ),
        "gradient_by_objective": {
            name: {
                "raw_loss_mean": float(curve[f"raw_{name}"].mean()),
                "balanced_weighted_loss_mean": float(curve[f"weighted_{name}"].mean()),
                "raw_configured_gradient_l2_mean": float(
                    aux_diag[aux_diag.objective == name].raw_configured_gradient_l2.mean()
                ),
                "balanced_gradient_l2_mean": float(
                    aux_diag[aux_diag.objective == name].balanced_gradient_l2.mean()
                ),
                "gradient_over_base_mean": float(
                    aux_diag[aux_diag.objective == name].gradient_over_base.mean()
                ),
                "cosine_vs_base_mean": float(
                    aux_diag[aux_diag.objective == name].cosine_vs_base.mean()
                ),
            }
            for name in AUXILIARIES
        },
        "preclip_gradient_l2_mean": float(curve.preclip_gradient_l2.mean()),
        "preclip_gradient_l2_median": float(curve.preclip_gradient_l2.median()),
        "preclip_gradient_l2_max": float(curve.preclip_gradient_l2.max()),
        "gradient_clip_threshold": CLIP_NORM,
        "gradient_clipping_count": clipping_count,
        "gradient_clipping_fraction": clipping_count / args.steps,
        "first_50_base_loss_mean": float(curve.head(50).loss_base.mean()),
        "last_50_base_loss_mean": float(curve.tail(50).loss_base.mean()),
        "final_validation_original_fixed_loss": validation,
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }
    summary["checkpoint_sha256"] = sha256(checkpoint_path)
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (output / "training_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        row = {
            key: value for key, value in summary.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
        writer = csv.DictWriter(stream, fieldnames=list(row), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=TRAINING_SEED)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-kind", choices=("short_diagnostic", "main"), required=True)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
