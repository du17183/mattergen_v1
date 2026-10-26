"""Identity-audit and frozen-backbone training for Cell--Atom Fusion P0."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mattergen.cell_atom_fusion import CellAtomBidirectionalFusion
from mattergen.common.data.collate import collate
from mattergen.common.data.dataset import CrystalDatasetBuilder
from mattergen.common.data.transform import symmetrize_lattice
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.common.utils.eval_utils import load_model_diffusion


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/cell_atom_fusion_p0"
MODEL_ROOT = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/"
    "hf_mattergen/checkpoints/dft_mag_density"
)
DATA_ROOT = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/"
    "global_transformer_adapter_p0/data/cache"
)
EXPECTED_CHECKPOINT_SHA256 = "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e"
TRAINING_SEED = 20260912
FIELD_NAMES = ("atomic_numbers", "pos", "cell")


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


def digest_frozen_state(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if ".cell_atom_fusion." in name:
            continue
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def field_differences(left, right) -> dict[str, float]:
    return {
        field: float((left[field].float() - right[field].float()).abs().max().item())
        for field in FIELD_NAMES
    }


def loss_on_corruption(diffusion, clean, noisy, t):
    output = diffusion.model(noisy, t)
    return diffusion.loss_fn(
        multi_corruption=diffusion.corruption,
        batch=clean,
        noisy_batch=noisy,
        score_model_output=output,
        t=t,
    )


def fresh_training_loss(diffusion, batch):
    clean = diffusion.pre_corruption_fn(batch)
    noisy, t = diffusion._corrupt_batch(clean)
    total, fields = loss_on_corruption(diffusion, clean, noisy, t)
    return total, fields


@torch.no_grad()
def build_fixed_validation(diffusion, loader, device, seed: int):
    seed_everything(seed)
    fixed = []
    for batch in loader:
        clean = diffusion.pre_corruption_fn(batch.to(device))
        noisy, t = diffusion._corrupt_batch(clean)
        fixed.append((clean, noisy, t))
    return fixed


@torch.no_grad()
def validate_fixed(diffusion, fixed_validation) -> dict[str, float]:
    totals: list[float] = []
    fields: dict[str, list[float]] = {name: [] for name in FIELD_NAMES}
    for clean, noisy, t in fixed_validation:
        total, batch_fields = loss_on_corruption(diffusion, clean, noisy, t)
        totals.append(float(total.item()))
        for field in FIELD_NAMES:
            fields[field].append(float(batch_fields[field].item()))
    return {
        "loss_total": float(np.mean(totals)),
        **{f"loss_{field}": float(np.mean(values)) for field, values in fields.items()},
    }


@torch.no_grad()
def fit_cell_normalization(diffusion, loader, device, seed: int) -> dict:
    """Fit only on noisy cells drawn from the fixed training partition."""
    seed_everything(seed)
    feature_rows = []
    for batch in loader:
        clean = diffusion.pre_corruption_fn(batch.to(device))
        noisy, _ = diffusion._corrupt_batch(clean)
        feature_rows.append(
            CellAtomBidirectionalFusion.raw_cell_features(noisy.cell).detach().cpu()
        )
    features = torch.cat(feature_rows, dim=0)
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False).clamp_min(1e-6)
    return {
        "mean": mean,
        "std": std,
        "sample_count": int(len(features)),
        "raw_feature_min": features.min(dim=0).values.tolist(),
        "raw_feature_max": features.max(dim=0).values.tolist(),
        "normalized_abs_max": float(((features - mean) / std).abs().max().item()),
        "all_finite": bool(torch.isfinite(features).all()),
    }


@torch.no_grad()
def strict_identity_check(diffusion, gemnet, fusion, fixed_batch) -> dict:
    clean, noisy, t = fixed_batch
    diffusion.eval()
    # The official path is evaluated before the new module is registered.
    baseline_first = diffusion.model(noisy, t)
    baseline_repeat = diffusion.model(noisy, t)
    gemnet.cell_atom_fusion = fusion
    enabled = diffusion.model(noisy, t)
    enabled_repeat = diffusion.model(noisy, t)
    official_repeat = field_differences(baseline_first, baseline_repeat)
    enabled_difference = field_differences(baseline_first, enabled)
    enabled_repeat_difference = field_differences(enabled, enabled_repeat)

    # Directly verify the algebraic identity independently of the surrounding
    # CUDA graph kernels as well.
    h = torch.randn(13, 512, device=noisy.cell.device)
    graph_batch = torch.tensor([0] * 5 + [1] * 8, device=h.device)
    num_atoms = torch.tensor([5, 8], device=h.device)
    lattice = noisy.cell[:2]
    module_output = fusion(
        h=h, lattice=lattice, batch=graph_batch, num_atoms=num_atoms, block_index=2
    )
    module_max_difference = float((module_output - h).abs().max().item())
    passed = (
        float(fusion.alpha_a2c.item()) == 0.0
        and float(fusion.alpha_c2a.item()) == 0.0
        and module_max_difference == 0.0
        and all(value <= 1e-5 for value in enabled_difference.values())
    )
    return {
        "alpha_a2c": float(fusion.alpha_a2c.item()),
        "alpha_c2a": float(fusion.alpha_c2a.item()),
        "module_max_abs_difference": module_max_difference,
        "official_repeat_max_abs_difference": official_repeat,
        "fusion_vs_official_max_abs_difference": enabled_difference,
        "fusion_repeat_max_abs_difference": enabled_repeat_difference,
        "required_max_abs_difference": 1e-5,
        "passed": passed,
    }


def relative_improvement(baseline: float, candidate: float) -> float:
    return float((baseline - candidate) / baseline)


def validation_signal(baseline: dict[str, float], candidate: dict[str, float]) -> dict:
    improvements = {
        name: relative_improvement(baseline[f"loss_{name}"], candidate[f"loss_{name}"])
        for name in ("total", *FIELD_NAMES)
    }
    other_guardrail = {
        "atomic_numbers": improvements["atomic_numbers"] >= -0.005,
        "pos": improvements["pos"] >= -0.005,
        "cell": improvements["cell"] >= -0.005,
    }
    pos_signal = improvements["pos"] >= 0.02 and all(
        other_guardrail[name] for name in ("atomic_numbers", "cell")
    )
    cell_signal = improvements["cell"] >= 0.02 and all(
        other_guardrail[name] for name in ("atomic_numbers", "pos")
    )
    return {
        "relative_improvement": improvements,
        "total_gate": improvements["total"] >= 0.005,
        "position_gate": pos_signal,
        "cell_gate": cell_signal,
        "validation_go_signal": bool(
            improvements["total"] >= 0.005 or pos_signal or cell_signal
        ),
    }


def write_curve(path: Path, rows: list[dict]) -> None:
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def train(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("real Cell--Atom Fusion training requires CUDA")
    checkpoint_path = MODEL_ROOT / "checkpoints/last.ckpt"
    checkpoint_hash = sha256(checkpoint_path)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"official checkpoint hash mismatch: {checkpoint_hash}")
    output = ROOT / "checkpoints"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint directory: {output}")

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    seed_everything(args.seed)
    device = torch.device("cuda")

    train_data = CleanDataset(DATA_ROOT / "train")
    val_data = CleanDataset(DATA_ROOT / "val")
    if (len(train_data), len(val_data)) != (1024, 128):
        raise RuntimeError("expected fixed 1024 train / 128 validation split")
    shuffle_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=0,
        collate_fn=collate, generator=shuffle_generator,
    )
    normalization_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=collate,
    )

    # Official weights are loaded strictly before the optional module exists.
    pl_module = load_model_diffusion(
        MatterGenCheckpointInfo(model_path=str(MODEL_ROOT.resolve()))
    ).to(device)
    diffusion = pl_module.diffusion_module
    gemnet = diffusion.model.gemnet
    if getattr(gemnet, "cell_atom_fusion", None) is not None:
        raise RuntimeError("official checkpoint unexpectedly contains Cell--Atom Fusion")
    pl_module.eval()
    for parameter in pl_module.parameters():
        parameter.requires_grad_(False)

    normalization = fit_cell_normalization(
        diffusion, normalization_loader, device, args.seed + 1000
    )
    if not normalization["all_finite"]:
        raise RuntimeError("non-finite training-only cell normalization features")
    fusion = CellAtomBidirectionalFusion(
        cell_feature_mean=normalization["mean"],
        cell_feature_std=normalization["std"],
    ).to(device)

    # Corrupt validation exactly once, before the identity comparison. Every
    # later validation therefore sees identical samples, timesteps, and noise.
    fixed_validation = build_fixed_validation(
        diffusion, val_loader, device, args.seed + 1101
    )
    identity = strict_identity_check(
        diffusion, gemnet, fusion, fixed_validation[0]
    )
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "identity_check.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not identity["passed"]:
        raise RuntimeError(f"strict identity check failed: {identity}")

    for parameter in pl_module.parameters():
        parameter.requires_grad_(False)
    for parameter in fusion.parameters():
        parameter.requires_grad_(True)
    trainable_names = [
        name for name, parameter in pl_module.named_parameters() if parameter.requires_grad
    ]
    if not trainable_names or not all(
        ".cell_atom_fusion." in name for name in trainable_names
    ):
        raise RuntimeError(f"unexpected trainable parameter set: {trainable_names}")
    trainable_params = sum(parameter.numel() for parameter in fusion.parameters())
    total_params = sum(parameter.numel() for parameter in pl_module.parameters())
    base_params = total_params - trainable_params
    if trainable_params >= 1_000_000 or trainable_params / base_params > 0.05:
        raise RuntimeError("fusion parameter budget exceeded")

    frozen_before = digest_frozen_state(pl_module)
    initial_state = {
        name: value.detach().cpu().clone() for name, value in fusion.state_dict().items()
    }
    baseline_validation = validate_fixed(diffusion, fixed_validation)
    rows: list[dict] = [{"split": "validation_baseline", "step": 0, **baseline_validation}]
    print(json.dumps({
        "event": "audit_complete",
        "identity": identity,
        "base_params": base_params,
        "new_trainable_params": trainable_params,
        "trainable_percent_of_base": 100.0 * trainable_params / base_params,
        "normalization": {
            key: value.tolist() if isinstance(value, torch.Tensor) else value
            for key, value in normalization.items()
        },
        "baseline_validation": baseline_validation,
    }), flush=True)

    optimizer = torch.optim.AdamW(
        fusion.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    iterator = iter(train_loader)
    best_validation = None
    best_step = None
    best_state = None
    first_gradient_check = None
    torch.cuda.reset_peak_memory_stats()
    seed_everything(args.seed + 2000)
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        pl_module.eval()
        fusion.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        total, fields = fresh_training_loss(diffusion, batch.to(device))
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite loss at step {step}: {total.item()}")
        total.backward()
        gradients = {
            name: parameter.grad for name, parameter in fusion.named_parameters()
            if parameter.grad is not None
        }
        frozen_gradients = [
            name for name, parameter in pl_module.named_parameters()
            if ".cell_atom_fusion." not in name and parameter.grad is not None
        ]
        if frozen_gradients:
            raise RuntimeError(f"frozen parameter received gradient: {frozen_gradients[:3]}")
        if step == 1:
            first_gradient_check = {
                "parameters_with_gradient": sorted(gradients),
                "nonzero_gradient_parameters": sorted(
                    name for name, gradient in gradients.items()
                    if float(gradient.detach().float().norm().item()) > 0
                ),
                "all_gradients_finite": all(
                    bool(torch.isfinite(gradient).all()) for gradient in gradients.values()
                ),
                "expected_staged_zero_init": (
                    "only alpha_c2a receives a nonzero first-step gradient; downstream "
                    "MLPs begin receiving signal after that scalar leaves zero"
                ),
                "frozen_parameters_with_gradient": frozen_gradients,
            }
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), args.gradient_clip).item()
        )
        if not np.isfinite(gradient_norm):
            raise RuntimeError(f"non-finite gradient norm at step {step}")
        optimizer.step()
        torch.cuda.synchronize()
        stats = dict(fusion.last_stats)
        row = {
            "split": "train", "step": step,
            "loss_total": float(total.item()),
            **{f"loss_{name}": float(fields[name].item()) for name in FIELD_NAMES},
            "gradient_norm": gradient_norm,
            "step_seconds": time.perf_counter() - step_started,
            **stats,
        }
        rows.append(row)

        if step % args.validation_every == 0 or step == args.steps:
            pl_module.eval()
            fusion.eval()
            validation = validate_fixed(diffusion, fixed_validation)
            validation_row = {"split": "validation", "step": step, **validation,
                              **fusion.last_stats}
            rows.append(validation_row)
            if best_validation is None or validation["loss_total"] < best_validation["loss_total"]:
                best_validation = validation
                best_step = step
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in fusion.state_dict().items()
                }
            print(json.dumps({
                "event": "validation", "step": step,
                "train_loss": row["loss_total"], "validation": validation,
                "signal": validation_signal(baseline_validation, validation),
                "alpha_a2c": fusion.last_stats.get("alpha_a2c"),
                "alpha_c2a": fusion.last_stats.get("alpha_c2a"),
                "residual_hidden_ratio": fusion.last_stats.get("residual_hidden_ratio"),
            }), flush=True)

    elapsed = time.perf_counter() - started
    if best_state is None or best_validation is None or best_step is None:
        raise RuntimeError("training ended without a validation checkpoint")
    frozen_after = digest_frozen_state(pl_module)
    if frozen_before != frozen_after:
        raise RuntimeError("frozen official model state changed")

    final_state = copy.deepcopy(fusion.state_dict())
    fusion.load_state_dict(best_state, strict=True)
    fusion.eval()
    best_recomputed = validate_fixed(diffusion, fixed_validation)
    if abs(best_recomputed["loss_total"] - best_validation["loss_total"]) > 1e-7:
        raise RuntimeError("best checkpoint validation is not reproducible")
    # Refresh mechanism statistics on a fixed held-out forward.
    loss_on_corruption(diffusion, *fixed_validation[0])
    best_mechanism = dict(fusion.last_stats)
    signal = validation_signal(baseline_validation, best_validation)
    train_rows = [row for row in rows if row["split"] == "train"]
    both_best_alphas_near_zero = (
        abs(best_mechanism["alpha_a2c"]) < 1e-2
        and abs(best_mechanism["alpha_c2a"]) < 1e-2
    )
    fusion_unused = bool(
        both_best_alphas_near_zero and best_mechanism["residual_hidden_ratio"] < 0.01
    )
    max_abs_alpha_a2c = max(abs(row["alpha_a2c"]) for row in train_rows)
    max_abs_alpha_c2a = max(abs(row["alpha_c2a"]) for row in train_rows)
    alphas_always_near_zero = max_abs_alpha_a2c < 1e-2 and max_abs_alpha_c2a < 1e-2
    training_signal_weak = bool(
        signal["relative_improvement"]["total"] < 0.002 and alphas_always_near_zero
    )
    val_clearly_worse = bool(
        signal["relative_improvement"]["total"] < -0.02
        and max(signal["relative_improvement"][name] for name in FIELD_NAMES) <= 0.0
    )

    changed_trainable = [
        name for name, value in final_state.items()
        if name in dict(fusion.named_parameters())
        and not torch.equal(value.detach().cpu(), initial_state[name])
    ]
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = {
        "architecture": fusion.audit_metadata,
        "fusion_state_dict": best_state,
        "base_model_root": str(MODEL_ROOT),
        "base_checkpoint_sha256": checkpoint_hash,
        "best_step": best_step,
        "training": {
            "steps": args.steps, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
            "gradient_clip": args.gradient_clip, "seed": args.seed,
            "validation_every": args.validation_every,
            "objective": "official MatterGen mixed-field diffusion loss",
            "frozen_backbone": True,
        },
    }
    torch.save(checkpoint, output / "best_fusion.pt")
    write_curve(output / "training_curve.csv", rows)
    summary = {
        "architecture": fusion.audit_metadata,
        "base_model": "Official dft_mag_density Full FT",
        "base_model_root": str(MODEL_ROOT),
        "base_checkpoint_sha256": checkpoint_hash,
        "data_root": str(DATA_ROOT),
        "train_samples": len(train_data), "validation_samples": len(val_data),
        "fixed_validation_noise": True,
        "identity_check": identity,
        "normalization": {
            key: value.tolist() if isinstance(value, torch.Tensor) else value
            for key, value in normalization.items()
        },
        "base_params": base_params, "new_params": trainable_params,
        "trainable_params": trainable_params,
        "trainable_percent_of_base": 100.0 * trainable_params / base_params,
        "trainable_parameter_names": trainable_names,
        "steps": args.steps, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "validation_every": args.validation_every,
        "baseline_validation": baseline_validation,
        "best_validation": best_validation, "best_step": best_step,
        "best_checkpoint_recomputed_validation": best_recomputed,
        "validation_signal": signal,
        "training_signal_weak": training_signal_weak,
        "fusion_unused": fusion_unused,
        "alpha_near_zero_abs_threshold": 0.01,
        "max_abs_alpha_a2c_during_training": max_abs_alpha_a2c,
        "max_abs_alpha_c2a_during_training": max_abs_alpha_c2a,
        "alphas_always_near_zero": alphas_always_near_zero,
        "validation_clearly_worse": val_clearly_worse,
        "best_mechanism": best_mechanism,
        "first_gradient_check": first_gradient_check,
        "changed_trainable_parameter_count": len(changed_trainable),
        "trainable_parameter_tensor_count": len(list(fusion.named_parameters())),
        "frozen_digest_before": frozen_before,
        "frozen_digest_after": frozen_after,
        "frozen_parameters_unchanged": frozen_before == frozen_after,
        "all_train_losses_finite": bool(all(np.isfinite(row["loss_total"]) for row in train_rows)),
        "first_50_train_loss_mean": float(np.mean([row["loss_total"] for row in train_rows[:50]])),
        "last_50_train_loss_mean": float(np.mean([row["loss_total"] for row in train_rows[-50:]])),
        "mean_step_seconds": float(np.mean([row["step_seconds"] for row in train_rows])),
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "checkpoint_path": str((output / "best_fusion.pt").resolve()),
        "checkpoint_sha256": sha256(output / "best_fusion.pt"),
        "generation_allowed": bool(not val_clearly_worse),
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "training_complete", **summary}, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=TRAINING_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
