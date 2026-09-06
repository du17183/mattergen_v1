"""Real-checkpoint P0 training for the static distribution-constrained adapter."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from mattergen.common.data.collate import collate
from mattergen.common.data.dataset import CrystalDatasetBuilder
from mattergen.common.data.transform import symmetrize_lattice
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.common.utils.eval_utils import load_model_diffusion
from mattergen.quality_adapter import (
    QualityAdapterStack,
    output_anchor_loss,
    weighted_materials_loss,
)


class QualityDataset(Dataset):
    def __init__(self, cache_path: Path, labels: pd.DataFrame, method: str) -> None:
        self.base = CrystalDatasetBuilder.from_cache_path(
            str(cache_path),
            transforms=[symmetrize_lattice],
            properties=["dft_mag_density"],
        ).build()
        labels = labels.sort_values("split_index")
        if labels["split_index"].tolist() != list(range(len(self.base))):
            raise RuntimeError("quality labels do not match the dataset cache")
        values = labels["quality_weight"].to_numpy(np.float32)
        self.weights = np.ones_like(values) if method == "M1" else values

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        return self.base[index].replace(
            quality_weight=torch.tensor([self.weights[index]], dtype=torch.float)
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def frozen_parameter_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if ".quality_adapter." in name:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def attach_adapter(pl_module, *, stage_aware: bool = False) -> QualityAdapterStack:
    stack = QualityAdapterStack(
        hidden_dim=512,
        bottleneck_dim=64,
        block_indices=(1, 2),
        stage_aware=stage_aware,
    )
    pl_module.diffusion_module.model.gemnet.quality_adapter = stack
    for parameter in pl_module.parameters():
        parameter.requires_grad_(False)
    for parameter in stack.parameters():
        parameter.requires_grad_(True)
    return stack


def field_differences(left, right) -> dict[str, float]:
    return {
        field: float((left[field].float() - right[field].float()).abs().max().item())
        for field in ("atomic_numbers", "pos", "cell")
    }


def zero_init_check(pl_module, stack: QualityAdapterStack, batch, seed: int):
    diffusion = pl_module.diffusion_module
    pl_module.eval()
    seed_everything(seed)
    with torch.no_grad():
        clean = diffusion.pre_corruption_fn(batch)
        noisy, t = diffusion._corrupt_batch(clean)
        with stack.disabled():
            baseline_first = diffusion.model(noisy, t)
            baseline_repeat = diffusion.model(noisy, t)
        enabled = diffusion.model(noisy, t)
    repeat_differences = field_differences(baseline_first, baseline_repeat)
    enabled_differences = field_differences(baseline_first, enabled)
    projection_max_abs = {
        name: float(parameter.detach().abs().max().item())
        for name, parameter in stack.named_parameters()
        if name.endswith("up.weight") or name.endswith("up.bias")
    }
    # GemNet GPU scatter reductions are not bitwise deterministic. Compare
    # against a disabled repeat; the adapter is analytically zero iff every
    # final up projection is exactly zero.
    tolerances = {"atomic_numbers": 0.02, "pos": 0.005, "cell": 0.001}
    passed = all(value == 0.0 for value in projection_max_abs.values()) and all(
        enabled_differences[field]
        <= max(tolerances[field], 2.0 * repeat_differences[field])
        for field in tolerances
    )
    return {
        "projection_max_abs": projection_max_abs,
        "disabled_repeat_max_abs_difference": repeat_differences,
        "enabled_max_abs_difference": enabled_differences,
        "absolute_tolerances": tolerances,
        "passed": passed,
    }


def calculate_batch_loss(
    pl_module,
    stack: QualityAdapterStack,
    batch,
    *,
    replay_fraction: float,
    anchor_lambda: float,
):
    diffusion = pl_module.diffusion_module
    clean = diffusion.pre_corruption_fn(batch)
    noisy, t = diffusion._corrupt_batch(clean)
    with stack.disabled(), torch.no_grad():
        teacher = diffusion.model(noisy, t)
    student = diffusion.model(noisy, t)
    diffusion_loss, metrics = weighted_materials_loss(
        base_loss=diffusion.loss_fn,
        multi_corruption=diffusion.corruption,
        batch=clean,
        noisy_batch=noisy,
        score_model_output=student,
        t=t,
        quality_weight=clean.quality_weight.flatten(),
        replay_fraction=replay_fraction,
    )
    anchor, anchor_metrics = output_anchor_loss(student, teacher)
    total = diffusion_loss + anchor_lambda * anchor
    return total, diffusion_loss, anchor, {**metrics, **anchor_metrics}


@torch.no_grad()
def validate(pl_module, stack, loader, device, replay_fraction, anchor_lambda):
    pl_module.eval()
    totals = []
    diffusion_values = []
    anchors = []
    for batch in loader:
        batch = batch.to(device)
        total, diffusion_loss, anchor, _ = calculate_batch_loss(
            pl_module,
            stack,
            batch,
            replay_fraction=replay_fraction,
            anchor_lambda=anchor_lambda,
        )
        totals.append(float(total.item()))
        diffusion_values.append(float(diffusion_loss.item()))
        anchors.append(float(anchor.item()))
    return {
        "loss_total": float(np.mean(totals)),
        "loss_diffusion": float(np.mean(diffusion_values)),
        "loss_anchor": float(np.mean(anchors)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("M1", "M2"), required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--anchor-lambda", type=float, default=0.05)
    parser.add_argument("--replay-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--validation-every", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("P0 real training requires CUDA")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("high")
    seed_everything(args.seed)
    device = torch.device("cuda")
    labels = pd.read_csv(args.labels)
    train_data = QualityDataset(
        args.data_root / "cache/train",
        labels[labels["split"] == "train"],
        args.method,
    )
    val_data = QualityDataset(
        args.data_root / "cache/val",
        labels[labels["split"] == "val"],
        args.method,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        generator=generator,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    info = MatterGenCheckpointInfo(model_path=str(args.model_root.resolve()))
    pl_module = load_model_diffusion(info).to(device)
    stack = attach_adapter(pl_module, stage_aware=False).to(device)
    total_params = sum(parameter.numel() for parameter in pl_module.parameters())
    trainable = [
        name for name, parameter in pl_module.named_parameters() if parameter.requires_grad
    ]
    trainable_params = sum(
        parameter.numel() for parameter in pl_module.parameters() if parameter.requires_grad
    )
    if trainable_params != 134272:
        raise RuntimeError(f"unexpected trainable parameter count: {trainable_params}")
    if not trainable or not all(".quality_adapter." in name for name in trainable):
        raise RuntimeError(f"unexpected trainable parameters: {trainable}")
    print(json.dumps({"total_params": total_params, "trainable_params": trainable_params, "trainable_names": trainable}, indent=2))

    first_val = next(iter(val_loader)).to(device)
    zero_check = zero_init_check(pl_module, stack, first_val, args.seed + 10)
    if not zero_check["passed"]:
        raise RuntimeError(f"zero-init did not recover baseline: {zero_check}")
    print(json.dumps({"zero_init_check": zero_check}, indent=2))

    before_digest = frozen_parameter_digest(pl_module)
    optimizer = torch.optim.AdamW(
        stack.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    iterator = iter(train_loader)
    rows: list[dict[str, object]] = []
    first_step_gradient_check = None
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
        optimizer.zero_grad(set_to_none=True)
        total, diffusion_loss, anchor, metrics = calculate_batch_loss(
            pl_module,
            stack,
            batch,
            replay_fraction=args.replay_fraction,
            anchor_lambda=args.anchor_lambda,
        )
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite loss at step {step}: {total.item()}")
        total.backward()
        adapter_grad_names = [
            name
            for name, parameter in pl_module.named_parameters()
            if ".quality_adapter." in name and parameter.grad is not None
        ]
        frozen_grad_names = [
            name
            for name, parameter in pl_module.named_parameters()
            if ".quality_adapter." not in name and parameter.grad is not None
        ]
        if step == 1:
            first_step_gradient_check = {
                "adapter_parameters_with_gradient": adapter_grad_names,
                "frozen_parameters_with_gradient": frozen_grad_names,
            }
        if frozen_grad_names:
            raise RuntimeError(f"frozen parameter received gradient: {frozen_grad_names[:3]}")
        grad_norm = float(torch.nn.utils.clip_grad_norm_(stack.parameters(), 1.0).item())
        optimizer.step()
        row = {
            "method": args.method,
            "split": "train",
            "step": step,
            "loss_total": float(total.item()),
            "loss_diffusion": float(diffusion_loss.item()),
            "loss_anchor": float(anchor.item()),
            "grad_norm": grad_norm,
            **{name: float(value.detach().item()) for name, value in metrics.items()},
        }
        rows.append(row)
        if step == 1 or step % args.validation_every == 0 or step == args.steps:
            validation = validate(
                pl_module,
                stack,
                val_loader,
                device,
                args.replay_fraction,
                args.anchor_lambda,
            )
            rows.append({"method": args.method, "split": "validation", "step": step, **validation})
            print(json.dumps({"step": step, "train_total": row["loss_total"], "validation": validation, "grad_norm": grad_norm}), flush=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    after_digest = frozen_parameter_digest(pl_module)
    if before_digest != after_digest:
        raise RuntimeError("a frozen MatterGen parameter changed during adapter training")
    last_grad_names = [
        name
        for name, parameter in pl_module.named_parameters()
        if ".quality_adapter." in name and parameter.grad is not None
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curve = pd.DataFrame(rows)
    curve.to_csv(args.output_dir / "training_curve.csv", index=False)
    checkpoint = {
        "method": args.method,
        "adapter_state_dict": {
            key: value.detach().cpu() for key, value in stack.state_dict().items()
        },
        "base_model_root": str(args.model_root.resolve()),
        "stage_aware": False,
        "block_indices": [1, 2],
        "hidden_dim": 512,
        "bottleneck_dim": 64,
        "trainable_parameter_names": trainable,
        "training": vars(args),
    }
    torch.save(checkpoint, args.output_dir / "adapter.pt")
    train_rows = curve[curve["split"] == "train"]
    summary = {
        "method": args.method,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_parameter_names": trainable,
        "zero_init_check": zero_check,
        "first_step_gradient_check": first_step_gradient_check,
        "last_step_adapter_parameters_with_gradient": last_grad_names,
        "frozen_parameter_digest_before": before_digest,
        "frozen_parameter_digest_after": after_digest,
        "frozen_parameters_unchanged": before_digest == after_digest,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "anchor_lambda": args.anchor_lambda,
        "replay_fraction": args.replay_fraction,
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "first_20_train_loss_mean": float(train_rows.head(20)["loss_total"].mean()),
        "last_20_train_loss_mean": float(train_rows.tail(20)["loss_total"].mean()),
        "final_validation": validate(pl_module, stack, val_loader, device, args.replay_fraction, args.anchor_lambda),
        "all_losses_finite": bool(np.isfinite(train_rows["loss_total"]).all()),
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
