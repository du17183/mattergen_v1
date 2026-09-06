"""Train matched MLP and global Transformer adapters with one frozen budget."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
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
from mattergen.global_transformer_adapter import build_global_adapter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/global_transformer_adapter_p0"
MODEL_ROOT = PROJECT_ROOT / "checkpoints/official/hf_mattergen/checkpoints/dft_mag_density"
DATA_ROOT = ROOT / "data/cache"
ARCHITECTURES = ("MLP", "Transformer")
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
        if ".global_adapter." in name:
            continue
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def adapter_parameter_norm(adapter: torch.nn.Module) -> float:
    return float(
        torch.sqrt(
            sum(parameter.detach().float().square().sum() for parameter in adapter.parameters())
        ).item()
    )


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
    total, fields = diffusion.loss_fn(
        multi_corruption=diffusion.corruption,
        batch=clean,
        noisy_batch=noisy,
        score_model_output=output,
        t=t,
    )
    return total, fields


@torch.no_grad()
def validate(pl_module, loader, device) -> dict[str, float]:
    pl_module.eval()
    totals: list[float] = []
    fields: dict[str, list[float]] = {
        "atomic_numbers": [], "pos": [], "cell": []
    }
    for batch in loader:
        total, batch_fields = calculate_loss(pl_module, batch.to(device))
        totals.append(float(total.item()))
        for field in fields:
            fields[field].append(float(batch_fields[field].item()))
    return {
        "loss_total": float(np.mean(totals)),
        **{f"loss_{field}": float(np.mean(values)) for field, values in fields.items()},
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
    projection = {
        name: float(parameter.detach().abs().max().item())
        for name, parameter in adapter.named_parameters()
        if name.endswith("output_projection.weight")
        or name.endswith("output_projection.bias")
    }
    tolerances = {"atomic_numbers": 0.02, "pos": 0.005, "cell": 0.001}
    passed = all(value == 0.0 for value in projection.values()) and all(
        enabled_difference[field] <= max(tolerances[field], 2.0 * repeat[field])
        for field in tolerances
    )
    return {
        "projection_max_abs": projection,
        "disabled_repeat_max_abs_difference": repeat,
        "enabled_max_abs_difference": enabled_difference,
        "absolute_tolerances": tolerances,
        "passed": passed,
    }


def train_one(architecture: str, args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("real adapter training requires CUDA")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("high")
    seed_everything(args.seed)
    device = torch.device("cuda")
    train_data = CleanDataset(DATA_ROOT / "train")
    val_data = CleanDataset(DATA_ROOT / "val")
    if len(train_data) != 1024 or len(val_data) != 128:
        raise RuntimeError("expected fixed 1024 train / 128 validation split")
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=0,
        collate_fn=collate, generator=loader_generator,
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=collate,
    )

    pl_module = load_model_diffusion(
        MatterGenCheckpointInfo(model_path=str(MODEL_ROOT.resolve()))
    ).to(device)
    adapter = build_global_adapter(architecture).to(device)
    pl_module.diffusion_module.model.gemnet.global_adapter = adapter
    for parameter in pl_module.parameters():
        parameter.requires_grad_(False)
    for parameter in adapter.parameters():
        parameter.requires_grad_(True)
    trainable_names = [
        name for name, parameter in pl_module.named_parameters() if parameter.requires_grad
    ]
    trainable_params = sum(
        parameter.numel() for parameter in pl_module.parameters() if parameter.requires_grad
    )
    total_params = sum(parameter.numel() for parameter in pl_module.parameters())
    if not trainable_names or not all(
        ".global_adapter." in name for name in trainable_names
    ):
        raise RuntimeError(f"unexpected trainable parameters: {trainable_names[:5]}")
    print(json.dumps({
        "architecture": architecture, "total_params": total_params,
        "trainable_params": trainable_params, "trainable_names": trainable_names,
    }, indent=2), flush=True)

    first_val = next(iter(val_loader)).to(device)
    zero_check = zero_init_check(pl_module, adapter, first_val, args.seed + 10)
    if not zero_check["passed"]:
        raise RuntimeError(f"zero-init baseline recovery failed: {zero_check}")
    print(json.dumps({"zero_init": zero_check}, indent=2), flush=True)

    frozen_before = digest_frozen(pl_module)
    initial_adapter = {
        name: value.detach().cpu().clone() for name, value in adapter.state_dict().items()
    }
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    iterator = iter(train_loader)
    rows: list[dict] = []
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
        adapter_gradients = [
            name for name, parameter in pl_module.named_parameters()
            if ".global_adapter." in name and parameter.grad is not None
        ]
        frozen_gradients = [
            name for name, parameter in pl_module.named_parameters()
            if ".global_adapter." not in name and parameter.grad is not None
        ]
        if step == 1:
            first_gradient_check = {
                "adapter_parameters_with_gradient": adapter_gradients,
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
            "architecture": architecture, "split": "train", "step": step,
            "loss_total": float(total.item()),
            "loss_atomic_numbers": float(fields["atomic_numbers"].item()),
            "loss_pos": float(fields["pos"].item()),
            "loss_cell": float(fields["cell"].item()),
            "gradient_norm": gradient_norm,
            "parameter_norm": adapter_parameter_norm(adapter),
            "step_seconds": time.perf_counter() - step_started,
        }
        rows.append(row)
        if step == 1 or step % args.validation_every == 0 or step == args.steps:
            validation = validate(pl_module, val_loader, device)
            rows.append({
                "architecture": architecture, "split": "validation", "step": step,
                **validation,
            })
            print(json.dumps({
                "architecture": architecture, "step": step,
                "train_total": row["loss_total"], "validation": validation,
                "gradient_norm": gradient_norm,
            }), flush=True)

    elapsed = time.perf_counter() - started
    frozen_after = digest_frozen(pl_module)
    if frozen_before != frozen_after:
        raise RuntimeError("a frozen checkpoint parameter changed")
    changed = [
        name for name, value in adapter.state_dict().items()
        if not torch.equal(value.detach().cpu(), initial_adapter[name])
    ]
    output = ROOT / "checkpoints" / architecture
    output.mkdir(parents=True, exist_ok=False)
    curve = pd.DataFrame(rows)
    curve.to_csv(output / "training_curve.csv", index=False)
    checkpoint = {
        "architecture": architecture,
        "adapter_state_dict": {
            key: value.detach().cpu() for key, value in adapter.state_dict().items()
        },
        "base_model_root": str(MODEL_ROOT.resolve()),
        "block_index": 1,
        "training": {
            "steps": args.steps, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "seed": args.seed,
            "optimizer": "AdamW", "weight_decay": 1e-4,
            "objective": "original MatterGen mixed-field diffusion loss",
        },
    }
    torch.save(checkpoint, output / "adapter.pt")
    train_rows = curve[curve["split"] == "train"]
    validation_rows = curve[curve["split"] == "validation"]
    summary = {
        "architecture": architecture, "total_params": total_params,
        "trainable_params": trainable_params, "trainable_parameter_names": trainable_names,
        "zero_init_check": zero_check, "frozen_digest_before": frozen_before,
        "frozen_digest_after": frozen_after, "frozen_parameters_unchanged": frozen_before == frozen_after,
        "first_gradient_check": first_gradient_check,
        "changed_adapter_state_entries": changed,
        "changed_adapter_state_entry_count": len(changed),
        "adapter_state_entry_count": len(initial_adapter),
        "steps": args.steps, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "elapsed_seconds": elapsed,
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "first_50_train_loss_mean": float(train_rows.head(50)["loss_total"].mean()),
        "last_50_train_loss_mean": float(train_rows.tail(50)["loss_total"].mean()),
        "last_50_atomic_loss_mean": float(train_rows.tail(50)["loss_atomic_numbers"].mean()),
        "last_50_pos_loss_mean": float(train_rows.tail(50)["loss_pos"].mean()),
        "last_50_cell_loss_mean": float(train_rows.tail(50)["loss_cell"].mean()),
        "mean_step_seconds": float(train_rows["step_seconds"].mean()),
        "final_parameter_norm": float(train_rows.iloc[-1]["parameter_norm"]),
        "final_validation": {
            key: (int(value) if key == "step" else float(value))
            for key, value in validation_rows.iloc[-1].dropna().to_dict().items()
            if key not in {"architecture", "split"}
        },
        "all_losses_finite": bool(np.isfinite(train_rows["loss_total"]).all()),
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def launch(args: argparse.Namespace) -> None:
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    running = []
    for gpu, architecture in enumerate(ARCHITECTURES):
        output = ROOT / "checkpoints" / architecture
        if output.exists():
            raise FileExistsError(output)
        log_path = logs / f"training_{architecture}.log"
        stream = log_path.open("x", encoding="utf-8")
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--architecture", architecture, "--steps", str(args.steps),
            "--batch-size", str(args.batch_size), "--learning-rate", str(args.learning_rate),
            "--validation-every", str(args.validation_every), "--seed", str(args.seed),
        ]
        environment = dict(os.environ)
        environment.update(
            CUDA_VISIBLE_DEVICES=str(gpu), TMPDIR=str(PROJECT_ROOT / ".tmp"),
            XDG_CACHE_HOME=str(PROJECT_ROOT / ".cache"),
            HF_HOME=str(PROJECT_ROOT / ".cache/huggingface"),
            TORCH_HOME=str(PROJECT_ROOT / ".cache/torch"), PYTHONUNBUFFERED="1",
        )
        process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, env=environment,
            stdout=stream, stderr=subprocess.STDOUT,
        )
        running.append((architecture, process, stream, log_path))
        print(json.dumps({"event": "training_started", "architecture": architecture, "gpu": gpu, "pid": process.pid}), flush=True)
    failures = []
    for architecture, process, stream, log_path in running:
        return_code = process.wait()
        stream.close()
        print(json.dumps({"event": "training_complete", "architecture": architecture, "return_code": return_code}), flush=True)
        if return_code:
            failures.append((architecture, return_code, str(log_path)))
    if failures:
        raise RuntimeError(f"adapter training failures: {failures}")
    rows = []
    for architecture in ARCHITECTURES:
        output = ROOT / "checkpoints" / architecture
        summary = json.loads((output / "training_summary.json").read_text())
        final_val = summary["final_validation"]
        rows.append({
            "architecture": architecture, "trainable_params": summary["trainable_params"],
            "steps": summary["steps"], "batch_size": summary["batch_size"],
            "learning_rate": summary["learning_rate"],
            "first_50_train_loss_mean": summary["first_50_train_loss_mean"],
            "last_50_train_loss_mean": summary["last_50_train_loss_mean"],
            "last_50_atomic_loss_mean": summary["last_50_atomic_loss_mean"],
            "last_50_pos_loss_mean": summary["last_50_pos_loss_mean"],
            "last_50_cell_loss_mean": summary["last_50_cell_loss_mean"],
            "final_validation_loss": final_val["loss_total"],
            "final_validation_atomic_loss": final_val["loss_atomic_numbers"],
            "final_validation_pos_loss": final_val["loss_pos"],
            "final_validation_cell_loss": final_val["loss_cell"],
            "all_losses_finite": summary["all_losses_finite"],
            "zero_init_passed": summary["zero_init_check"]["passed"],
            "frozen_parameters_unchanged": summary["frozen_parameters_unchanged"],
            "changed_adapter_state_entries": summary["changed_adapter_state_entry_count"],
            "adapter_state_entries": summary["adapter_state_entry_count"],
            "mean_step_seconds": summary["mean_step_seconds"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "peak_cuda_memory_mb": summary["peak_cuda_memory_mb"],
            "final_parameter_norm": summary["final_parameter_norm"],
            "checkpoint_sha256": sha256(output / "adapter.pt"),
        })
    with (ROOT / "training_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=ARCHITECTURES)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=TRAINING_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.architecture:
        train_one(args.architecture, args)
    else:
        launch(args)


if __name__ == "__main__":
    main()
