"""Train only the motivated 83-parameter timestep radial gate."""
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
import torch
from torch.utils.data import DataLoader, Dataset

from mattergen.adaptive_neighborhood import TimestepRadialGate
from mattergen.common.data.collate import collate
from mattergen.common.data.dataset import CrystalDatasetBuilder
from mattergen.common.data.transform import symmetrize_lattice
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.common.utils.eval_utils import load_model_diffusion


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/adaptive_neighborhood_p0"
MODEL_ROOT = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/"
    "hf_mattergen/checkpoints/dft_mag_density"
)
DATA_ROOT = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/"
    "global_transformer_adapter_p0/data/cache"
)
FIELDS = ("atomic_numbers", "pos", "cell")
GATE_TIMES = (0.8, 0.5, 0.25, 0.1, 0.05, 0.02)


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


def differences(left, right) -> dict[str, float]:
    return {
        field: float((left[field].float() - right[field].float()).abs().max().item())
        for field in FIELDS
    }


def frozen_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        if ".adaptive_radial_gate." in name:
            continue
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


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
def validate(pl_module, loader, device: torch.device, seed: int) -> dict[str, float]:
    pl_module.eval()
    seed_everything(seed)
    totals = []
    fields = {field: [] for field in FIELDS}
    for batch in loader:
        total, values = calculate_loss(pl_module, batch.to(device))
        totals.append(float(total.item()))
        for field in FIELDS:
            fields[field].append(float(values[field].item()))
    return {
        "loss_total": float(np.mean(totals)),
        **{f"loss_{field}": float(np.mean(values)) for field, values in fields.items()},
    }


def gate_grid(gate: TimestepRadialGate, device: torch.device) -> list[dict[str, float]]:
    values = gate.gate_values(torch.tensor(GATE_TIMES, device=device)).detach().cpu()
    return [
        {
            "timestep": float(timestep),
            "g_near": float(row[0]),
            "g_mid": float(row[1]),
            "g_far": float(row[2]),
        }
        for timestep, row in zip(GATE_TIMES, values)
    ]


@torch.no_grad()
def identity_check(pl_module, batch, seed: int) -> dict:
    diffusion = pl_module.diffusion_module
    denoiser = diffusion.model
    seed_everything(seed)
    clean = diffusion.pre_corruption_fn(batch)
    noisy, t = diffusion._corrupt_batch(clean)
    original = denoiser(noisy, t)
    denoiser.gemnet.adaptive_radial_gate = TimestepRadialGate().to(batch["cell"].device)
    gated = denoiser(noisy, t)
    delta = differences(original, gated)
    parameter_count = sum(
        parameter.numel() for parameter in denoiser.gemnet.adaptive_radial_gate.parameters()
    )
    maximum = max(delta.values())
    return {
        "same_batch_same_noise": True,
        "field_max_abs_difference": delta,
        "maximum_output_difference": maximum,
        "tolerance": 1.0e-5,
        "gate_parameter_count": parameter_count,
        "passed": maximum <= 1.0e-5 and parameter_count < 5000,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train(args: argparse.Namespace) -> None:
    diagnostic = json.loads((ROOT / "diagnostic_summary.json").read_text())
    if not diagnostic["ADAPTIVE_NEIGHBORHOOD_MOTIVATED"]:
        raise RuntimeError("diagnostic did not authorize gate implementation/training")
    if (ROOT / "checkpoint/radial_gate.pt").exists():
        raise FileExistsError("refusing to overwrite radial gate checkpoint")
    if not torch.cuda.is_available():
        raise RuntimeError("real radial-gate training requires CUDA")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("high")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed_everything(args.seed)
    device = torch.device("cuda")
    train_data = CleanDataset(DATA_ROOT / "train")
    val_data = CleanDataset(DATA_ROOT / "val")
    if (len(train_data), len(val_data)) != (1024, 128):
        raise RuntimeError("expected fixed 1024 train / 128 validation split")
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    pl_module = load_model_diffusion(
        MatterGenCheckpointInfo(model_path=str(MODEL_ROOT))
    ).to(device)
    first_val = next(iter(val_loader)).to(device)
    pl_module.eval()
    identity = identity_check(pl_module, first_val, args.validation_seed)
    print(json.dumps({"identity_check": identity}, indent=2), flush=True)
    if not identity["passed"]:
        raise RuntimeError("strict identity check failed; gate training is prohibited")
    gate = pl_module.diffusion_module.model.gemnet.adaptive_radial_gate
    for parameter in pl_module.parameters():
        parameter.requires_grad_(False)
    for parameter in gate.parameters():
        parameter.requires_grad_(True)
    trainable_names = [
        name for name, parameter in pl_module.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_names or not all(".adaptive_radial_gate." in name for name in trainable_names):
        raise RuntimeError("non-gate parameter selected for training")
    trainable_params = sum(parameter.numel() for parameter in gate.parameters())
    total_params = sum(parameter.numel() for parameter in pl_module.parameters())
    if trainable_params >= 5000:
        raise RuntimeError("radial gate exceeds the preregistered parameter limit")
    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    frozen_before = frozen_digest(pl_module)
    initial_gate = {name: value.detach().cpu().clone() for name, value in gate.named_parameters()}
    trace = []
    gates = []
    initial_validation = validate(pl_module, val_loader, device, args.validation_seed)
    trace.append({"split": "validation", "step": 0, **initial_validation})
    for row in gate_grid(gate, device):
        gates.append({"step": 0, **row})
    best_validation = initial_validation["loss_total"]
    iterator = iter(train_loader)
    seed_everything(args.seed + 100)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    early_stopped = False
    actual_steps = 0
    first_gradient_check = None
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        pl_module.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        total, fields = calculate_loss(pl_module, batch.to(device))
        if not bool(torch.isfinite(total)):
            raise RuntimeError(f"non-finite loss at step {step}")
        total.backward()
        frozen_gradients = [
            name for name, parameter in pl_module.named_parameters()
            if not parameter.requires_grad and parameter.grad is not None
        ]
        if step == 1:
            first_gradient_check = {
                "gate_with_gradient": sorted(
                    name for name, parameter in pl_module.named_parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ),
                "frozen_with_gradient": frozen_gradients,
            }
        if frozen_gradients:
            raise RuntimeError(f"frozen parameter received gradient: {frozen_gradients[:3]}")
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(gate.parameters(), args.gradient_clip).item())
        optimizer.step()
        torch.cuda.synchronize()
        actual_steps = step
        trace.append({
            "split": "train",
            "step": step,
            "loss_total": float(total.item()),
            "loss_atomic_numbers": float(fields["atomic_numbers"].item()),
            "loss_pos": float(fields["pos"].item()),
            "loss_cell": float(fields["cell"].item()),
            "gradient_norm": gradient_norm,
            "step_seconds": time.perf_counter() - step_started,
        })
        if step == 1 or step % args.validation_every == 0 or step == args.steps:
            validation = validate(pl_module, val_loader, device, args.validation_seed)
            best_validation = min(best_validation, validation["loss_total"])
            trace.append({"split": "validation", "step": step, **validation})
            grid = gate_grid(gate, device)
            for row in grid:
                gates.append({"step": step, **row})
            identity_deviation = max(
                abs(row[key] - 1.0)
                for row in grid for key in ("g_near", "g_mid", "g_far")
            )
            print(json.dumps({
                "step": step,
                "train_loss": float(total.item()),
                "validation": validation,
                "maximum_gate_identity_deviation": identity_deviation,
                "gradient_norm": gradient_norm,
            }), flush=True)
            if (
                step >= args.early_stop_after
                and best_validation >= initial_validation["loss_total"]
                and identity_deviation <= args.early_stop_gate_identity_tolerance
            ):
                early_stopped = True
                break

    elapsed = time.perf_counter() - started
    frozen_after = frozen_digest(pl_module)
    if frozen_before != frozen_after:
        raise RuntimeError("a frozen original MatterGen parameter changed")
    changed_gate = [
        name for name, value in gate.named_parameters()
        if not torch.equal(value.detach().cpu(), initial_gate[name])
    ]
    checkpoint_dir = ROOT / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = checkpoint_dir / "radial_gate.pt"
    torch.save({
        "method": "Diffusion-Stage Adaptive Neighborhood GemNet-T",
        "base_model_root": str(MODEL_ROOT),
        "gate_state_dict": {key: value.detach().cpu() for key, value in gate.state_dict().items()},
        "training": {
            "steps": actual_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
    }, checkpoint_path)
    write_csv(ROOT / "training_trace.csv", trace)
    write_csv(ROOT / "gate_trace.csv", gates)
    training_rows = [row for row in trace if row["split"] == "train"]
    validation_rows = [row for row in trace if row["split"] == "validation"]
    summary = {
        "identity_check": identity,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_params / total_params,
        "trainable_parameter_names": trainable_names,
        "changed_gate_state_entries": changed_gate,
        "frozen_digest_before": frozen_before,
        "frozen_digest_after": frozen_after,
        "frozen_parameters_unchanged": frozen_before == frozen_after,
        "first_gradient_check": first_gradient_check,
        "steps": actual_steps,
        "early_stopped": early_stopped,
        "elapsed_seconds": elapsed,
        "mean_step_seconds": float(np.mean([row["step_seconds"] for row in training_rows])),
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "initial_validation": initial_validation,
        "final_validation": validation_rows[-1],
        "best_validation_loss": best_validation,
        "final_gate_grid": gate_grid(gate, device),
        "checkpoint_sha256": sha256(checkpoint_path),
        "adaptive_cfg_used": False,
        "dft_verified": False,
    }
    (ROOT / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--early-stop-after", type=int, default=500)
    parser.add_argument("--early-stop-gate-identity-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--validation-seed", type=int, default=20260924)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
