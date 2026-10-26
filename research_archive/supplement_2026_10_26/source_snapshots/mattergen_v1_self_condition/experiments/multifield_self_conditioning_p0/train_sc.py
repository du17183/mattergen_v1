"""Train the independent multi-field self-conditioning P0 model."""
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

from mattergen.common.data.collate import collate
from mattergen.common.data.dataset import CrystalDatasetBuilder
from mattergen.common.data.transform import symmetrize_lattice
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.common.utils.eval_utils import load_model_diffusion
from mattergen.self_conditioning import MultiFieldSelfConditioner, clean_prediction


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/multifield_self_conditioning_p0"
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


def field_differences(left, right) -> dict[str, float]:
    return {
        field: float((left[field].float() - right[field].float()).abs().max().item())
        for field in FIELDS
    }


def selected_parameter_names(module: torch.nn.Module) -> list[str]:
    prefixes = (
        "diffusion_module.model.self_conditioner.",
        "diffusion_module.model.gemnet.int_blocks.3.",
        "diffusion_module.model.gemnet.out_blocks.",
        "diffusion_module.model.gemnet.lattice_out_blocks.",
        "diffusion_module.model.fc_atom.",
    )
    return [
        name for name, _ in module.named_parameters()
        if name.startswith(prefixes)
    ]


def configure_trainable(module: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    selected = set(selected_parameter_names(module))
    sc_parameters: list[torch.nn.Parameter] = []
    backbone_parameters: list[torch.nn.Parameter] = []
    for name, parameter in module.named_parameters():
        if name not in selected:
            continue
        parameter.requires_grad_(True)
        if name.startswith("diffusion_module.model.self_conditioner."):
            sc_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)
    if not sc_parameters or not backbone_parameters:
        raise RuntimeError("self-conditioning/backbone trainable groups are incomplete")
    actual = {name for name, parameter in module.named_parameters() if parameter.requires_grad}
    if actual != selected:
        raise RuntimeError("unexpected trainable parameter selection")
    return sc_parameters, backbone_parameters


def frozen_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            continue
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def gates_at_grid(denoiser, device: torch.device) -> list[dict[str, float]]:
    times = torch.tensor(GATE_TIMES, device=device)
    embedding = denoiser.noise_level_encoding(times).to(device)
    gates = denoiser.self_conditioner.gate_values(embedding).detach().cpu()
    return [
        {
            "timestep": float(t),
            "g_atom": float(gate[0]),
            "g_pos": float(gate[1]),
            "g_cell": float(gate[2]),
        }
        for t, gate in zip(GATE_TIMES, gates)
    ]


@torch.no_grad()
def zero_init_check(pl_module, batch, seed: int) -> dict:
    diffusion = pl_module.diffusion_module
    denoiser = diffusion.model
    pl_module.eval()
    seed_everything(seed)
    clean = diffusion.pre_corruption_fn(batch)
    noisy, t = diffusion._corrupt_batch(clean)
    if hasattr(denoiser, "self_conditioner"):
        raise RuntimeError("zero-init check requires the strict official model first")
    original = denoiser(noisy, t)
    denoiser.self_conditioner = MultiFieldSelfConditioner().to(batch["cell"].device)
    preliminary = denoiser(noisy, t)
    previous = clean_prediction(
        diffusion=diffusion,
        noisy_batch=noisy,
        t=t,
        output=preliminary,
        output_is_score=False,
    )
    with_condition = denoiser(noisy, t, self_condition=previous)
    disabled_difference = field_differences(original, preliminary)
    enabled_difference = field_differences(original, with_condition)
    final_layer = denoiser.self_conditioner.gate_network[-1]
    gate_parameter_max = max(
        float(final_layer.weight.detach().abs().max().item()),
        float(final_layer.bias.detach().abs().max().item()),
    )
    maximum = max(enabled_difference.values())
    return {
        "same_batch_same_noise": True,
        "gate_final_layer_max_abs": gate_parameter_max,
        "sc_disabled_max_abs_difference": disabled_difference,
        "sc_enabled_zero_gate_max_abs_difference": enabled_difference,
        "maximum_output_difference": maximum,
        "tolerance": 1.0e-5,
        "passed": gate_parameter_max == 0.0 and maximum <= 1.0e-5,
    }


def make_previous_prediction(diffusion, noisy, t, preliminary, probability: float):
    batch_size = noisy.get_batch_size()
    mask = (torch.rand(batch_size, device=t.device) < probability).to(t.dtype)
    previous = clean_prediction(
        diffusion=diffusion,
        noisy_batch=noisy,
        t=t,
        output=preliminary,
        output_is_score=False,
        sample_mask=mask,
    )
    return previous, mask


def training_loss(pl_module, batch, probability: float):
    diffusion = pl_module.diffusion_module
    denoiser = diffusion.model
    clean = diffusion.pre_corruption_fn(batch)
    noisy, t = diffusion._corrupt_batch(clean)
    denoiser.clear_self_condition()
    with torch.no_grad():
        preliminary = denoiser(noisy, t)
        previous, mask = make_previous_prediction(
            diffusion, noisy, t, preliminary, probability
        )
    output = denoiser(noisy, t, self_condition=previous)
    total, fields = diffusion.loss_fn(
        multi_corruption=diffusion.corruption,
        batch=clean,
        noisy_batch=noisy,
        score_model_output=output,
        t=t,
    )
    denoiser.clear_self_condition()
    return total, fields, float(mask.mean().item())


@torch.no_grad()
def validate(pl_module, loader, device: torch.device, seed: int) -> dict[str, float]:
    pl_module.eval()
    diffusion = pl_module.diffusion_module
    denoiser = diffusion.model
    values = {
        f"{mode}_{field}": []
        for mode in ("sc_on", "sc_off")
        for field in ("total", *FIELDS)
    }
    seed_everything(seed)
    for batch in loader:
        batch = batch.to(device)
        clean = diffusion.pre_corruption_fn(batch)
        noisy, t = diffusion._corrupt_batch(clean)
        denoiser.clear_self_condition()
        preliminary = denoiser(noisy, t)
        previous = clean_prediction(
            diffusion=diffusion,
            noisy_batch=noisy,
            t=t,
            output=preliminary,
            output_is_score=False,
        )
        output_on = denoiser(noisy, t, self_condition=previous)
        denoiser.self_conditioner.enabled = False
        output_off = denoiser(noisy, t)
        denoiser.self_conditioner.enabled = True
        for mode, output in (("sc_on", output_on), ("sc_off", output_off)):
            total, fields = diffusion.loss_fn(
                multi_corruption=diffusion.corruption,
                batch=clean,
                noisy_batch=noisy,
                score_model_output=output,
                t=t,
            )
            values[f"{mode}_total"].append(float(total.item()))
            for field in FIELDS:
                values[f"{mode}_{field}"].append(float(fields[field].item()))
    denoiser.clear_self_condition()
    return {key: float(np.mean(value)) for key, value in values.items()}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
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
    if not torch.cuda.is_available():
        raise RuntimeError("real self-conditioning training requires CUDA")
    if (ROOT / "checkpoint/sc_p0.pt").exists():
        raise FileExistsError("refusing to overwrite checkpoint/sc_p0.pt")
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
    zero_check = zero_init_check(pl_module, first_val, args.validation_seed)
    print(json.dumps({"zero_init_check": zero_check}, indent=2), flush=True)
    if not zero_check["passed"]:
        raise RuntimeError("strict zero-init recovery failed; training is prohibited")

    sc_parameters, backbone_parameters = configure_trainable(pl_module)
    trainable_names = [
        name for name, parameter in pl_module.named_parameters()
        if parameter.requires_grad
    ]
    total_params = sum(parameter.numel() for parameter in pl_module.parameters())
    trainable_params = sum(parameter.numel() for parameter in pl_module.parameters() if parameter.requires_grad)
    trainable_ratio = trainable_params / total_params
    print(json.dumps({
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_ratio,
        "trainable_tensor_count": len(trainable_names),
        "trainable_parameter_names": trainable_names,
    }, indent=2), flush=True)

    optimizer = torch.optim.AdamW(
        [
            {"params": sc_parameters, "lr": args.sc_lr},
            {"params": backbone_parameters, "lr": args.backbone_lr},
        ],
        weight_decay=args.weight_decay,
    )
    initial_trainable = {
        name: parameter.detach().cpu().clone()
        for name, parameter in pl_module.named_parameters()
        if parameter.requires_grad
    }
    frozen_before = frozen_digest(pl_module)
    trace: list[dict] = []
    gate_trace: list[dict] = []
    initial_validation = validate(
        pl_module, val_loader, device, args.validation_seed
    )
    trace.append({"split": "validation", "step": 0, **initial_validation})
    for row in gates_at_grid(pl_module.diffusion_module.model, device):
        gate_trace.append({"step": 0, **row})
    print(json.dumps({"step": 0, "validation": initial_validation}), flush=True)

    iterator = iter(train_loader)
    seed_everything(args.seed + 100)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    best_validation = initial_validation["sc_on_total"]
    early_stopped = False
    first_gradient_check = None
    actual_steps = 0
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
        total, fields, sc_fraction = training_loss(
            pl_module, batch, args.sc_probability
        )
        if not bool(torch.isfinite(total)):
            raise RuntimeError(f"non-finite loss at step {step}")
        total.backward()
        trainable_gradients = {
            name: parameter.grad
            for name, parameter in pl_module.named_parameters()
            if parameter.requires_grad and parameter.grad is not None
        }
        frozen_gradients = [
            name for name, parameter in pl_module.named_parameters()
            if not parameter.requires_grad and parameter.grad is not None
        ]
        if step == 1:
            first_gradient_check = {
                "trainable_with_gradient": sorted(trainable_gradients),
                "nonzero_trainable_gradient": sorted(
                    name for name, gradient in trainable_gradients.items()
                    if float(gradient.detach().float().norm().item()) > 0
                ),
                "frozen_with_gradient": frozen_gradients,
            }
        if frozen_gradients:
            raise RuntimeError(f"frozen parameter received gradient: {frozen_gradients[:3]}")
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            [*sc_parameters, *backbone_parameters], args.gradient_clip
        ).item())
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
            "sc_sample_fraction": sc_fraction,
            "gradient_norm": gradient_norm,
            "step_seconds": time.perf_counter() - step_started,
        })
        if step == 1 or step % args.validation_every == 0 or step == args.steps:
            validation = validate(
                pl_module, val_loader, device, args.validation_seed
            )
            best_validation = min(best_validation, validation["sc_on_total"])
            trace.append({"split": "validation", "step": step, **validation})
            grid = gates_at_grid(pl_module.diffusion_module.model, device)
            for row in grid:
                gate_trace.append({"step": step, **row})
            gate_max = max(
                abs(row[key])
                for row in grid
                for key in ("g_atom", "g_pos", "g_cell")
            )
            print(json.dumps({
                "step": step,
                "train_loss": float(total.item()),
                "validation": validation,
                "gate_max_abs": gate_max,
                "gradient_norm": gradient_norm,
            }), flush=True)
            if (
                step >= args.early_stop_after
                and best_validation >= initial_validation["sc_on_total"]
                and gate_max <= args.early_stop_gate_tolerance
            ):
                early_stopped = True
                break

    elapsed = time.perf_counter() - started
    frozen_after = frozen_digest(pl_module)
    if frozen_before != frozen_after:
        raise RuntimeError("a frozen official parameter changed")
    changed_trainable = [
        name for name, parameter in pl_module.named_parameters()
        if parameter.requires_grad
        and not torch.equal(parameter.detach().cpu(), initial_trainable[name])
    ]
    checkpoint_dir = ROOT / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = checkpoint_dir / "sc_p0.pt"
    selected_state = {
        name: parameter.detach().cpu()
        for name, parameter in pl_module.named_parameters()
        if parameter.requires_grad
    }
    torch.save({
        "method": "Multi-field Self-Conditioned MatterGen",
        "base_model_root": str(MODEL_ROOT),
        "trainable_state_dict": selected_state,
        "training": {
            "steps": actual_steps,
            "batch_size": args.batch_size,
            "sc_probability": args.sc_probability,
            "sc_learning_rate": args.sc_lr,
            "block4_and_heads_learning_rate": args.backbone_lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
    }, checkpoint_path)
    write_csv(ROOT / "training_trace.csv", trace)
    write_csv(ROOT / "gate_trace.csv", gate_trace)
    training_rows = [row for row in trace if row["split"] == "train"]
    validation_rows = [row for row in trace if row["split"] == "validation"]
    summary = {
        "base_model_root": str(MODEL_ROOT),
        "base_checkpoint_sha256": "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e",
        "zero_init_check": zero_check,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_ratio,
        "trainable_tensor_count": len(trainable_names),
        "trainable_parameter_names": trainable_names,
        "changed_trainable_tensor_count": len(changed_trainable),
        "frozen_digest_before": frozen_before,
        "frozen_digest_after": frozen_after,
        "frozen_parameters_unchanged": frozen_before == frozen_after,
        "first_gradient_check": first_gradient_check,
        "steps": actual_steps,
        "early_stopped": early_stopped,
        "elapsed_seconds": elapsed,
        "mean_step_seconds": float(np.mean([row["step_seconds"] for row in training_rows])),
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "mean_sc_sample_fraction": float(np.mean([row["sc_sample_fraction"] for row in training_rows])),
        "initial_validation": initial_validation,
        "final_validation": validation_rows[-1],
        "best_sc_on_validation_loss": best_validation,
        "checkpoint_sha256": sha256(checkpoint_path),
        "adaptive_cfg_used": False,
        "dft_verified": False,
    }
    (ROOT / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sc-probability", type=float, default=0.5)
    parser.add_argument("--sc-lr", type=float, default=1.0e-3)
    parser.add_argument("--backbone-lr", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--early-stop-after", type=int, default=500)
    parser.add_argument("--early-stop-gate-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--validation-seed", type=int, default=20260922)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
