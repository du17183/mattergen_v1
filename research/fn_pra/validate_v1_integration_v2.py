"""Deterministic CPU identity check plus GPU training-gradient integration check."""

from __future__ import annotations

import json

import torch
from hydra.utils import instantiate

from mattergen.common.data.collate import collate
from mattergen.common.data.dataset_transform import filter_sparse_properties
from mattergen.common.data.transform import set_chemical_system_string, symmetrize_lattice
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.fn_pra.data import RepaCrystalDataset
from research.fn_pra.phase1_common import REPORTS, atomic_json, now, set_stage
from research.fn_pra.validate_v1_integration import (
    CHECKPOINT_ROOT,
    NEW_PARAMETER_PARTS,
    TEACHER_CACHE,
    TRAIN_CACHE,
    frozen_checksum,
    is_new_parameter,
    output_comparison,
    repa_config,
)


def main() -> None:
    set_stage(
        "fn_pra_implementation",
        "running",
        "Running deterministic CPU identity and GPU gradient/freeze validation.",
    )
    torch.set_num_threads(1)
    info = MatterGenCheckpointInfo(str(CHECKPOINT_ROOT))
    config = info.config
    checkpoint = torch.load(info.checkpoint_path, map_location="cpu")
    base = instantiate(config.lightning_module)
    base.load_state_dict(checkpoint["state_dict"], strict=True)
    repa = instantiate(repa_config(config))
    incompatible = repa.load_state_dict(checkpoint["state_dict"], strict=False)
    expected_missing = {
        "diffusion_module.model.repa_adapter.norm.weight",
        "diffusion_module.model.repa_adapter.norm.bias",
        "diffusion_module.model.repa_adapter.down.weight",
        "diffusion_module.model.repa_adapter.up.weight",
        "diffusion_module.model.student_projection.weight",
        "diffusion_module.model.teacher_projection.weight",
    }
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint incompatibility: {incompatible}")
    dataset = RepaCrystalDataset.from_cache_path(
        cache_path=str(TRAIN_CACHE),
        teacher_cache_path=str(TEACHER_CACHE),
        properties=["dft_mag_density"],
        transforms=[symmetrize_lattice, set_chemical_system_string],
        dataset_transforms=[filter_sparse_properties],
    )
    batch_cpu = collate([dataset[index] for index in (0, 1)])
    timestep_cpu = torch.tensor([0.25, 0.75])
    base.eval()
    repa.eval()
    projection_calls = {"student": 0, "teacher": 0}

    def student_hook(*_args):
        projection_calls["student"] += 1

    def teacher_hook(*_args):
        projection_calls["teacher"] += 1

    student_handle = repa.diffusion_module.model.student_projection.register_forward_hook(student_hook)
    teacher_handle = repa.diffusion_module.model.teacher_projection.register_forward_hook(teacher_hook)
    with torch.no_grad():
        base_output = base.diffusion_module.model(batch_cpu, timestep_cpu)
        repa.diffusion_module.model.repa_enabled = False
        disabled_output = repa.diffusion_module.model(batch_cpu, timestep_cpu)
        repa.diffusion_module.model.repa_enabled = True
        zero_output = repa.diffusion_module.model(batch_cpu, timestep_cpu)
    student_handle.remove()
    teacher_handle.remove()
    disabled_comparison = output_comparison(base_output, disabled_output)
    zero_comparison = output_comparison(base_output, zero_output)
    if not all(item["bitwise"] for item in disabled_comparison.values()):
        raise RuntimeError(f"Disabled V1 is not bitwise identical on CPU: {disabled_comparison}")
    if not all(item["bitwise"] for item in zero_comparison.values()):
        raise RuntimeError(f"Zero-init V1 is not bitwise identical on CPU: {zero_comparison}")
    if projection_calls != {"student": 0, "teacher": 0}:
        raise RuntimeError(f"Projection heads executed during inference: {projection_calls}")
    del base, base_output, disabled_output, zero_output

    device = torch.device("cuda:0")
    batch = batch_cpu.to(device)
    repa = repa.to(device)
    for parameter in repa.parameters():
        parameter.requires_grad_(False)
    trainable_names = []
    for name, parameter in repa.named_parameters():
        if is_new_parameter(name):
            parameter.requires_grad_(True)
            trainable_names.append(name)
    trainable = [parameter for parameter in repa.parameters() if parameter.requires_grad]
    total_parameters = sum(parameter.numel() for parameter in repa.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in trainable)
    inference_parameters = sum(
        parameter.numel()
        for name, parameter in repa.named_parameters()
        if "repa_adapter" in name
    )
    before_checksum = frozen_checksum(repa)
    repa.train()
    torch.manual_seed(20260725)
    loss, metrics = repa.diffusion_module.calc_loss(batch)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite integration loss: {loss}")
    loss.backward()
    gradient_report = {}
    for name, parameter in repa.named_parameters():
        if parameter.requires_grad:
            gradient_report[name] = {
                "present": parameter.grad is not None,
                "finite": bool(parameter.grad is not None and torch.isfinite(parameter.grad).all()),
                "norm": float(parameter.grad.norm().item()) if parameter.grad is not None else 0.0,
            }
        elif parameter.grad is not None:
            raise RuntimeError(f"Frozen parameter received a gradient: {name}")
    for required in (
        "diffusion_module.model.repa_adapter.up.weight",
        "diffusion_module.model.student_projection.weight",
        "diffusion_module.model.teacher_projection.weight",
    ):
        if gradient_report[required]["norm"] <= 0:
            raise RuntimeError(f"Required trainable parameter has zero gradient: {required}")
    optimizer = torch.optim.Adam(trainable, lr=1e-4)
    optimizer.step()
    after_checksum = frozen_checksum(repa)
    if before_checksum != after_checksum:
        raise RuntimeError("Frozen backbone checksum changed after optimizer step")
    report = {
        "schema_version": 2,
        "created_at": now(),
        "checkpoint": info.checkpoint_path,
        "checkpoint_missing_keys": sorted(incompatible.missing_keys),
        "checkpoint_unexpected_keys": list(incompatible.unexpected_keys),
        "identity_device": "cpu_single_thread",
        "feature_disabled_comparison": disabled_comparison,
        "zero_init_comparison": zero_comparison,
        "inference_projection_calls": projection_calls,
        "loss": float(loss.detach().item()),
        "metrics": {key: float(value.detach().item()) for key, value in metrics.items()},
        "gradient_report": gradient_report,
        "frozen_checksum_before": before_checksum,
        "frozen_checksum_after": after_checksum,
        "frozen_checksum_unchanged": before_checksum == after_checksum,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_ratio": trainable_parameters / total_parameters,
        "inference_parameters": inference_parameters,
        "new_parameter_parts": list(NEW_PARAMETER_PARTS),
        "trainable_names": trainable_names,
        "passed": True,
    }
    atomic_json(REPORTS / "v1_implementation_validation.json", report)
    set_stage(
        "fn_pra_implementation",
        "success",
        "Static V1 passed CPU bitwise identity, inference isolation, GPU gradients and frozen checksum.",
        {
            "trainable_parameters": trainable_parameters,
            "inference_parameters": inference_parameters,
            "trainable_ratio": trainable_parameters / total_parameters,
        },
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
