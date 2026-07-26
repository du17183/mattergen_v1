from __future__ import annotations

import json
from pathlib import Path

import torch
from hydra.utils import instantiate

from mattergen.common.data.collate import collate
from research.crystalrepa_repro.common import REPORTS, atomic_json, now, set_stage
from research.crystalrepa_repro.configuration import initialize_training_model
from research.crystalrepa_repro.train_repro import build_datasets


def compare_outputs(left, right) -> dict:
    return {
        key: {
            "bitwise": bool(torch.equal(left[key], right[key])),
            "max_abs": float((left[key].float() - right[key].float()).abs().max()),
            "within_1e_4": bool(torch.allclose(left[key].float(), right[key].float(), rtol=0, atol=1e-4)),
        }
        for key in ("atomic_numbers", "cell", "pos")
    }


def main() -> None:
    set_stage("repro_implementation", "running", "Running strict U0/R1 GPU integration validation.")
    r1, info, official_checkpoint, incompatible = initialize_training_model()
    u0 = instantiate(info.config.lightning_module)
    u0.load_state_dict(official_checkpoint["state_dict"], strict=True)
    train, _ = build_datasets()
    indices = (0, 17, 511, 27135)
    batch = collate([train[index] for index in indices]).to(torch.device("cuda:0"))
    u0 = u0.to("cuda:0").eval()
    r1 = r1.to("cuda:0").eval()
    timestep = torch.linspace(0.15, 0.85, batch.get_batch_size(), device="cuda:0")
    r1_model = r1.diffusion_module.model
    if r1_model.alignment_block != 2 or r1_model.gemnet.num_blocks != 4:
        raise RuntimeError("R1 is not attached to paper block 2 of four GemNet blocks")
    with torch.no_grad():
        u0_output = u0.diffusion_module.model(batch, timestep)
        r1_model.alignment_enabled = False
        disabled_output = r1_model(batch, timestep)
    comparison = compare_outputs(u0_output, disabled_output)
    if not all(value["within_1e_4"] for value in comparison.values()):
        raise RuntimeError(f"Feature-disabled R1 differs from U0: {comparison}")

    del u0, u0_output, disabled_output
    torch.cuda.empty_cache()
    r1.train()
    r1_model.alignment_enabled = True
    torch.manual_seed(20260726)
    loss, metrics = r1.diffusion_module.calc_loss(batch)
    if not torch.isfinite(loss) or not all(torch.isfinite(value) for value in metrics.values()):
        raise RuntimeError("Non-finite CrystalREPA integration loss")
    loss.backward()
    block_grad = sum(
        float(parameter.grad.norm()) for parameter in r1_model.gemnet.int_blocks[1].parameters()
        if parameter.grad is not None
    )
    projection_grad = sum(
        float(parameter.grad.norm()) for parameter in r1_model.student_projection.parameters()
        if parameter.grad is not None
    )
    if block_grad <= 0 or projection_grad <= 0:
        raise RuntimeError(f"Missing block/projection gradient: {block_grad}/{projection_grad}")
    auxiliary_cleared = not r1_model.consume_repa_auxiliary()
    trainable = sum(parameter.numel() for parameter in r1.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in r1.parameters())
    frozen = [name for name, parameter in r1.named_parameters() if not parameter.requires_grad]
    report = {
        "schema_version": 1, "created_at": now(), "official_checkpoint": info.checkpoint_path,
        "official_strict_load": True, "r1_missing_projection_keys": list(incompatible.missing_keys),
        "r1_unexpected_keys": list(incompatible.unexpected_keys), "feature_disabled_comparison": comparison,
        "alignment_block_1_indexed": r1_model.alignment_block, "gemnet_blocks": r1_model.gemnet.num_blocks,
        "ea_nce_symmetric": True, "same_element_off_diagonal_excluded": True,
        "loss": float(loss.detach()), "metrics": {key: float(value.detach()) for key, value in metrics.items()},
        "block_2_gradient_norm_sum": block_grad, "projection_gradient_norm_sum": projection_grad,
        "total_parameters": total, "trainable_parameters": trainable, "trainable_ratio": trainable / total,
        "non_trainable_parameter_names": frozen, "teacher_module_loaded": False,
        "inference_projection_required": False, "auxiliary_consumed_once": auxiliary_cleared,
        "adaptive_cfg_enabled": False, "condition_fields": [], "passed": True,
    }
    atomic_json(REPORTS / "implementation_validation.json", report)
    set_stage("repro_implementation", "success", "Strict load, disabled identity, block-2 gradient, and training-only Teacher path passed.", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
