"""Clean-state cross-timestep consistency for MatterGen fields."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from mattergen.diffusion.training.field_loss import aggregate_per_sample


def _estimated_clean_position(corruption, noisy, output, t, batch_idx, clean):
    _, std = corruption.marginal_prob(
        x=clean["pos"], t=t, batch_idx=batch_idx, batch=clean
    )
    return corruption.wrap(noisy["pos"] + std * output["pos"])


def _estimated_clean_cell(corruption, noisy, output, t, clean):
    _, std = corruption.marginal_prob(x=clean["cell"], t=t, batch=clean)
    alpha, _ = corruption.mean_coeff_and_std(clean["cell"], t=t, batch=clean)
    limit_mean = corruption.get_limit_mean(x=clean["cell"], batch=clean)
    limit_std = corruption.get_limit_var(x=clean["cell"], batch=clean).sqrt()
    predicted_mean = noisy["cell"] + std * output["cell"]
    estimated_x0 = (predicted_mean - (1.0 - alpha) * limit_mean) / alpha
    return (estimated_x0 - limit_mean) / limit_std


def consistency_losses(
    *, corruption, clean, low, high, output_low, output_high, t_low, t_high
) -> dict[str, torch.Tensor]:
    batch_size = clean.get_batch_size()
    atom_batch = clean.get_batch_idx("atomic_numbers")
    position_batch = clean.get_batch_idx("pos")

    low_probability = F.softmax(output_low["atomic_numbers"], dim=-1).detach()
    atomic_rows = F.kl_div(
        F.log_softmax(output_high["atomic_numbers"], dim=-1),
        low_probability,
        reduction="none",
    ).sum(dim=-1, keepdim=True)
    atomic = aggregate_per_sample(
        atomic_rows, atom_batch, reduce="mean", batch_size=batch_size
    ).mean()

    position_corruption = corruption.sdes["pos"]
    clean_position_low = _estimated_clean_position(
        position_corruption, low, output_low, t_low, position_batch, clean
    ).detach()
    clean_position_high = _estimated_clean_position(
        position_corruption, high, output_high, t_high, position_batch, clean
    )
    position_difference = torch.remainder(
        clean_position_high - clean_position_low + 0.5, 1.0
    ) - 0.5
    position = aggregate_per_sample(
        position_difference.square(), position_batch,
        reduce="mean", batch_size=batch_size,
    ).mean()

    cell_corruption = corruption.sdes["cell"]
    clean_cell_low = _estimated_clean_cell(
        cell_corruption, low, output_low, t_low, clean
    ).detach()
    clean_cell_high = _estimated_clean_cell(
        cell_corruption, high, output_high, t_high, clean
    )
    cell = aggregate_per_sample(
        (clean_cell_high - clean_cell_low).square(), None,
        reduce="mean", batch_size=batch_size,
    ).mean()

    return {
        "atomic_consistency": atomic,
        "position_consistency": position,
        "cell_consistency": cell,
        "consistency": (atomic + position + cell) / 3.0,
    }


def consistency_weight(step: int, maximum: float = 0.1, warmup_steps: int = 100) -> float:
    if warmup_steps <= 1:
        return maximum
    fraction = min(max((step - 1) / (warmup_steps - 1), 0.0), 1.0)
    return maximum * fraction
