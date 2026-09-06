"""Shared, fair two-view corruption and original MatterGen field losses."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from mattergen.common.diffusion.corruption import make_noise_symmetric_preserve_variance
from mattergen.diffusion.corruption.multi_corruption import apply


@dataclass
class TwoViewBatch:
    low: object
    high: object
    t_low: torch.Tensor
    t_high: torch.Tensor


def sample_timestep_pair(diffusion, batch, min_normalized_gap: float = 0.2):
    """Sample two original-distribution timesteps conditional on a minimum gap."""
    batch_size = batch.get_batch_size()
    first = diffusion.sample_timesteps(batch)
    second = diffusion.sample_timesteps(batch)
    minimum_gap = float(diffusion.corruption.T) * min_normalized_gap
    invalid = (first - second).abs() < minimum_gap
    while bool(invalid.any()):
        second[invalid] = diffusion.timestep_sampler(
            batch_size=int(invalid.sum().item()), device=first.device
        )
        invalid = (first - second).abs() < minimum_gap
    return torch.minimum(first, second), torch.maximum(first, second)


def corrupt_two_views(diffusion, clean, t_low, t_high) -> TwoViewBatch:
    """Couple continuous noise while leaving D3PM draws independent."""
    batch_indices = diffusion.corruption._get_batch_indices(clean)
    low_fields = {}
    high_fields = {}
    for field, corruption in diffusion.corruption.sdes.items():
        x = clean[field]
        batch_idx = batch_indices[field]
        low_mean, low_std = corruption.marginal_prob(
            x=x, t=t_low, batch_idx=batch_idx, batch=clean
        )
        high_mean, high_std = corruption.marginal_prob(
            x=x, t=t_high, batch_idx=batch_idx, batch=clean
        )
        noise = torch.randn_like(x)
        if field == "cell":
            noise = make_noise_symmetric_preserve_variance(noise)
        low_value = low_mean + low_std * noise
        high_value = high_mean + high_std * noise
        if hasattr(corruption, "wrap"):
            low_value = corruption.wrap(low_value)
            high_value = corruption.wrap(high_value)
        low_fields[field] = low_value
        high_fields[field] = high_value
    for field, corruption in diffusion.corruption.discrete_corruptions.items():
        x = clean[field]
        batch_idx = batch_indices[field]
        low_fields[field] = corruption.sample_marginal(
            x=x, t=t_low, batch_idx=batch_idx, batch=clean
        )
        high_fields[field] = corruption.sample_marginal(
            x=x, t=t_high, batch_idx=batch_idx, batch=clean
        )
    return TwoViewBatch(
        low=clean.replace(**low_fields),
        high=clean.replace(**high_fields),
        t_low=t_low,
        t_high=t_high,
    )


def per_sample_field_losses(
    loss_fn, multi_corruption, clean, noisy, model_output, t
) -> dict[str, torch.Tensor]:
    """Return the existing MatterGen field losses before batch reduction."""
    batch_idx = {key: clean.get_batch_idx(key) for key in loss_fn.loss_fns}
    losses = apply(
        fns=loss_fn.loss_fns,
        corruption=multi_corruption.corruptions,
        x=clean,
        noisy_x=noisy,
        score_model_output=model_output,
        batch_idx=batch_idx,
        broadcast={"t": t, "batch_size": clean.get_batch_size(), "batch": clean},
        node_is_unmasked={key: None for key in loss_fn.loss_fns},
    )
    expected = (clean.get_batch_size(),)
    if any(value.shape != expected for value in losses.values()):
        raise RuntimeError("field loss did not preserve per-structure reduction")
    return losses


def fixed_weight_loss(loss_fn, per_field: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack(
        [loss_fn.loss_weights[field] * value for field, value in per_field.items()], dim=0
    ).sum(dim=0).mean()


def mean_field_metrics(
    low: dict[str, torch.Tensor], high: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    return {
        field: 0.5 * (low[field].mean() + high[field].mean())
        for field in low
    }
