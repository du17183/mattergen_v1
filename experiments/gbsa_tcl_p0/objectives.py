"""Minimal GBSA-TCL objectives; deliberately contains no clean-cell TCL path."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from mattergen.diffusion.training.field_loss import aggregate_per_sample


def warmup_fraction(step: int, warmup_steps: int = 100) -> float:
    if warmup_steps <= 1:
        return 1.0
    return min(max((step - 1) / (warmup_steps - 1), 0.0), 1.0)


def reverse_progress(t: torch.Tensor, diffusion_end_time: float) -> torch.Tensor:
    """Convert forward diffusion time to reverse-sampling progress."""
    normalized_t = t / float(diffusion_end_time)
    return (1.0 - normalized_t).clamp(0.0, 1.0)


def stage_weights(
    t: torch.Tensor,
    diffusion_end_time: float,
    *,
    early: float,
    middle: float,
    late: float,
) -> torch.Tensor:
    progress = reverse_progress(t, diffusion_end_time)
    return torch.where(
        progress < 0.3,
        torch.full_like(progress, early),
        torch.where(
            progress < 0.7,
            torch.full_like(progress, middle),
            torch.full_like(progress, late),
        ),
    )


def atomic_tcl_per_structure(
    output_low, output_high, atom_batch: torch.Tensor, batch_size: int
) -> torch.Tensor:
    low_probability = F.softmax(output_low["atomic_numbers"], dim=-1).detach()
    rows = F.kl_div(
        F.log_softmax(output_high["atomic_numbers"], dim=-1),
        low_probability,
        reduction="none",
    ).sum(dim=-1, keepdim=True)
    return aggregate_per_sample(
        rows, atom_batch, reduce="mean", batch_size=batch_size
    )


def _estimated_clean_position(corruption, noisy, output, t, batch_idx, clean):
    _, std = corruption.marginal_prob(
        x=clean["pos"], t=t, batch_idx=batch_idx, batch=clean
    )
    estimate = corruption.wrap(noisy["pos"] + std * output["pos"])
    return estimate, std


def normalized_position_tcl_per_structure(
    *, corruption, clean, low, high, output_low, output_high, t_low, t_high,
    floor: float,
) -> torch.Tensor:
    """Stable periodic position TCL normalized by the two-view noise scale."""
    batch_size = clean.get_batch_size()
    position_batch = clean.get_batch_idx("pos")
    position_corruption = corruption.sdes["pos"]
    estimate_low, std_low = _estimated_clean_position(
        position_corruption, low, output_low, t_low, position_batch, clean
    )
    estimate_high, std_high = _estimated_clean_position(
        position_corruption, high, output_high, t_high, position_batch, clean
    )
    estimate_low = estimate_low.detach()
    difference = torch.remainder(estimate_high - estimate_low + 0.5, 1.0) - 0.5
    numerator = aggregate_per_sample(
        difference.square(), position_batch, reduce="mean", batch_size=batch_size
    )
    paired_noise = 0.5 * (std_low.detach().square() + std_high.detach().square())
    denominator = aggregate_per_sample(
        paired_noise, position_batch, reduce="mean", batch_size=batch_size
    )
    return numerator / (denominator + float(floor))


def score_anchor_per_structure(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    batch_idx: torch.Tensor | None,
    batch_size: int,
    floor: float,
) -> torch.Tensor:
    """Relative score-field MSE with a fixed floor, reduced per structure."""
    teacher = teacher.detach()
    numerator = aggregate_per_sample(
        (student - teacher).square(), batch_idx,
        reduce="mean", batch_size=batch_size,
    )
    denominator = aggregate_per_sample(
        teacher.square(), batch_idx,
        reduce="mean", batch_size=batch_size,
    )
    return numerator / (denominator + float(floor))


def gbsa_objectives(
    *, diffusion, clean, views, student_low, student_high,
    teacher_low, teacher_high, step: int,
    position_tcl_floor: float = 0.01,
    position_anchor_floor: float = 0.01,
    cell_anchor_floor: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Return raw and configured-weight auxiliary objectives.

    There is intentionally no old clean-cell consistency estimate or 1/alpha
    operation in this function.
    """
    batch_size = clean.get_batch_size()
    atom_batch = clean.get_batch_idx("atomic_numbers")
    position_batch = clean.get_batch_idx("pos")
    warmup = warmup_fraction(step, warmup_steps=100)

    atomic_rows = atomic_tcl_per_structure(
        student_low, student_high, atom_batch, batch_size
    )
    position_rows = normalized_position_tcl_per_structure(
        corruption=diffusion.corruption,
        clean=clean,
        low=views.low,
        high=views.high,
        output_low=student_low,
        output_high=student_high,
        t_low=views.t_low,
        t_high=views.t_high,
        floor=position_tcl_floor,
    )

    pos_anchor_low = score_anchor_per_structure(
        student_low["pos"], teacher_low["pos"],
        batch_idx=position_batch, batch_size=batch_size, floor=position_anchor_floor,
    )
    pos_anchor_high = score_anchor_per_structure(
        student_high["pos"], teacher_high["pos"],
        batch_idx=position_batch, batch_size=batch_size, floor=position_anchor_floor,
    )
    cell_anchor_low = score_anchor_per_structure(
        student_low["cell"], teacher_low["cell"],
        batch_idx=None, batch_size=batch_size, floor=cell_anchor_floor,
    )
    cell_anchor_high = score_anchor_per_structure(
        student_high["cell"], teacher_high["cell"],
        batch_idx=None, batch_size=batch_size, floor=cell_anchor_floor,
    )

    end_time = float(diffusion.corruption.T)
    pos_weight_low = stage_weights(
        views.t_low, end_time, early=0.01, middle=0.05, late=0.10
    )
    pos_weight_high = stage_weights(
        views.t_high, end_time, early=0.01, middle=0.05, late=0.10
    )
    cell_weight_low = stage_weights(
        views.t_low, end_time, early=0.01, middle=0.025, late=0.05
    )
    cell_weight_high = stage_weights(
        views.t_high, end_time, early=0.01, middle=0.025, late=0.05
    )

    raw = {
        "atomic_tcl": atomic_rows.mean(),
        "position_tcl": position_rows.mean(),
        "position_anchor": 0.5 * (pos_anchor_low.mean() + pos_anchor_high.mean()),
        "cell_anchor": 0.5 * (cell_anchor_low.mean() + cell_anchor_high.mean()),
    }
    configured = {
        "atomic_tcl": warmup * (1.0 / 30.0) * raw["atomic_tcl"],
        "position_tcl": warmup * (1.0 / 30.0) * raw["position_tcl"],
        "position_anchor": warmup * 0.5 * (
            (pos_weight_low * pos_anchor_low).mean()
            + (pos_weight_high * pos_anchor_high).mean()
        ),
        "cell_anchor": warmup * 0.5 * (
            (cell_weight_low * cell_anchor_low).mean()
            + (cell_weight_high * cell_anchor_high).mean()
        ),
    }
    return {
        **{f"raw_{name}": value for name, value in raw.items()},
        **{f"weighted_{name}": value for name, value in configured.items()},
        "warmup_fraction": student_low["cell"].new_tensor(warmup),
    }
