from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.denoiser import GemNetTDenoiser


def _distributed_counts(local_count: int, device: torch.device) -> tuple[list[int], int]:
    if not dist.is_available() or not dist.is_initialized():
        return [local_count], 0
    rank = dist.get_rank()
    value = torch.tensor([local_count], device=device, dtype=torch.long)
    gathered = [torch.zeros_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, value)
    counts = [int(item.item()) for item in gathered]
    return counts, sum(counts[:rank])


def _gather_rows(
    student: torch.Tensor,
    teacher: torch.Tensor,
    elements: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Gather variable atom rows, retaining autograd only for student features."""
    if not dist.is_available() or not dist.is_initialized():
        return student, teacher, elements, 0
    counts, positive_offset = _distributed_counts(student.shape[0], student.device)
    max_count = max(counts)
    student_padded = F.pad(student, (0, 0, 0, max_count - student.shape[0]))
    teacher_padded = F.pad(teacher, (0, 0, 0, max_count - teacher.shape[0]))
    element_padded = F.pad(elements, (0, max_count - elements.shape[0]), value=-1)

    from torch.distributed.nn.functional import all_gather as differentiable_all_gather

    gathered_student = differentiable_all_gather(student_padded)
    gathered_teacher = [torch.empty_like(teacher_padded) for _ in counts]
    gathered_elements = [torch.empty_like(element_padded) for _ in counts]
    dist.all_gather(gathered_teacher, teacher_padded)
    dist.all_gather(gathered_elements, element_padded)
    all_student = torch.cat(
        [rows[:count] for rows, count in zip(gathered_student, counts, strict=True)]
    )
    all_teacher = torch.cat(
        [rows[:count] for rows, count in zip(gathered_teacher, counts, strict=True)]
    )
    all_elements = torch.cat(
        [rows[:count] for rows, count in zip(gathered_elements, counts, strict=True)]
    )
    return all_student, all_teacher, all_elements, positive_offset


def _masked_directional_ce(
    anchors: torch.Tensor,
    candidates: torch.Tensor,
    anchor_elements: torch.Tensor,
    candidate_elements: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    logits = anchors @ candidates.transpose(0, 1) / temperature
    allowed = anchor_elements[:, None].ne(candidate_elements[None, :])
    allowed.scatter_(1, targets[:, None], True)
    logits = logits.masked_fill(~allowed, -torch.inf)
    return F.cross_entropy(logits, targets)


def symmetric_element_aware_nce(
    student: torch.Tensor,
    teacher: torch.Tensor,
    elements: torch.Tensor,
    temperature: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric EA-NCE with same-element off-diagonal pairs removed.

    Positives are the student and teacher rows for the same structure and atom index.
    Every off-diagonal candidate with the same element as the anchor is excluded.
    """
    if student.ndim != 2 or teacher.ndim != 2 or student.shape != teacher.shape:
        raise ValueError(
            f"Student/teacher shapes must be equal 2-D rows: {student.shape}/{teacher.shape}"
        )
    if elements.ndim != 1 or elements.shape[0] != student.shape[0]:
        raise ValueError("Element IDs must contain exactly one entry per atom")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    student = F.normalize(student.float(), dim=-1)
    teacher = F.normalize(teacher.float(), dim=-1)
    all_student, all_teacher, all_elements, offset = _gather_rows(
        student, teacher, elements
    )
    targets = offset + torch.arange(student.shape[0], device=student.device)
    student_to_teacher = _masked_directional_ce(
        student,
        all_teacher,
        elements,
        all_elements,
        targets,
        temperature,
    )
    teacher_to_student = _masked_directional_ce(
        teacher,
        all_student,
        elements,
        all_elements,
        targets,
        temperature,
    )
    loss = 0.5 * (student_to_teacher + teacher_to_student)
    positive_cosine = (student * teacher).sum(dim=-1).mean()
    return loss, positive_cosine


class ResidualProjection(nn.Module):
    """Paper-style one-block residual MLP followed by a linear projection."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.residual = nn.Linear(input_dim, input_dim)
        self.output = nn.Linear(input_dim, output_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(value)
        hidden = normalized + F.silu(self.residual(normalized))
        return self.output(hidden)


class CrystalRepaDenoiser(GemNetTDenoiser):
    """Training-only CrystalREPA objective attached to an intermediate GemNet block."""

    def __init__(
        self,
        *args: Any,
        alignment_block: int = 2,
        teacher_feature_dim: int = 64,
        alignment_weight: float = 1.0,
        alignment_temperature: float = 0.1,
        alignment_enabled: bool = True,
        inference_only: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if alignment_block < 1 or alignment_block > len(self.gemnet.int_blocks):
            raise ValueError(
                f"alignment_block={alignment_block} outside "
                f"1..{len(self.gemnet.int_blocks)}"
            )
        self.alignment_block = alignment_block
        self.alignment_weight = alignment_weight
        self.alignment_temperature = alignment_temperature
        self.alignment_enabled = alignment_enabled
        self.inference_only = inference_only
        self.student_projection = (
            None
            if inference_only
            else ResidualProjection(self.hidden_dim, teacher_feature_dim)
        )
        self._capture_active = False
        self._captured_hidden: torch.Tensor | None = None
        self._repa_auxiliary: dict[str, torch.Tensor] = {}
        self._alignment_hook = self.gemnet.int_blocks[
            alignment_block - 1
        ].register_forward_hook(self._capture_block_output)

    def _capture_block_output(
        self, _module: nn.Module, _inputs: tuple[Any, ...], output: tuple[torch.Tensor, ...]
    ) -> None:
        if self._capture_active:
            self._captured_hidden = output[0]

    def consume_repa_auxiliary(self) -> dict[str, torch.Tensor]:
        auxiliary = self._repa_auxiliary
        self._repa_auxiliary = {}
        return auxiliary

    def forward(self, x: ChemGraph, t: torch.Tensor) -> ChemGraph:
        has_teacher = "teacher_features" in x and "teacher_atomic_numbers" in x
        self._capture_active = bool(
            self.alignment_enabled
            and not self.inference_only
            and self.student_projection is not None
            and has_teacher
        )
        self._captured_hidden = None
        self._repa_auxiliary = {}
        output = super().forward(x, t)
        self._capture_active = False
        if not (
            self.alignment_enabled
            and self.student_projection is not None
            and has_teacher
        ):
            return output
        if self._captured_hidden is None:
            raise RuntimeError(
                f"GemNet block {self.alignment_block} did not emit an atom representation"
            )
        teacher = x["teacher_features"].to(
            self._captured_hidden.device, dtype=self._captured_hidden.dtype
        )
        elements = x["teacher_atomic_numbers"].to(
            self._captured_hidden.device, dtype=torch.long
        )
        student = self.student_projection(self._captured_hidden)
        alignment, cosine = symmetric_element_aware_nce(
            student,
            teacher,
            elements,
            temperature=self.alignment_temperature,
        )
        self._repa_auxiliary = {
            "alignment_raw": alignment,
            "alignment_weighted": alignment * self.alignment_weight,
            "positive_cosine": cosine,
        }
        return output
