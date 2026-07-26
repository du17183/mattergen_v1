from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from mattergen.adapter import GemNetTAdapter
from mattergen.common.data.chemgraph import ChemGraph
from mattergen.denoiser import get_chemgraph_from_denoiser_output
from mattergen.property_embeddings import (
    get_property_embeddings,
    get_use_unconditional_embedding,
)


class LowRankAtomAdapter(nn.Module):
    """Zero-initialized residual bottleneck after the final GemNet block."""

    def __init__(self, hidden_dim: int, rank: int = 16) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=False)
        self.activation = nn.SiLU()
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.up(self.activation(self.down(self.norm(hidden))))


def _gather_teacher_rows(
    teacher: torch.Tensor,
    elements: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Differentiably gather variable-length teacher rows and their element IDs."""
    if not dist.is_available() or not dist.is_initialized():
        return teacher, elements, 0
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_count = torch.tensor([teacher.shape[0]], device=teacher.device, dtype=torch.long)
    count_tensors = [torch.zeros_like(local_count) for _ in range(world_size)]
    dist.all_gather(count_tensors, local_count)
    counts = [int(item.item()) for item in count_tensors]
    max_count = max(counts)
    teacher_padded = F.pad(teacher, (0, 0, 0, max_count - teacher.shape[0]))
    elements_padded = F.pad(elements, (0, max_count - elements.shape[0]), value=-1)

    from torch.distributed.nn.functional import all_gather as differentiable_all_gather

    gathered_teacher = differentiable_all_gather(teacher_padded)
    gathered_elements = [torch.empty_like(elements_padded) for _ in range(world_size)]
    dist.all_gather(gathered_elements, elements_padded)
    teacher_rows = torch.cat(
        [value[:count] for value, count in zip(gathered_teacher, counts, strict=True)], dim=0
    )
    element_rows = torch.cat(
        [value[:count] for value, count in zip(gathered_elements, counts, strict=True)], dim=0
    )
    positive_offset = sum(counts[:rank])
    return teacher_rows, element_rows, positive_offset


def element_aware_nce(
    student: torch.Tensor,
    teacher: torch.Tensor,
    elements: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """EA-NCE with same-atom positives and same-element off-diagonal exclusion."""
    if student.ndim != 2 or teacher.ndim != 2 or student.shape != teacher.shape:
        raise ValueError(f"Student/teacher shapes must match 2-D rows: {student.shape}/{teacher.shape}")
    if elements.ndim != 1 or elements.shape[0] != student.shape[0]:
        raise ValueError("Element IDs must contain one entry per atom")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    student = F.normalize(student.float(), dim=-1)
    teacher = F.normalize(teacher.float(), dim=-1)
    global_teacher, global_elements, positive_offset = _gather_teacher_rows(teacher, elements)
    logits = student @ global_teacher.transpose(0, 1) / temperature
    targets = positive_offset + torch.arange(student.shape[0], device=student.device)
    allowed = elements[:, None].ne(global_elements[None, :])
    allowed.scatter_(1, targets[:, None], True)
    logits = logits.masked_fill(~allowed, -torch.inf)
    return F.cross_entropy(logits, targets)


class StaticRepaAdapter(GemNetTAdapter):
    """Static FN-PRA V1 adapter; the teacher and projection heads are training-only."""

    def __init__(
        self,
        *args: Any,
        adapter_rank: int = 16,
        teacher_feature_dim: int = 64,
        projection_dim: int = 128,
        alignment_weight: float = 0.1,
        alignment_temperature: float = 0.07,
        repa_enabled: bool = True,
        inference_only: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.repa_adapter = LowRankAtomAdapter(self.hidden_dim, rank=adapter_rank)
        self.repa_enabled = repa_enabled
        self.alignment_weight = alignment_weight
        self.alignment_temperature = alignment_temperature
        self.inference_only = inference_only
        if inference_only:
            self.student_projection = None
            self.teacher_projection = None
        else:
            self.student_projection = nn.Linear(self.hidden_dim, projection_dim, bias=False)
            self.teacher_projection = nn.Linear(teacher_feature_dim, projection_dim, bias=False)
        self._repa_auxiliary: dict[str, torch.Tensor] = {}

    def _set_alignment_loss(
        self,
        hidden: torch.Tensor,
        x: ChemGraph,
    ) -> None:
        self._repa_auxiliary = {}
        if (
            not self.training
            or self.inference_only
            or self.student_projection is None
            or self.teacher_projection is None
            or "teacher_features" not in x
            or "teacher_atomic_numbers" not in x
        ):
            return
        teacher_features = x["teacher_features"].to(hidden.device, dtype=hidden.dtype)
        elements = x["teacher_atomic_numbers"].to(hidden.device, dtype=torch.long)
        student_projected = self.student_projection(hidden)
        teacher_projected = self.teacher_projection(teacher_features)
        raw = element_aware_nce(
            student_projected,
            teacher_projected,
            elements,
            temperature=self.alignment_temperature,
        )
        self._repa_auxiliary = {
            "alignment_raw": raw,
            "alignment_weighted": raw * self.alignment_weight,
        }

    def consume_repa_auxiliary(self) -> dict[str, torch.Tensor]:
        auxiliary = self._repa_auxiliary
        self._repa_auxiliary = {}
        return auxiliary

    def forward(self, x: ChemGraph, t: torch.Tensor) -> ChemGraph:
        frac_coords, lattice, atom_types, num_atoms, batch = (
            x["pos"],
            x["cell"],
            x["atomic_numbers"],
            x["num_atoms"],
            x.get_batch_idx("pos"),
        )
        z_per_crystal = self.noise_level_encoding(t).to(lattice.device)
        conditions_base_model = get_property_embeddings(
            property_embeddings=self.property_embeddings,
            batch=x,
        )
        if len(conditions_base_model) > 0:
            z_per_crystal = torch.cat([z_per_crystal, conditions_base_model], dim=-1)
        conditions_adapt_dict = {}
        conditions_adapt_mask_dict = {}
        for cond_field, property_embedding in self.property_embeddings_adapt.items():
            conditions_adapt_dict[cond_field] = property_embedding.forward(batch=x)
            try:
                conditions_adapt_mask_dict[cond_field] = get_use_unconditional_embedding(
                    batch=x,
                    cond_field=cond_field,
                )
            except KeyError:
                conditions_adapt_mask_dict[cond_field] = torch.ones_like(
                    x["num_atoms"], dtype=torch.bool
                ).reshape(-1, 1)
        output = self.gemnet(
            z=z_per_crystal,
            frac_coords=frac_coords,
            atom_types=atom_types,
            num_atoms=num_atoms,
            batch=batch,
            lengths=None,
            angles=None,
            lattice=lattice,
            edge_index=None,
            to_jimages=None,
            num_bonds=None,
            cond_adapt=conditions_adapt_dict,
            cond_adapt_mask=conditions_adapt_mask_dict,
        )
        base_hidden = output.node_embeddings
        hidden = self.repa_adapter(base_hidden) if self.repa_enabled else base_hidden
        self._set_alignment_loss(hidden, x)
        pred_atom_types = self.fc_atom(hidden)
        return get_chemgraph_from_denoiser_output(
            pred_atom_types=pred_atom_types,
            pred_lattice_eps=output.stress,
            pred_cart_pos_eps=output.forces,
            training=self.training,
            element_mask_func=self.element_mask_func,
            x_input=x,
        )
