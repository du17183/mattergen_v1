from __future__ import annotations

import torch

from mattergen.diffusion.diffusion_module import DiffusionModule, T


class RepaDiffusionModule(DiffusionModule[T]):
    """MatterGen diffusion loss plus the training-only CrystalREPA objective."""

    def calc_loss(
        self,
        batch: T,
        node_is_unmasked: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        diffusion_loss, metrics = super().calc_loss(
            batch, node_is_unmasked=node_is_unmasked
        )
        consume = getattr(self.model, "consume_repa_auxiliary", None)
        if not callable(consume):
            return diffusion_loss, metrics
        auxiliary = consume()
        if not auxiliary:
            return diffusion_loss, metrics
        total = diffusion_loss + auxiliary["alignment_weighted"]
        metrics = dict(metrics)
        metrics["loss_diffusion"] = diffusion_loss.detach()
        metrics["loss_repa_alignment"] = auxiliary["alignment_raw"].detach()
        metrics["loss_repa_weighted"] = auxiliary["alignment_weighted"].detach()
        metrics["repa_positive_cosine"] = auxiliary["positive_cosine"].detach()
        return total, metrics
