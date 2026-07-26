from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from mattergen.crystalrepa.model import symmetric_element_aware_nce
from research.crystalrepa_repro.common import REPORTS, atomic_json, now


def main() -> None:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    count = rank + 2
    generator = torch.Generator(device=device).manual_seed(1000 + rank)
    student = torch.randn(count, 7, generator=generator, device=device, requires_grad=True)
    teacher = torch.randn(count, 7, generator=generator, device=device)
    elements = torch.tensor([(rank + index) % 4 + 1 for index in range(count)], device=device)
    loss, cosine = symmetric_element_aware_nce(student, teacher, elements, temperature=0.1)
    loss.backward()
    local = {
        "rank": rank, "count": count, "loss_finite": bool(torch.isfinite(loss)),
        "cosine_finite": bool(torch.isfinite(cosine)), "gradient_present": student.grad is not None,
        "gradient_finite": bool(student.grad is not None and torch.isfinite(student.grad).all()),
        "gradient_norm": float(student.grad.norm()) if student.grad is not None else 0.0,
    }
    gathered: list[dict | None] = [None for _ in range(world)]
    dist.all_gather_object(gathered, local)
    if rank == 0:
        passed = all(
            item is not None and item["loss_finite"] and item["cosine_finite"]
            and item["gradient_present"] and item["gradient_finite"] and item["gradient_norm"] > 0
            for item in gathered
        )
        report = {
            "schema_version": 1, "created_at": now(), "world_size": world,
            "backend": dist.get_backend(), "variable_row_counts": [item["count"] for item in gathered],
            "positive_offset_rule": "sum of preceding rank row counts", "same_element_mask": True,
            "ranks": gathered, "passed": passed,
        }
        atomic_json(REPORTS / "ddp_ea_nce_validation.json", report)
        print(json.dumps(report, indent=2))
        if not passed:
            raise RuntimeError("DDP EA-NCE validation failed")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
