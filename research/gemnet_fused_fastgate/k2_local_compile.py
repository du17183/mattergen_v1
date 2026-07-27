from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import torch

from mattergen.common.gemnet.layers.atom_update_block import AtomUpdateBlock, OutputBlock


@dataclass
class CompiledForward:
    name: str
    module: AtomUpdateBlock
    original_forward: Callable
    compiled_forward: Callable


def iter_k2_modules(gemnet: torch.nn.Module) -> Iterator[tuple[str, AtomUpdateBlock]]:
    """Yield exactly the nine modules in the profiled K2 update family."""
    for name, module in gemnet.named_modules():
        if type(module) in {AtomUpdateBlock, OutputBlock}:
            yield name, module


def enable_k2_local_compile(
    gemnet: torch.nn.Module,
    *,
    dynamic: bool = True,
) -> list[CompiledForward]:
    """Compile only K2 module forwards; the GemNet/sampler graph stays eager.

    The original bound methods are retained so this change is reversible and does
    not alter parameters, state_dict keys, edge ordering, or scatter semantics.
    """
    handles: list[CompiledForward] = []
    modules = list(iter_k2_modules(gemnet))
    if len(modules) != 9:
        raise RuntimeError(f"expected 9 K2 modules, found {len(modules)}")
    for name, module in modules:
        original = module.forward
        compiled = torch.compile(
            original,
            dynamic=dynamic,
            fullgraph=False,
            mode="reduce-overhead",
        )
        module.forward = compiled
        handles.append(
            CompiledForward(
                name=name,
                module=module,
                original_forward=original,
                compiled_forward=compiled,
            )
        )
    return handles


def disable_k2_local_compile(handles: list[CompiledForward]) -> None:
    for handle in handles:
        handle.module.forward = handle.original_forward


def k2_implementation_manifest(gemnet: torch.nn.Module) -> dict:
    modules = [
        {
            "name": name,
            "type": module.__class__.__name__,
            "source": "mattergen/common/gemnet/layers/atom_update_block.py",
        }
        for name, module in iter_k2_modules(gemnet)
    ]
    return {
        "implementation": "LOCAL_COMPILE",
        "precision": "strict IEEE FP32 (torch matmul precision highest)",
        "scope": "K2 AtomUpdateBlock and OutputBlock forwards only",
        "module_count": len(modules),
        "modules": modules,
        "sampler_compiled": False,
        "full_gemnet_compiled": False,
        "scatter_semantics_changed": False,
    }
