from __future__ import annotations

import torch

from mattergen.common.gemnet.layers.atom_update_block import AtomUpdateBlock, OutputBlock
from research.gemnet_fused_fastgate.k2_local_compile import (
    disable_k2_local_compile,
    enable_k2_local_compile,
    iter_k2_modules,
)
from research.gemnet_fused_fastgate.persistent_runtime import tensor_digest


class NineK2Modules(torch.nn.Module):
    def __init__(self, scale_file: str):
        super().__init__()
        common = dict(
            emb_size_atom=4,
            emb_size_edge=4,
            emb_size_rbf=3,
            nHidden=1,
            activation="silu",
            scale_file=scale_file,
        )
        self.outputs = torch.nn.ModuleList(
            [
                OutputBlock(
                    **common,
                    num_targets=2,
                    direct_forces=True,
                    name=f"out_{index}",
                )
                for index in range(5)
            ]
        )
        self.atoms = torch.nn.ModuleList(
            [AtomUpdateBlock(**common, name=f"atom_{index}") for index in range(4)]
        )


def test_local_compile_scope_is_exact_and_reversible(tmp_path, monkeypatch):
    scale_file = tmp_path / "scale.json"
    scale_file.write_text("{}", encoding="utf-8")
    model = NineK2Modules(str(scale_file))
    modules = list(iter_k2_modules(model))
    assert len(modules) == 9
    originals = {name: module.forward for name, module in modules}
    state_keys = tuple(model.state_dict())

    def fake_compile(function, **_kwargs):
        def wrapper(*args, **kwargs):
            return function(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(torch, "compile", fake_compile)
    handles = enable_k2_local_compile(model)
    assert len(handles) == 9
    assert tuple(model.state_dict()) == state_keys
    assert all(module.forward is not originals[name] for name, module in modules)
    disable_k2_local_compile(handles)
    assert all(module.forward == originals[name] for name, module in modules)


def test_output_digest_is_bitwise_sensitive():
    baseline = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    repeated = baseline.clone()
    changed = baseline.clone()
    changed[0, 1] = torch.nextafter(changed[0, 1], torch.tensor(float("inf")))
    assert tensor_digest(baseline) == tensor_digest(repeated)
    assert tensor_digest(baseline) != tensor_digest(changed)
