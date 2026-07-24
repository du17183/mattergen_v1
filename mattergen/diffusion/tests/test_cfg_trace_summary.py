from __future__ import annotations

import json

import torch

from mattergen.diffusion.tests.test_cfg_acceleration import _batch, _sampler


def test_trace_memory_mode_writes_summary_without_trace_file(tmp_path) -> None:
    sampler = _sampler(True)
    sampler._cfg_trace_mode = "memory"
    sampler._trace_enabled = True
    sampler._trace_to_disk = False
    sampler._cfg_summary_path = tmp_path / "cfg_summary.json"
    sampler._guidance_trace_path = tmp_path / "guidance_trace.csv"
    sampler._on_sampling_start()
    sampler._sampling_context = {
        "sampling_step": 1,
        "num_steps": 10,
        "progress": 0.5,
        "phase": "predictor",
        "score_call_index": 1,
    }
    sampler._score_fn(_batch(), torch.ones(1))
    sampler._on_sampling_end(None)

    summary = json.loads(sampler._cfg_summary_path.read_text())
    assert summary["trace_mode"] == "memory"
    assert summary["trace_rows"] == 1
    assert summary["nfe"]["joint_batch_forward_count"] == 1
    assert not sampler._guidance_trace_path.exists()


def test_trace_off_preserves_full_cfg_nfe_summary(tmp_path) -> None:
    sampler = _sampler(False)
    sampler._cfg_summary_path = tmp_path / "cfg_summary.json"
    sampler._score_fn(_batch(), torch.ones(1))
    sampler._on_sampling_end(None)

    summary = json.loads(sampler._cfg_summary_path.read_text())
    assert summary["trace_mode"] == "off"
    assert summary["trace_rows"] == 0
    assert summary["nfe"]["conditional_logical_nfe"] == 1
    assert summary["nfe"]["unconditional_logical_nfe"] == 1
