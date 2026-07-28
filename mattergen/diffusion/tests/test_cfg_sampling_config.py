from __future__ import annotations

from pathlib import Path

import hydra


def test_cfg_trace_summary_keys_are_structured_sampling_options() -> None:
    config_dir = Path(__file__).resolve().parents[3] / "sampling_conf"
    with hydra.initialize_config_dir(str(config_dir), version_base=None):
        config = hydra.compose(
            config_name="default",
            overrides=[
                "sampler_partial.cfg_trace_mode=memory",
                "sampler_partial.cfg_summary_path=/tmp/cfg_summary.json",
            ],
        )
    assert config.sampler_partial.cfg_trace_mode == "memory"
    assert config.sampler_partial.cfg_summary_path == "/tmp/cfg_summary.json"
