# Reproduction

Project and environment:

```bash
cd /data/dxl/mattergen_v1
source /data/dxl/env.sh
conda activate /data/dxl/envs/mattergen_py310
```

Run/resume/status/stop:

```bash
bash research/gemnet_fused_fastgate/scripts/run.sh
bash research/gemnet_fused_fastgate/scripts/resume.sh
bash research/gemnet_fused_fastgate/scripts/status.sh
bash research/gemnet_fused_fastgate/scripts/stop.sh
```

Individual decision stages:

```bash
python -m research.gemnet_fused_fastgate.profile_hotspots
python -m research.gemnet_fused_fastgate.validate_and_benchmark
python -m research.gemnet_fused_fastgate.persistent_runtime
python -m pytest tests/test_gemnet_fused_fastgate.py -q
```

The runtime runner atomically caches completed worker-level summaries and skips a
fully successful 32-seed level when resumed.
