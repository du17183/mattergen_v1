#!/usr/bin/env bash
set -euo pipefail
cat /data/dxl/results/cg_tdr/phase0/progress/master_progress.json
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
