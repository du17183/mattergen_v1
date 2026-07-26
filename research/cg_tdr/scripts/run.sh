#!/usr/bin/env bash
set -euo pipefail
source /data/dxl/env.sh
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate /data/dxl/envs/mattergen_py310
cd /data/dxl/mattergen_v1
exec python research/cg_tdr/launch_teacher_generation.py
