#!/usr/bin/env bash
set -euo pipefail
source /data/dxl/env.sh
cd /data/dxl/mattergen_v1
/data/dxl/envs/mattergen_py310/bin/python -m research.a0_e3g_compat64 status
