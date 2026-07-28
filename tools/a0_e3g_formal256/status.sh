#!/usr/bin/env bash
set -euo pipefail

cd /data/dxl/mattergen_v1
exec /data/dxl/envs/mattergen_py310/bin/python -m research.a0_e3g_formal256 status
