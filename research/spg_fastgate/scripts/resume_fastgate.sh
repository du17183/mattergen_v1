#!/usr/bin/env bash
set -euo pipefail

stop_marker="/data/dxl/results/spg_fastgate/progress/STOP_REQUESTED"
if [[ -f "$stop_marker" ]]; then
  mv "$stop_marker" "${stop_marker}.cleared.$(date +%Y%m%dT%H%M%S)"
fi
exec /data/dxl/mattergen_v1/research/spg_fastgate/scripts/run_fastgate.sh
