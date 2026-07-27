#!/usr/bin/env bash
set -euo pipefail

progress="/data/dxl/results/spg_fastgate/progress"
mkdir -p "$progress"
date --iso-8601=seconds > "$progress/STOP_REQUESTED"
echo "Stop marker written. No new Fast Gate stage will be dispatched."
