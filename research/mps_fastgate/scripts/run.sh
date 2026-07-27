#!/usr/bin/env bash
set -euo pipefail

session="mattergen_mps_fastgate"
project="/data/dxl/mattergen_v1"
log="/data/dxl/logs/mps_fastgate/mps_fastgate.log"
env_path="/data/dxl/envs/mattergen_py310"

mkdir -p "$(dirname "$log")"

if tmux has-session -t "$session" 2>/dev/null; then
  echo "tmux session already exists: $session"
  exit 1
fi

tmux new-session -d -s "$session" \
  "source /data/dxl/env.sh && source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate '$env_path' && cd '$project' && exec python -m research.mps_fastgate.benchmark 2>&1 | tee -a '$log'"

echo "started: $session"
