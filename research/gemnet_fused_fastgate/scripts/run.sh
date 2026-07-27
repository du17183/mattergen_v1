#!/usr/bin/env bash
set -euo pipefail

session="mattergen_gemnet_fused_fastgate"
project="/data/dxl/mattergen_v1"
python="/data/dxl/envs/mattergen_py310/bin/python"
log="/data/dxl/logs/gemnet_fused_fastgate/gemnet_fused_fastgate.log"
progress="/data/dxl/results/gemnet_fused_fastgate/progress/master_progress.json"

if tmux has-session -t "$session" 2>/dev/null; then
  echo "session already exists: $session"
  exit 0
fi

stage="profile"
if [[ -f "$progress" ]]; then
  termination="$(jq -r '.termination_state // empty' "$progress")"
  profile_status="$(jq -r '.stages.hotspot_profile.status' "$progress")"
  fusion_status="$(jq -r '.stages.fusion_go_no_go.status' "$progress")"
  runtime_status="$(jq -r '.stages.runtime_fallback.status' "$progress")"
  if [[ -n "$termination" ]]; then
    echo "fast-gate already terminated: $termination"
    exit 0
  elif [[ "$fusion_status" == "failed" && "$runtime_status" != "success" ]]; then
    stage="runtime"
  elif [[ "$profile_status" == "success" ]]; then
    stage="fusion"
  fi
fi

case "$stage" in
  profile) module="research.gemnet_fused_fastgate.profile_hotspots" ;;
  fusion) module="research.gemnet_fused_fastgate.validate_and_benchmark" ;;
  runtime) module="research.gemnet_fused_fastgate.persistent_runtime" ;;
esac

tmux new-session -d -s "$session" \
  "cd '$project' && PYTHONPATH='$project' '$python' -m '$module' >> '$log' 2>&1"
echo "started $session stage=$stage module=$module"
