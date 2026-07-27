#!/usr/bin/env bash
set -euo pipefail

progress=/data/dxl/results/cg_tdr/phase0/progress/master_progress.json
if [[ -f "$progress" ]] \
  && grep -q '"stage": "stop_for_review"' "$progress" \
  && grep -q '"overall_status": "completed"' "$progress"; then
  echo "CG-TDR evaluation is complete and frozen at stop_for_review."
  exit 0
fi

if tmux has-session -t mattergen_cg_tdr_eval 2>/dev/null; then
  echo "mattergen_cg_tdr_eval is already active."
  exit 0
fi

echo "No completed state exists. Use the stage-specific resume-safe launchers documented in research/cg_tdr/README.md."
exit 2
