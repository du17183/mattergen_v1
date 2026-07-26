#!/usr/bin/env bash
set -euo pipefail
echo "TMUX"
tmux list-panes -t mattergen_rp_qtfg_phase0 \
  -F '#{pane_pid} #{pane_current_command}' 2>/dev/null || true
echo "PROGRESS"
if [[ -f /data/dxl/results/rp_qtfg/phase0/progress/master_progress.json ]]; then
  /data/dxl/envs/mattergen_py310/bin/python -m json.tool \
    /data/dxl/results/rp_qtfg/phase0/progress/master_progress.json
fi
echo "MAG_ORACLE"
if [[ -f /data/dxl/reports/rp_qtfg/phase0/mag_oracle_report.json ]]; then
  /data/dxl/envs/mattergen_py310/bin/python -c \
    'import json,sys; x=json.load(open(sys.argv[1])); print(json.dumps({k:x.get(k) for k in ("CHGNET_MAG_ORACLE_GO","n","mae","spearman","target_region_mae","top_k_target_enrichment")},indent=2))' \
    /data/dxl/reports/rp_qtfg/phase0/mag_oracle_report.json
fi
echo "OFFLINE_PROBE"
if [[ -f /data/dxl/results/rp_qtfg/phase0/offline_probe/probe_summary.json ]]; then
  /data/dxl/envs/mattergen_py310/bin/python -m json.tool \
    /data/dxl/results/rp_qtfg/phase0/offline_probe/probe_summary.json
fi
echo "OFFLINE_RELAX"
cd /data/dxl/mattergen_v1
/data/dxl/envs/mattergen_py310/bin/python \
  -m research.rp_qtfg.offline_relax status 2>/dev/null || true
echo "GATE0_DECISION"
if [[ -f /data/dxl/reports/rp_qtfg/phase0/offline_direction/offline_direction_report.json ]]; then
  /data/dxl/envs/mattergen_py310/bin/python -c \
    'import json,sys; x=json.load(open(sys.argv[1])); print(json.dumps({k:x.get(k) for k in ("PHYSICS_DIRECTION_GO","SELECTED_OFFLINE_VARIANT","MATTERSIM_DIRECTION_AGREEMENT","ENERGY_IMPROVEMENT_RATE","FORCE_IMPROVEMENT_RATE","RMSD_IMPROVEMENT_RATE")},indent=2))' \
    /data/dxl/reports/rp_qtfg/phase0/offline_direction/offline_direction_report.json
fi
echo "GPU"
nvidia-smi \
  --query-gpu=index,memory.used,utilization.gpu \
  --format=csv,noheader
