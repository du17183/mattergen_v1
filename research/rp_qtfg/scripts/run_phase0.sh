#!/usr/bin/env bash
set -Eeuo pipefail
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=/data/dxl/mattergen_v1

cd /data/dxl/mattergen_v1

launcher_pid=$$
launcher_pgid=$(ps -o pgid= -p "$launcher_pid" | xargs)
launcher_exe=$(readlink -f "/proc/$launcher_pid/exe")
/data/dxl/envs/mattergen_py310/bin/python -c \
  'import json,os,sys; p=sys.argv[1]; v={"pid":int(sys.argv[2]),"ppid":os.getppid(),"pgid":int(sys.argv[3]),"cwd":sys.argv[4],"exe":sys.argv[5],"argv":["run_phase0.sh"]}; t=p+".tmp"; open(t,"w").write(json.dumps(v,indent=2)+"\n"); os.replace(t,p)' \
  /data/dxl/reports/rp_qtfg/phase0/launcher.json \
  "$launcher_pid" \
  "$launcher_pgid" \
  /data/dxl/mattergen_v1 \
  "$launcher_exe"

export CUDA_VISIBLE_DEVICES=0
/data/dxl/envs/fn_pra_teacher/bin/python -m research.rp_qtfg.run_phase0
/data/dxl/envs/fn_pra_teacher/bin/python -m research.rp_qtfg.offline_probe

unset CUDA_VISIBLE_DEVICES
/data/dxl/envs/mattergen_py310/bin/python \
  -m research.rp_qtfg.offline_relax launch --workers 16
/data/dxl/envs/mattergen_py310/bin/python \
  -m research.rp_qtfg.analyze_offline
