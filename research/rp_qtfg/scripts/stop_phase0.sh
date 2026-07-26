#!/usr/bin/env bash
set -euo pipefail
progress=/data/dxl/results/rp_qtfg/phase0/progress
mkdir -p "$progress"
touch "$progress/STOP_REQUESTED"
launcher=/data/dxl/reports/rp_qtfg/phase0/launcher.json
if [[ ! -f "$launcher" ]]; then
  echo "No verified launcher metadata; stop marker created only."
  exit 0
fi
readarray -t meta < <(/data/dxl/envs/mattergen_py310/bin/python -c \
  'import json,sys; x=json.load(open(sys.argv[1])); print(x["pid"]); print(x["pgid"]); print(x["cwd"]); print(x["exe"])' \
  "$launcher")
pid=${meta[0]}
pgid=${meta[1]}
cwd=${meta[2]}
exe=${meta[3]}
actual_cwd=$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)
actual_exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)
actual_user=$(ps -o user= -p "$pid" 2>/dev/null | xargs || true)
actual_pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | xargs || true)
if [[ "$actual_user" != "ubuntu" || "$actual_cwd" != "$cwd" || "$actual_exe" != "$exe" || "$actual_pgid" != "$pgid" ]]; then
  echo "Launcher identity validation failed; stop marker created but no signal sent."
  exit 1
fi
kill -INT -- "-$pgid"
echo "SIGINT sent to verified RP-QTFG process group $pgid"
