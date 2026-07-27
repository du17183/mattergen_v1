#!/usr/bin/env bash
set -euo pipefail

declare -A groups=()
while read -r pid user comm args; do
  [[ "$user" == "ubuntu" && "$comm" == "python" ]] || continue
  if [[ "$args" != *"research.cg_tdr"* && "$args" != *"research/cg_tdr"* ]]; then
    continue
  fi
  cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
  exe="$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)"
  pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
  if [[ "$cwd" != "/data/dxl/mattergen_v1" \
     || "$exe" != /data/dxl/envs/*/bin/python \
     || -z "$pgid" ]]; then
    echo "Refusing unverified process: pid=$pid cwd=$cwd exe=$exe args=$args" >&2
    exit 2
  fi
  groups["$pgid"]=1
done < <(ps -eo pid=,user=,comm=,args=)

if [[ "${#groups[@]}" -eq 0 ]]; then
  echo "No active verified CG-TDR evaluation process."
  exit 0
fi

for pgid in "${!groups[@]}"; do
  echo "Sending SIGINT to verified CG-TDR process group $pgid"
  kill -INT -- "-$pgid"
done
