#!/usr/bin/env bash
set -euo pipefail

mps_root="/data/dxl/results/mps_fastgate/mps_runtime"
pipe_dir="$mps_root/pipe"
log_dir="$mps_root/log"
pid_file="$mps_root/control.pid"
export CUDA_MPS_PIPE_DIRECTORY="$pipe_dir"
export CUDA_MPS_LOG_DIRECTORY="$log_dir"
export CUDA_VISIBLE_DEVICES="${MPS_VISIBLE_GPUS:-0}"

case "${1:-}" in
  start)
    mkdir -p "$pipe_dir" "$log_dir"
    if [[ -S "$pipe_dir/control" ]]; then
      echo "project MPS control socket already exists: $pipe_dir/control"
      exit 1
    fi
    nvidia-cuda-mps-control -d
    for _ in $(seq 1 50); do
      [[ -S "$pipe_dir/control" ]] && break
      sleep 0.1
    done
    [[ -S "$pipe_dir/control" ]]
    control_pid="$(pgrep -n -u "$(id -u)" -f 'nvidia-cuda-mps-control -d')"
    [[ -n "$control_pid" ]]
    tr '\0' '\n' < "/proc/$control_pid/environ" | grep -Fx "CUDA_MPS_PIPE_DIRECTORY=$pipe_dir" >/dev/null
    printf '%s\n' "$control_pid" > "$pid_file"
    ;;
  status)
    if [[ ! -S "$pipe_dir/control" ]]; then
      echo "stopped"
      exit 1
    fi
    printf 'get_server_list\n' | nvidia-cuda-mps-control
    ;;
  stop)
    control_pid=""
    [[ -f "$pid_file" ]] && control_pid="$(<"$pid_file")"
    if [[ -S "$pipe_dir/control" ]]; then
      printf 'quit\n' | nvidia-cuda-mps-control || true
    fi
    for _ in $(seq 1 100); do
      if [[ -z "$control_pid" ]] || ! kill -0 "$control_pid" 2>/dev/null; then
        exit 0
      fi
      sleep 0.1
    done
    echo "MPS control daemon did not exit cooperatively" >&2
    exit 1
    ;;
  clean)
    if [[ -f "$pid_file" ]] && kill -0 "$(<"$pid_file")" 2>/dev/null; then
      echo "refusing cleanup while project MPS control daemon is alive" >&2
      exit 1
    fi
    find "$pipe_dir" -mindepth 1 -maxdepth 1 -type s -delete 2>/dev/null || true
    find "$pipe_dir" -mindepth 1 -maxdepth 1 -type f -delete 2>/dev/null || true
    find "$log_dir" -mindepth 1 -maxdepth 1 -type f -delete 2>/dev/null || true
    [[ -f "$pid_file" ]] && rm "$pid_file"
    rmdir "$pipe_dir" "$log_dir" "$mps_root" 2>/dev/null || true
    ;;
  *)
    echo "usage: $0 {start|status|stop|clean}" >&2
    exit 2
    ;;
esac
