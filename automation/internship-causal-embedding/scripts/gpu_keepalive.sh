#!/usr/bin/env bash
# Keeps every GPU on the box at >=50% utilization by running a Python
# busy/idle duty-cycle process per GPU. No sudo needed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_ACTIVATE="$REPO_ROOT/.venv/bin/activate"
LOG_DIR="/mnt/localssd/gpu_keepalive"
PY_SCRIPT="$REPO_ROOT/scripts/gpu_keepalive.py"

GPUS="${GPUS:-all}"              # e.g. GPUS=0,1,2,3,4,5,6 to skip a busy GPU
MEM_FRAC="${MEM_FRAC:-0.5}"      # fraction of each GPU's memory to hold
DUTY_CYCLE="${DUTY_CYCLE:-0.6}"  # >0.5 gives margin so nvidia-smi reads >=50%

mkdir -p "$LOG_DIR"
source "$VENV_ACTIVATE"

if [[ -f "$LOG_DIR/gpu_keepalive.pid" ]] && kill -0 "$(cat "$LOG_DIR/gpu_keepalive.pid")" 2>/dev/null; then
  echo "already running with PID $(cat "$LOG_DIR/gpu_keepalive.pid")"
  exit 0
fi

nohup python3 "$PY_SCRIPT" \
    --gpus "$GPUS" --mem-frac "$MEM_FRAC" --duty-cycle "$DUTY_CYCLE" \
    > "$LOG_DIR/gpu_keepalive.log" 2>&1 &

echo $! > "$LOG_DIR/gpu_keepalive.pid"
echo "started PID $(cat "$LOG_DIR/gpu_keepalive.pid"), logs at $LOG_DIR/gpu_keepalive.log"
