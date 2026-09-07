#!/usr/bin/env bash
# clocks.sh — lock GB10 GPU clocks for serving (rodman80's validated lever:
# +5-7% decode, +2-3% prefill vs stock ~2281 MHz, no excessive heat; measured
# on this exact image/lane). Run on EACH node before launch. Idempotent;
# re-apply after a GPU reset/host reboot. `reset` returns to stock auto-boost.
# Uses a privileged throwaway container (no sudo needed, needs docker perms).
# Usage: ./tools/clocks.sh [2400|reset]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
[ -f "$SCRIPT_DIR/.env" ] && { set -a; source "$SCRIPT_DIR/.env"; set +a; }
IMAGE="${IMAGE:-ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6}"
MHZ="${1:-2400}"

run_gpu() {
  docker run --rm --privileged --gpus all --entrypoint '' "$IMAGE" nvidia-smi "$@"
}

if [ "$MHZ" = "reset" ]; then
  echo "[clocks] resetting to stock auto-boost"
  run_gpu -rgc >/dev/null 2>&1 || true
  sleep 4
elif [[ "$MHZ" =~ ^[0-9]+$ ]]; then
  echo "[clocks] locking to ${MHZ}MHz"
  run_gpu -lgc "$MHZ,$MHZ"
else
  echo "usage: $0 [max_mhz|reset] (default 2400)" >&2
  exit 2
fi

sleep 4
echo "[clocks] current graphics clock: $(nvidia-smi -q -d CLOCK 2>/dev/null | awk '/Graphics/{print $3; exit}')"
