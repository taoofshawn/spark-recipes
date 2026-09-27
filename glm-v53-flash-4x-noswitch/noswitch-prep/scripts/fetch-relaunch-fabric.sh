#!/usr/bin/env bash
# glm-v53-flash-4x-noswitch — (re)launch the FP8 weight fetch on rank 0 with the fabric
# data plane. Run: ssh spark-0f0b 'bash -s' < fetch-relaunch-fabric.sh
set -euo pipefail

# stop any previous fetch instance (exact process match, not our own argv)
pkill -f 'fetch-fp8-weights.sh$' 2>/dev/null || true
sleep 1
pgrep -fa fetch-fp8-weights | grep -v pgrep || echo "no fetch running"

cd ~/tp4
nohup env \
  HF_BIN="$HOME/.hfenv/bin/hf" \
  TP4_HOSTS="sdrew@10.69.42.170 sdrew@10.69.42.171 sdrew@10.69.42.172 sdrew@10.69.42.173" \
  XFER_HOSTS="sdrew@10.69.42.170 sdrew@10.10.1.2 sdrew@10.10.2.3 sdrew@10.10.4.4" \
  RELAY_RANK2=1 \
  ./scripts/fetch-fp8-weights.sh > "$HOME/fetch-fp8-fabric.log" 2>&1 < /dev/null &
echo "relaunched pid=$!"
sleep 15
tail -5 "$HOME/fetch-fp8-fabric.log"
