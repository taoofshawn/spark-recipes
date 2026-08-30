#!/usr/bin/env bash
# fleet_watchdog.sh — auto-recovery for the GLM-5.3 vLLM 2-node fleet.
# Adapted from tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark
# (fleet_watchdog.sh) for this repo's docker-compose layout (2 nodes).
#
# vLLM v1 CANNOT recover a dead engine core. Docker restart policies are
# unsafe here: headless workers exit 0 on head death (on-failure never
# fires) and the dead head often never exits at all. Full orchestrated
# relaunch — worker first — is the only cure.
#
# Probes /health (NOT /v1/models: that returns 200 even with a dead engine;
# /health returns 503 on EngineDeadError). On N consecutive failures:
# tear down BOTH nodes, re-run the GB10 memory ritual, relaunch worker
# (rank 1) first, then the head (rank 0).
#
# Run on the head node:  ./tools/fleet_watchdog.sh &
set -u

### ---- config -------------------------------------------------------------
HEALTH_URL="http://127.0.0.1:${PORT:-4000}/health"
CHECK_INTERVAL=60          # seconds between probes
FAIL_THRESHOLD=3           # consecutive failures before recovery fires
CURL_TIMEOUT=15            # per-probe timeout
POST_TEARDOWN_SLEEP=10     # let master-port TIME_WAIT / NVRM settle
CONTAINER="glm53-nvfp4"
RECIPE_DIR="${RECIPE_DIR:-$HOME/code/spark-recipes/glm-v53-flash}"
# rank -> ssh target; empty string = local (head). Launch order: worker, head.
declare -A NODE=(
  [1]="sdrew@spark-6d14.shawndo.intra"
  [0]=""
)
LOCKFILE="$HOME/.fleet_watchdog.lock"
LOGFILE="$HOME/fleet_watchdog.log"
SSH_OPTS=(-o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=accept-new)
### -------------------------------------------------------------------------

log() { printf '%s %s\n' "$(date '+%F %T')" "$*" >> "$LOGFILE"; }

run_on() {  # run_on <rank> <command string>
  local rank="$1"; shift
  local target="${NODE[$rank]}"
  if [[ -z "$target" ]]; then
    bash -lc "$*" >> "$LOGFILE" 2>&1
  else
    ssh "${SSH_OPTS[@]}" "$target" "$*" >> "$LOGFILE" 2>&1
  fi
}

healthy() {
  curl -fsS -m "$CURL_TIMEOUT" "$HEALTH_URL" >/dev/null 2>&1
}

recover() {
  log "RECOVERY: tearing down both nodes"
  run_on 1 "cd '$RECIPE_DIR' && docker compose --env-file .env --env-file .env.node1 down"
  run_on 0 "cd '$RECIPE_DIR' && docker compose --env-file .env --env-file .env.node0 down"
  sleep "$POST_TEARDOWN_SLEEP"
  # GB10 memory ritual (README: mandatory pre-launch step)
  run_on 1 "sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null"
  run_on 0 "sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null"
  log "RECOVERY: relaunching worker (rank 1) first"
  run_on 1 "cd '$RECIPE_DIR' && docker compose --env-file .env --env-file .env.node1 up -d"
  sleep 30
  log "RECOVERY: relaunching head (rank 0)"
  run_on 0 "cd '$RECIPE_DIR' && docker compose --env-file .env --env-file .env.node0 up -d"
}

mkdir -p "$(dirname "$LOCKFILE")"
exec 9>"$LOCKFILE"
flock -n 9 || { echo "another watchdog is already running" >&2; exit 1; }
log "watchdog started (probing $HEALTH_URL every ${CHECK_INTERVAL}s)"

fails=0
while true; do
  sleep "$CHECK_INTERVAL"
  if healthy; then
    if (( fails )); then log "health recovered after $fails failure(s)"; fi
    fails=0
    continue
  fi
  (( fails++ ))
  log "health check FAILED ($fails/$FAIL_THRESHOLD)"
  if (( fails >= FAIL_THRESHOLD )); then
    recover
    fails=0
  fi
done
