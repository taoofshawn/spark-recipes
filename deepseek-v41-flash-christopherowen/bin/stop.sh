#!/usr/bin/env bash
# bin/stop.sh — capture evidence, then stop THIS recipe's containers and REMOVE them.
#
# Remove-by-default is the repo rule (AGENTS.md "Teardown rule"): a preserved container
# blocks the next start on this lane too — upstream's `cluster start` refuses an existing
# exact container unless --replace, and its memguard unit can be left armed. Warm JIT caches
# are unaffected: they live in {checkout}/cache, not in the container layer.
#
# usage: bin/stop.sh [--keep-containers] [--dry-run]
#   --keep-containers  stop but leave the containers + memguard for inspection
#   --dry-run          show what would happen, do nothing
#
# Steps:
#   1. capture `docker logs` for every container this recipe owns (ALL FOUR nodes) into
#      logs/ — the container is the only place this lane keeps its log, and step 4 destroys it
#   2. delegate to upstream `cluster stop` (which also stops the memguard units)
#   3. verify: re-list, report anything still present by name + state
#   4. unless --keep-containers: remove this recipe's containers, verify absence
set -uo pipefail

RECIPE=$(cd "$(dirname "$0")/.." && pwd)
KEEP=0
DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --keep-containers) KEEP=1 ;;
    --dry-run)         DRY=1 ;;
    -h|--help)         sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)                 printf 'unknown arg: %s\n' "$1" >&2; exit 2 ;;
  esac
  shift
done

# shellcheck source=../site/lane.env
. "$RECIPE/site/lane.env"

GUARD_FACTS="$RECIPE/site/fabric/cluster.env"
GUARD_LAUNCHER="$RECIPE/bin/start.sh"
# shellcheck source=guard.sh
. "$RECIPE/bin/guard.sh"

LOG_DIR="$RECIPE/logs"
STAMP=$(date +%Y%m%d-%H%M%S)
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8)
rc=0

run_on() { # <node> <cmd>
  local node=$1 host; shift
  if [ "$node" = "$(hostname -s 2>/dev/null)" ]; then host="$node"; else host="$node.shawndo.intra"; fi
  ssh "${SSH_OPTS[@]}" "$host" "$*" 2>/dev/null
}

# ---------------------------------------------------------------- 1. capture logs
printf '[stop] capturing container logs to %s\n' "$LOG_DIR"
[ "$DRY" = 1 ] || mkdir -p "$LOG_DIR"
for node in ${NODES:-}; do
  for ctn in ${CONTAINER_NAMES:-}; do
    out="$LOG_DIR/$STAMP-$node-$ctn.log"
    if [ "$DRY" = 1 ]; then
      printf '[stop]   would capture %s:%s -> %s\n' "$node" "$ctn" "$out"
      continue
    fi
    if run_on "$node" "docker logs $ctn" >"$out" 2>&1; then
      printf '[stop]   %s %s (%s lines)\n' "$node" "$ctn" "$(wc -l <"$out" | tr -d ' ')"
    else
      : >"$out"   # keep an empty marker so a missing log is visible, not silent
      printf '[stop]   %s %s: no log captured (absent or unreadable)\n' "$node" "$ctn"
    fi
  done
done

if [ "$DRY" = 1 ]; then
  printf '[stop] dry run — would now run, from %s:\n' "$DEPLOY_PATH"
  printf '        python3 bin/spark --cluster-config %s cluster stop --remove --apply --parallel\n' "$CONFIG_REL"
  printf '[stop] then %s\n' "$([ "$KEEP" = 1 ] && echo "leave containers in place (--keep-containers)" || echo "verify absence and remove leftovers")"
  exit 0
fi

# ---------------------------------------------------------------- 2. delegate
printf '[stop] delegating to upstream cluster stop (stops the memguards too)\n'
if ! ( cd "$DEPLOY_PATH" && python3 bin/spark --cluster-config "$CONFIG_REL" cluster stop --remove --apply --parallel ); then
  printf '[stop] launcher stop reported a problem (continuing to verification)\n' >&2
  rc=1
fi

# ---------------------------------------------------------------- 3. verify
leftovers=""
for node in ${NODES:-}; do
  listing=$(run_on "$node" "docker ps -a --format '{{.Names}} {{.Status}}'")
  for ctn in ${CONTAINER_NAMES:-}; do
    printf '%s\n' "$listing" | grep -qF "$ctn" && leftovers="$leftovers $node:$ctn"
  done
done

if [ "$KEEP" = 1 ]; then
  printf '[stop] --keep-containers: leaving containers in place'
  [ -n "$leftovers" ] && printf ' (%s)' "$leftovers"
  printf '\n[stop] inspect with: docker logs <name>; remove later with ./bin/stop.sh\n'
  exit "$rc"
fi

if [ -z "$leftovers" ]; then
  printf '[stop] verified: no containers from this recipe remain\n'
  exit "$rc"
fi

# ---------------------------------------------------------------- 4. remove
printf '[stop] removing:%s\n' "$leftovers"
for item in $leftovers; do
  node=${item%%:*}; ctn=${item#*:}
  if run_on "$node" "docker rm -f $ctn" >/dev/null 2>&1; then
    printf '[stop]   removed %s:%s\n' "$node" "$ctn"
  else
    printf '[stop]   FAILED to remove %s:%s\n' "$node" "$ctn" >&2
    rc=1
  fi
done

still=""
for item in $leftovers; do
  node=${item%%:*}; ctn=${item#*:}
  run_on "$node" "docker ps -a --format '{{.Names}}'" | grep -qF "$ctn" && still="$still $node:$ctn"
done
if [ -z "$still" ]; then
  printf '[stop] verified: all removed\n'
else
  printf '[stop] STILL PRESENT:%s\n' "$still" >&2
  rc=1
fi

if [ "$rc" -ne 0 ]; then
  printf '[stop] finished with problems (see above)\n' >&2
else
  printf '[stop] done — logs in %s\n' "$LOG_DIR"
fi
exit "$rc"