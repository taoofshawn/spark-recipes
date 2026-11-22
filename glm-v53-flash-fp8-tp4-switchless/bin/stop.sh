#!/usr/bin/env bash
# bin/stop.sh — capture evidence, then stop THIS recipe's containers and REMOVE them.
#
# Remove-by-default is the repo rule (AGENTS.md "Teardown rule", decision D12): a
# preserved container blocks the next serve on lanes whose preflight refuses existing
# names, and it makes "delete any recipe at any time" false in practice. Warm JIT caches
# are unaffected — they live in host bind-mounts, not the container layer.
#
# usage: bin/stop.sh [--keep-containers] [--dry-run]
#   --keep-containers  stop but leave the containers for inspection (skips step 4)
#   --dry-run          show what would happen, do nothing
#
# Steps:
#   1. capture `docker logs` for every rank into bin/../../logs/ (ALWAYS — the container is
#      the only place some lanes keep their log, and step 4 destroys it)
#   2. delegate to this recipe's own stop/teardown
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
    -h|--help)         sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)                 printf 'unknown arg: %s\n' "$1" >&2; exit 2 ;;
  esac
  shift
done

GUARD_FACTS="$RECIPE/noswitch-prep/cluster.env"
GUARD_LAUNCHER="$RECIPE/upstream/scripts/tp4ctl"
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
  # Log every container this recipe owns that still exists (running or stopped).
  for pat in ${CONTAINER_NAMES:-}; do
    out="$LOG_DIR/$STAMP-$node-$pat.log"
    if [ "$DRY" = 1 ]; then
      printf '[stop]   would capture %s -> %s\n' "$pat" "$out"
      continue
    fi
    if run_on "$node" "docker logs $pat" >"$out" 2>&1; then
      printf '[stop]   %s %s (%s lines)\n' "$node" "$pat" "$(wc -l <"$out" | tr -d ' ')"
    else
      : >"$out"   # keep an empty marker so a missing log is visible, not silent
      printf '[stop]   %s %s: no log captured (absent or unreadable)\n' "$node" "$pat"
    fi
  done
  # Recipes without an exact-name list: capture by pattern, one file per container.
  if [ -z "${CONTAINER_NAMES:-}" ] && [ -n "${CONTAINER_PATTERN:-}" ]; then
    listing=$(run_on "$node" "docker ps -a --format '{{.Names}}' --filter name=$(printf '%s' "$CONTAINER_PATTERN" | tr -d '*')")
    for c in $listing; do
      out="$LOG_DIR/$STAMP-$node-$c.log"
      if [ "$DRY" = 1 ]; then printf '[stop]   would capture %s -> %s\n' "$c" "$out"; continue; fi
      run_on "$node" "docker logs $c" >"$out" 2>&1 || : >"$out"
      printf '[stop]   %s %s (%s lines)\n' "$node" "$c" "$(wc -l <"$out" | tr -d ' ')"
    done
  fi
done

if [ "$DRY" = 1 ]; then
  printf '[stop] dry run — would now: bash %s/upstream/scripts/tp4ctl down\n' "$RECIPE"
  printf '[stop] then %s\n' "$([ "$KEEP" = 1 ] && echo "leave containers in place (--keep-containers)" || echo "remove this recipe's containers")"
  exit 0
fi

# ---------------------------------------------------------------- 2. delegate
printf '[stop] delegating to tp4ctl down (all ranks, rank 0 first)\n'
if ! ( cd "$RECIPE/upstream" && bash ./scripts/tp4ctl down ); then
  printf '[stop] launcher stop reported a problem (continuing to verification)\n' >&2
  rc=1
fi

# ---------------------------------------------------------------- 3. verify
leftovers=""
for node in ${NODES:-}; do
  listing=$(run_on "$node" "docker ps -a --format '{{.Names}} {{.Status}}'")
  for c in ${CONTAINER_NAMES:-}; do
    printf '%s\n' "$listing" | grep -qF "$c" && leftovers="$leftovers $node:$c"
  done
  if [ -z "${CONTAINER_NAMES:-}" ] && [ -n "${CONTAINER_PATTERN:-}" ]; then
    pre=$(printf '%s' "$CONTAINER_PATTERN" | tr -d '*')
    while IFS= read -r l; do
      case "$l" in "$pre"*) leftovers="$leftovers $node:${l%% *}" ;; esac
    done <<<"$listing"
  fi
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

# verify absence
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