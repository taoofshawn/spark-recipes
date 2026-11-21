#!/usr/bin/env bash
# bin/start.sh — start THIS recipe, but only if nothing else is serving.
#
# This is the replacement for the old shared lane-switching document: the "stop the other
# lane first" rule is now an executable precondition (bin/guard.sh). The script DELEGATES
# to this recipe's launcher and changes no serving value.
#
# usage: bin/start.sh [--dry-run] [--allow-unreachable] [--] [launcher args]
#   --dry-run            run the guard and print the exact launch command, start nothing
#   --allow-unreachable  proceed even if a node's state cannot be determined (fail-closed
#                        is the default: see bin/guard.sh)
#
# Exit: 0 started/ready-to-start · 1 something else is serving · 2 state unknown
#       3 launcher failed
set -uo pipefail

RECIPE=$(cd "$(dirname "$0")/.." && pwd)
DRY=0
ALLOW_UNREACHABLE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)           DRY=1 ;;
    --allow-unreachable) ALLOW_UNREACHABLE=1 ;;
    -h|--help)           sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    --)                  shift; break ;;
    *)                   break ;;
  esac
  shift
done

GUARD_FACTS="$RECIPE/noswitch-prep/cluster.env"
GUARD_LAUNCHER="$RECIPE/upstream/scripts/tp4ctl"
# shellcheck source=guard.sh
. "$RECIPE/bin/guard.sh"

guard_args=()
[ "$ALLOW_UNREACHABLE" = 1 ] && guard_args+=(--allow-unreachable)
guard_check "${guard_args[@]+"${guard_args[@]}"}"
grc=$?
if [ "$grc" -ne 0 ]; then
  printf '[start] refusing to start (guard exit %d). Nothing was launched.\n' "$grc" >&2
  exit "$grc"
fi

# The launcher command, exactly as this recipe documents it (see README "bring-up").
# tp4ctl up runs its own health gate and refuses to start on any degradation.
LAUNCH=(bash "$RECIPE/upstream/scripts/tp4ctl" up)

if [ "$DRY" = 1 ]; then
  printf '[start] dry run — would run, from %s:\n' "$RECIPE/upstream"
  printf '        %s %s\n' "${LAUNCH[*]}" "$*"
  printf '[start] (tp4ctl reads the DEPLOYED ~/.local/tp4/cluster.env on the head;\n' >&2
  printf '[start]  after a repo config change, deploy first — see bin/FROM-ZERO.md)\n' >&2
  exit 0
fi

printf '[start] launching: %s\n' "${LAUNCH[*]}"
cd "$RECIPE/upstream" || { printf '[start] cannot cd to %s\n' "$RECIPE/upstream" >&2; exit 3; }
"${LAUNCH[@]}" "$@" || { printf '[start] launcher failed\n' >&2; exit 3; }