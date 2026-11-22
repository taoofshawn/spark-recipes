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

GUARD_FACTS="$RECIPE/site/fabric/cluster.env"
GUARD_LAUNCHER="$RECIPE/upstream/start.sh"
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
LAUNCH=(bash "$RECIPE/upstream/start.sh" serve)

# The launcher refuses to run without upstream/.env (start.sh:7) and that file is
# GITIGNORED, so a fresh checkout has none. Materialize it from the committed site env —
# the step the README documents as `cp ../site/env.switchless .env`. Never edit it in place:
# site/env.switchless is the source of truth and the node tree stays pristine.
ENV_SRC="$RECIPE/site/env.switchless"
ENV_DST="$RECIPE/upstream/.env"
needs_env=0
if [ ! -f "$ENV_DST" ]; then
  needs_env=1
elif ! cmp -s "$ENV_SRC" "$ENV_DST"; then
  needs_env=2
fi

if [ "$DRY" = 1 ]; then
  printf '[start] dry run — would run, from %s:\n' "$RECIPE/upstream"
  printf '        %s %s\n' "${LAUNCH[*]}" "$*"
  case "$needs_env" in
    0) printf '[start] upstream/.env is present and matches site/env.switchless\n' ;;
    1) printf '[start] would materialize upstream/.env from site/env.switchless (absent)\n' ;;
    2) printf '[start] would refresh upstream/.env from site/env.switchless (differs)\n' ;;
  esac
  exit 0
fi

case "$needs_env" in
  0) : ;;
  1) cp "$ENV_SRC" "$ENV_DST" && printf '[start] materialized upstream/.env from site/env.switchless\n' ;;
  2) cp "$ENV_SRC" "$ENV_DST" && printf '[start] refreshed upstream/.env from site/env.switchless (was stale)\n' ;;
esac
[ -f "$ENV_DST" ] || { printf '[start] cannot create %s\n' "$ENV_DST" >&2; exit 3; }

printf '[start] launching: %s\n' "${LAUNCH[*]}"
cd "$RECIPE/upstream" || { printf '[start] cannot cd to %s\n' "$RECIPE/upstream" >&2; exit 3; }
"${LAUNCH[@]}" "$@" || { printf '[start] launcher failed\n' >&2; exit 3; }