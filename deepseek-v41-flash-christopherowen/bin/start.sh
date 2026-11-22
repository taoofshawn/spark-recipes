#!/usr/bin/env bash
# bin/start.sh — start THIS recipe, but only if nothing else is serving.
#
# The "stop the other lane first" rule is an executable precondition (bin/guard.sh), not a
# document. This script DELEGATES to the vendored launcher (`upstream/bin/spark`) and changes
# no serving value: every site value lives in upstream/config/cluster.json (shape
# `tp4-ring-4k`), and the lane's constants live in site/lane.env.
#
# usage: bin/start.sh [--dry-run] [--allow-unreachable] [--] [launcher args]
#   --dry-run            run the guard, assert upstream agrees with site/lane.env, print the
#                        exact command; start nothing
#   --allow-unreachable  proceed even if a node's state cannot be determined (fail-closed is
#                        the default: see bin/guard.sh)
#
# Run this ON THE HEAD NODE (spark-0f0b): the vendored launcher SSHes to the workers itself,
# and `cluster start` also arms the per-node memory guards.
#
# Exit: 0 started/ready-to-start · 1 something else is serving · 2 state unknown
#       3 launcher failed · 4 site drift (lane.env vs upstream/config/cluster.json)
set -uo pipefail

RECIPE=$(cd "$(dirname "$0")/.." && pwd)
DRY=0
ALLOW_UNREACHABLE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)           DRY=1 ;;
    --allow-unreachable) ALLOW_UNREACHABLE=1 ;;
    -h|--help)           sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    --)                  shift; break ;;
    *)                   break ;;
  esac
  shift
done

# shellcheck source=../site/lane.env
. "$RECIPE/site/lane.env"

GUARD_FACTS="$RECIPE/site/fabric/cluster.env"
GUARD_LAUNCHER="$RECIPE/bin/start.sh (delegates to upstream/bin/spark cluster start)"
# shellcheck source=guard.sh
. "$RECIPE/bin/guard.sh"

# ---------------------------------------------------------------- 0. site drift check
# The launcher reads upstream/config/cluster.json; the guard and this script read
# site/lane.env. They must agree, or "started" and "what started" diverge silently.
drift() {
  local f="$DEPLOY_PATH/$CONFIG_REL" got
  if [ ! -f "$f" ]; then printf '[start] missing %s — is the vendored tree present?\n' "$f" >&2; return 1; fi
  got=$(python3 - "$f" <<'PY' 2>/dev/null
import json,sys
d=json.load(open(sys.argv[1]))
c=d.get("container",{})
print(d.get("default_shape",""), c.get("name",""), c.get("image",""),
      d.get("deployment",{}).get("path",""), sep="|")
PY
)
  [ -n "$got" ] || { printf '[start] cannot parse %s\n' "$f" >&2; return 1; }
  local shape name image dpath
  IFS='|' read -r shape name image dpath <<<"$got"
  [ "$shape" = "$SHAPE" ]      || { printf '[start] site drift: cluster.json default_shape=%s, lane.env SHAPE=%s\n' "$shape" "$SHAPE" >&2; return 1; }
  [ "$name" = "$CONTAINER_NAMES" ] || { printf '[start] site drift: cluster.json container.name=%s, lane.env CONTAINER_NAMES=%s\n' "$name" "$CONTAINER_NAMES" >&2; return 1; }
  [ "$image" = "$IMAGE" ]      || { printf '[start] site drift: cluster.json image=%s, lane.env IMAGE=%s\n' "$image" "$IMAGE" >&2; return 1; }
  case "$dpath" in
    *"/deepseek-v41-flash-christopherowen/upstream") ;;
    *) printf '[start] site drift: cluster.json deployment.path=%s (want …/deepseek-v41-flash-christopherowen/upstream)\n' "$dpath" >&2; return 1 ;;
  esac
  printf '[start] site agrees: shape %s, container %s, image %s\n' "$shape" "$name" "$image"
}
drift || exit 4

# ---------------------------------------------------------------- 1. guard
guard_args=()
[ "$ALLOW_UNREACHABLE" = 1 ] && guard_args+=(--allow-unreachable)
guard_check "${guard_args[@]+"${guard_args[@]}"}"
grc=$?
if [ "$grc" -ne 0 ]; then
  printf '[start] refusing to start (guard exit %d). Nothing was launched.\n' "$grc" >&2
  exit "$grc"
fi

# ---------------------------------------------------------------- 2. launch
LAUNCH=(python3 bin/spark --cluster-config "$CONFIG_REL" cluster start --apply --replace)
[ "${SHAPE_UNQUALIFIED:-0}" = 1 ] && LAUNCH+=(--unqualified-shape)

if [ "$DRY" = 1 ]; then
  printf '[start] dry run — would run, from %s:\n' "$DEPLOY_PATH"
  printf '        %s %s\n' "${LAUNCH[*]}" "$*"
  printf '[start] (this plan: memory defragment, startup memguard, ranks 3->2->1->0, then\n' >&2
  printf '[start]  ready_url %s; run it ON the head node)\n' \
    "$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['deployment']['ready_url'])" "$DEPLOY_PATH/$CONFIG_REL" 2>/dev/null)" >&2
  exit 0
fi

command -v python3 >/dev/null || { printf '[start] python3 not found\n' >&2; exit 3; }
cd "$DEPLOY_PATH" || { printf '[start] cannot cd to %s\n' "$DEPLOY_PATH" >&2; exit 3; }
printf '[start] launching: %s\n' "${LAUNCH[*]}"
"${LAUNCH[@]}" "$@" || { printf '[start] launcher failed\n' >&2; exit 3; }