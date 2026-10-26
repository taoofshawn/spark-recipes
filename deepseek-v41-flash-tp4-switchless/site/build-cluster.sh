#!/usr/bin/env bash
# build-cluster.sh — build the lane image on all four nodes WITHOUT the vendored
# launcher's worker stage.
#
# Why not `start-cluster.sh build`: cmd_build's worker stage rsyncs the recipe into
# WORKER_DIR, which on workers is the engram/weights/state dir (the worker build trees
# are ~/dsv41-miaai instead), and its `--delete` is destructive there: on 2026-10-23 it
# removed ~/.local/ds41f-miaai/weights on spark-6d14, which is the bind source of the
# dsv41-local-weights volume — the volume stopped mounting until the weights were
# re-materialized (research.md, 2026-10-23 deploy gotchas). This wrapper never rsyncs;
# it streams `git archive` of the checked-out commit into each build tree (LF-safe,
# keeps the gitignored state/logs/benchmarks), the same re-sync the head stack uses.
#
# Run FROM THE HEAD NODE, in a repo clone checked out at the commit to ship:
#   bash site/build-cluster.sh [commit-ish]      # default HEAD
#
# Per node: docker build -f $BUILD_DOCKERFILE -t $IMAGE $BUILD_ARGS <tree>, then verify
# the image's adapter matches the commit byte-for-byte (sha256 of
# adapter/encoding_compat.py — the file every site mod so far has touched) before the
# script reports success. The launcher's `serve`/`stop`/`status` remain
# start-cluster.sh's job; this script only builds.
set -euo pipefail

REPO_SITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # <repo>/deepseek-v41-flash-tp4-switchless/site
LANE_DIR="$(cd "$REPO_SITE_DIR/.." && pwd)"
REPO="$(cd "$LANE_DIR/.." && pwd)"
UPSTREAM="deepseek-v41-flash-tp4-switchless/upstream"
COMMIT="${1:-HEAD}"

STACK="$HOME/.local/ds41f-miaai/stack"
WORKER_TREE='~/dsv41-miaai'

head_info() { git -C "$REPO" rev-parse --short "$COMMIT"; }
fatal() { echo "build-cluster: $*" >&2; exit 1; }

[[ -d "$STACK" ]] || fatal "no stack at $STACK (bring-up never ran?)"
git -C "$REPO" rev-parse --verify --quiet "$COMMIT" >/dev/null || fatal "no such commit: $COMMIT"
[[ -f "$LANE_DIR/site/env.switchless" ]] || fatal "missing site/env.switchless"

echo "=== build-cluster: commit $(head_info) — head + workers, no rsync"

# 1. Render the site env exactly like start-cluster.sh does and read the build knobs
#    from the rendered file (single source of truth; no duplicated defaults).
ENV_RENDER="$(mktemp)"
envsubst '$HOME' < "$LANE_DIR/site/env.switchless" > "$ENV_RENDER" 2>/dev/null \
  || sed "s|\$HOME|$HOME|g" "$LANE_DIR/site/env.switchless" > "$ENV_RENDER"
# shellcheck disable=SC1090
source <(grep -E '^(IMAGE|BASE_IMAGE|BUILD_DOCKERFILE|BUILD_ARGS|WORKER_HOSTS|WORKER_USER)=' "$ENV_RENDER")
rm -f "$ENV_RENDER"
[[ -n "${IMAGE:-}" && -n "${BASE_IMAGE:-}" && -n "${BUILD_DOCKERFILE:-}" ]] \
  || fatal "could not read IMAGE/BASE_IMAGE/BUILD_DOCKERFILE from the site env"

REPO_SHA="$(git -C "$REPO" show "$COMMIT:$UPSTREAM/adapter/encoding_compat.py" | sha256sum | cut -d' ' -f1)"
echo "=== adapter sentinel sha256: $REPO_SHA"

image_has_commit_adapter() {  # <host|local> <image>
  local where="$1" image="$2" got
  if [[ "$where" == local ]]; then
    got=$(docker run --rm --entrypoint sha256sum "$image" /opt/dsv41/adapter/encoding_compat.py | cut -d' ' -f1)
  else
    got=$(ssh -o BatchMode=yes "${WORKER_USER:+${WORKER_USER}@}$where" \
      "docker run --rm --entrypoint sha256sum $(printf '%q' "$image") /opt/dsv41/adapter/encoding_compat.py" | cut -d' ' -f1)
  fi
  [[ "$got" == "$REPO_SHA" ]] || { echo "build-cluster: $where image adapter mismatch ($got)" >&2; return 1; }
  echo "build-cluster: $where image carries the $REPO_SHA adapter"
}

build_one() {  # <host|local> <tree-dir>
  local where="$1" tree="$2" run_ssh=()
  # expand a leading ~ locally (printf %q would escape it and the remote cd would
  # fail on a literal '\~/dsv41-miaai'); head and workers share the same $HOME
  tree="${tree/#\~/$HOME}"
  [[ "$where" == local ]] || run_ssh=(ssh -o BatchMode=yes "${WORKER_USER:+${WORKER_USER}@}$where")
  echo "=== build on $where ($tree)"
  local cmd="
    set -e
    docker image inspect $(printf '%q' "$BASE_IMAGE") >/dev/null 2>&1 \
      || docker pull --platform linux/arm64 $(printf '%q' "$BASE_IMAGE")
    cd $(printf '%q' "$tree")
    docker build -f $(printf '%q' "$BUILD_DOCKERFILE") -t $(printf '%q' "$IMAGE") $BUILD_ARGS .
    docker image inspect -f '{{.Architecture}}' $(printf '%q' "$IMAGE") | grep -qx arm64 \
      || { echo 'not arm64' >&2; exit 1; }
    docker image inspect -f '{{.Id}}' $(printf '%q' "$IMAGE")
  "
  # local: no ssh array — execute via bash directly (an empty array would run the
  # string as a command name and die with "No such file or directory")
  if [[ "$where" == local ]]; then bash -c "$cmd" || fatal "build failed on $where"
  else "${run_ssh[@]}" "$cmd" || fatal "build failed on $where"; fi
  image_has_commit_adapter "$where" "$IMAGE"
}

# 2. Head: re-sync the stack from the commit, then build from the stack copy.
#    The stack keeps gitignored state/logs/benchmarks — archive extract only overwrites
#    tracked files. Use the stack path WITH strip-components=2 (gotcha #3: the head's
#    stack lives one level deeper than the workers' build trees; the wrong -C scatters
#    the tree next to stack/ and the build silently reuses the cached adapter layer).
git -C "$REPO" archive "$COMMIT" "$UPSTREAM" \
  | tar -C "$STACK" --strip-components=2 -xf -
STACK_SHA="$(sha256sum "$STACK/adapter/encoding_compat.py" | cut -d' ' -f1)"
[[ "$STACK_SHA" == "$REPO_SHA" ]] || fatal "stack re-sync mismatch ($STACK_SHA)"
echo "=== head stack re-synced to $(head_info)"

build_one local "$STACK"

# 3. Workers: stream the same archive into each worker build tree and build there.
for h in ${WORKER_HOSTS:-}; do
  git -C "$REPO" archive "$COMMIT" "$UPSTREAM" \
    | ssh -o BatchMode=yes "${WORKER_USER:+${WORKER_USER}@}$h" \
      "tar -C $WORKER_TREE --strip-components=2 -xf -"
  got="$(ssh -o BatchMode=yes "${WORKER_USER:+${WORKER_USER}@}$h" \
    "sha256sum $WORKER_TREE/adapter/encoding_compat.py" | cut -d' ' -f1)"
  [[ "$got" == "$REPO_SHA" ]] || fatal "worker tree re-sync mismatch on $h ($got)"
  build_one "$h" "$WORKER_TREE"
done

echo "=== build-cluster: done — $IMAGE on head + workers, adapter @ $(head_info) verified in-image"
