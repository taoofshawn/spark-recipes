#!/usr/bin/env bash
#
# Bring a spark-recipes docker-compose recipe up on the 2-node DGX Spark cluster.
#
# Two fixed nodes (DNS hostnames — the 192.168.0.x RoCE IPs are NOT routable from
# here, they're the in-cluster fabric):
#   HEAD   = spark-0f0b.shawndo.intra   rank 0 (leader, API server)
#   WORKER = spark-6d14.shawndo.intra   rank 1 (follower, headless)
#
# Worker starts FIRST; head ~30s later. Every vLLM recipe uses all GPUs, so this
# tears down whatever model container is currently up before launching the target.
#
# Usage:
#   deploy_recipe.sh <recipe-dir> [remote-branch]
#     <recipe-dir>    name of the recipe under ~/code/spark-recipes (e.g. glm-v53-flash)
#     [remote-branch] git branch to check out + pull on both nodes (default = recipe-dir)
#
# Env overrides:
#   SKIP_TEARDOWN=1   don't stop/remove currently-running containers
#   NO_REBUILD=1      launch without --build (use existing local image)
#   NO_RITUAL=1       skip the drop_caches ritual
#   SSH_OPTS          extra ssh flags (default "-o BatchMode=yes")
#
set -euo pipefail

HEAD=spark-0f0b.shawndo.intra
WORKER=spark-6d14.shawndo.intra
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes}"

RECIPE="${1:?usage: deploy_recipe.sh <recipe-dir> [remote-branch]}"
BRANCH="${2:-$RECIPE}"
REPO_DIR="~/code/spark-recipes/$RECIPE"

say() { printf '\n\033[1;34m== %s ==\033[0m\n' "$*"; }
rsh() { # rsh <host> <cmd>
  local host="$1"; shift
  ssh $SSH_OPTS "$host" "$*"
}

say "Target recipe: $RECIPE  (branch: $BRANCH)"
say "State on both nodes"
rsh "$HEAD" "docker ps -a --format '{{.Names}} {{.Status}}' ; cd ~/code/spark-recipes && git rev-parse --abbrev-ref HEAD && git log --oneline -3"
rsh "$WORKER" "docker ps -a --format '{{.Names}} {{.Status}}'"

if [ "${SKIP_TEARDOWN:-0}" != "1" ]; then
  say "Teardown: stop/remove any model container, checkout+pull $BRANCH"
  # stop any running model container by a broad name filter; then remove the target's
  # stale container (may share the compose name even across branches on the node).
  rsh "$HEAD" "for c in \$(docker ps --format '{{.Names}}'); do docker stop \$c 2>/dev/null || true; docker rm -f \$c 2>/dev/null || true; done
    cd ~/code/spark-recipes && git checkout $BRANCH 2>&1 && git pull origin $BRANCH 2>&1 | tail -3"
  rsh "$WORKER" "for c in \$(docker ps --format '{{.Names}}'); do docker stop \$c 2>/dev/null || true; docker rm -f \$c 2>/dev/null || true; done
    cd ~/code/spark-recipes && git checkout $BRANCH 2>&1 && git pull origin $BRANCH 2>&1 | tail -3"
fi

say "Drop-caches ritual (both nodes)"
if [ "${NO_RITUAL:-0}" != "1" ]; then
  rsh "$HEAD" "sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null && echo head-dropped"
  rsh "$WORKER" "sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null && echo worker-dropped"
fi

say "Compose config check"
rsh "$HEAD" "cd $REPO_DIR && docker compose --env-file .env --env-file .env.node0 config -q && echo head-config-ok"
rsh "$WORKER" "cd $REPO_DIR && docker compose --env-file .env --env-file .env.node1 config -q && echo worker-config-ok"

BUILD_FLAG=""; [ "${NO_REBUILD:-0}" = "1" ] && BUILD_FLAG="--no-build"

say "Launch WORKER (rank 1) first"
rsh "$WORKER" "cd $REPO_DIR && docker compose --env-file .env --env-file .env.node1 up -d $BUILD_FLAG"

echo "…waiting 30 s for worker rendezvous…"
sleep 30

say "Launch HEAD (rank 0, API server)"
rsh "$HEAD" "cd $REPO_DIR && docker compose --env-file .env --env-file .env.node0 up -d $BUILD_FLAG"

say "Done. Watch: ssh $HEAD 'docker logs -f glm53-nvfp4'  (readiness ~15-25 min)"
echo "Verify:  curl -s http://127.0.0.1:4000/health   and   curl -s http://127.0.0.1:4000/v1/models"
