#!/bin/bash
# deploy.sh - Install + launch DeepSeek-V4-Flash-0731 (eugr b12x recipe) on the
# 2-node DGX Spark cluster. Run on the HEAD node (node0 / leader).
#
# Usage:
#   ./deploy.sh            # build (if needed) + launch
#   ./deploy.sh --build    # only build & sync the b12x image
#   ./deploy.sh --launch   # only launch (assumes image already built/synced)
set -euo pipefail

# Where eugr's spark-vllm-docker (b12x) lives on this machine.
EUGER_REPO="${EUGER_REPO:-$HOME/.cache/sparkrun/eugr-spark-vllm-docker}"
# Fall back to a local clone if the sparkrun cache path isn't present.
if [ ! -d "$EUGER_REPO/.git" ]; then
    EUGER_REPO="$HOME/spark-vllm-docker"
fi

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_FILE="$RECIPE_DIR/deepseek-v4-flash-eugr.yaml"

ACTION="${1:-all}"

echo "=== DeepSeek-V4-Flash-0731 (eugr b12x) deploy ==="
echo "eugr repo : $EUGER_REPO"
echo "recipe dir: $RECIPE_DIR"

# ---------------------------------------------------------------------------
# 1) Ensure eugr's spark-vllm-docker is on the b12x branch
# ---------------------------------------------------------------------------
ensure_repo() {
    if [ ! -d "$EUGER_REPO/.git" ]; then
        echo "[deploy] cloning eugr/spark-vllm-docker into $EUGER_REPO"
        mkdir -p "$(dirname "$EUGER_REPO")"
        git clone https://github.com/eugr/spark-vllm-docker.git "$EUGER_REPO"
    fi
    cd "$EUGER_REPO"
    git fetch origin b12x >/dev/null 2>&1 || true
    if [ "$(git rev-parse --abbrev-ref HEAD)" != "b12x" ]; then
        echo "[deploy] switching eugr repo to b12x branch"
        git checkout b12x || git checkout -b b12x origin/b12x
    fi
    echo "[deploy] eugr repo on branch: $(git branch --show-current) @ $(git log --oneline -1)"
    git pull --ff-only origin b12x >/dev/null 2>&1 || true
}

# ---------------------------------------------------------------------------
# 2) Install this recipe + mods into the eugr b12x checkout
# ---------------------------------------------------------------------------
install_recipe() {
    cd "$EUGER_REPO"
    echo "[deploy] installing recipe -> recipes/deepseek-v4-flash-eugr.yaml"
    cp "$RECIPE_FILE" recipes/deepseek-v4-flash-eugr.yaml
    echo "[deploy] installing mods -> mods/"
    mkdir -p mods
    cp -r "$RECIPE_DIR/mods/." mods/
    chmod +x mods/*/run.sh 2>/dev/null || true
    echo "[deploy] recipe + mods installed"
}

# ---------------------------------------------------------------------------
# 3) Build + distribute the b12x image
# ---------------------------------------------------------------------------
build_image() {
    cd "$EUGER_REPO"
    if docker image inspect vllm-node-b12x:latest >/dev/null 2>&1; then
        echo "[deploy] vllm-node-b12x:latest already present"
        read -rp "  Rebuild anyway? [y/N] " ans
        [[ "$ans" == [yY]* ]] || { echo "[deploy] skipping build"; return; }
        ./build-and-copy.sh -c --exp-b12x --force-build
    else
        echo "[deploy] building vllm-node-b12x (--exp-b12x). First build ~20-40 min."
        ./build-and-copy.sh -c --exp-b12x
    fi
}

# ---------------------------------------------------------------------------
# 4) Launch on the cluster via run-recipe
# ---------------------------------------------------------------------------
launch() {
    cd "$EUGER_REPO"
    echo "[deploy] launching recipes/deepseek-v4-flash-eugr.yaml (daemon mode)"
    # -d: detach so the cluster keeps serving after this returns and a stray
    #     Ctrl-C / closed terminal can't tear it down.
    python3 run-recipe.py recipes/deepseek-v4-flash-eugr.yaml -d
}

ensure_repo
install_recipe

case "$ACTION" in
    --build)  build_image ;;
    --launch) launch ;;
    *)        build_image && launch ;;
esac

echo "=== done ==="
