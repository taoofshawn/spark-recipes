#!/usr/bin/env bash
# start-cluster.sh — site wrapper for the MiaAI TP4 switchless launcher.
#
# Renders the site env into the vendored launcher's .env.tp4-switchless on the head
# node and execs ./start-tp4-switchless.sh with the requested command (doctor | build |
# pack | serve | stop | status | logs | smoke). Run FROM THE HEAD NODE — the launcher
# SSHes to the workers itself and the workstation has no python3 remote helper.
#
# Layout on the head:
#   ~/.local/ds41f-miaai/stack/   vendored upstream tree (rsynced from the repo clone)
#   ~/.local/ds41f-miaai/.env.tp4-switchless   rendered site env (THIS file's values)
#
# Idempotent: re-renders the env every run (repo is the source of truth; the node tree
# stays pristine except for this generated file + the stack copy).
set -euo pipefail

REPO_SITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # <repo>/deepseek-v41-flash-tp4-switchless/site
ROOT=$HOME/.local/ds41f-miaai
STACK=$ROOT/stack
# The launcher exports ENV_FILE=$STACK/.env.tp4-switchless and sources it when present.
# Render the site env EXACTLY there (the site env already carries the ring block), so
# the launcher's first-run example copy never fires.
ENV_OUT=$STACK/.env.tp4-switchless
SRC_ENV=$REPO_SITE_DIR/env.switchless
# 2. Render the env: site values with $HOME-expanded paths (the launcher runs as the
#    same user on the head, so $HOME is correct).
mkdir -p "$ROOT"
envsubst '$HOME' < "$SRC_ENV" > "$ENV_OUT.tmp" 2>/dev/null || sed "s|\$HOME|$HOME|g" "$SRC_ENV" > "$ENV_OUT.tmp"
mv "$ENV_OUT.tmp" "$ENV_OUT"
export ENV_FILE="$ENV_OUT"

# 3. Exec the launcher's switchless profile.
cd "$STACK"
# exec via bash, not ./ — the vendored tree ships 0644 scripts, so a stack re-sync
# from the repo (which also ships 0644) would otherwise lose the exec bit and die
# here with "Permission denied" (hit 2026-10-22 after a tar re-sync of the stack).
exec bash ./start-tp4-switchless.sh "$@"
