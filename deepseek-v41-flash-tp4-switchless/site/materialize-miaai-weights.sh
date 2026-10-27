#!/usr/bin/env bash
# materialize-miaai-weights.sh — materialize the DeepSeek-V4.1-Flash-1017 checkpoint as a
# real-file directory for the deepseek lane (NFS_SHARE=0 local weights; "miaai" in the name
# is historical — the file name and the ds41f-miaai node paths outlived the 2026-10-26
# vehicle swap to the knapcio stack, since renaming would force re-materialization +
# docker-volume recreation).
#
# Why: the launcher mounts MODEL_DIR directly on the head and binds a docker volume
# (NFS_VOLUME=dsv41-local-weights → $HOME/.local/ds41f-miaai/weights/deepseek-v41-flash-1017)
# on each worker. HF snapshots use relative symlinks into blobs/, which break inside a
# container bind mount — the same reason the NVFP4 lane materializes (site/materialize-
# weights.sh). The HF cache and destination share each node's root filesystem, so
# `cp -al` hardlinks: zero data copied, provably unmodified bytes (blob inode identity
# preserved).
#
# Usage (per node, from the repo root or with explicit args):
#   ./materialize-miaai-weights.sh <node>          # via ssh
#   ./materialize-miaai-weights.sh local           # on the current host
# Idempotent: a complete destination is verified, not rebuilt.
set -euo pipefail

REV=1bae0e1088993b9b82de08f59de5be5a34e42170   # DeepSeek-V4.1-Flash-1017 pin
EXPECTED_SHARDS=48
DEST=$HOME/.local/ds41f-miaai/weights/deepseek-v41-flash-1017
SRC=$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/$REV

REMOTE=$1
run() {
  if [[ "$REMOTE" == "local" ]]; then bash -c "$1"; else ssh -o BatchMode=yes "$REMOTE" "$1"; fi
}

run "
set -e
test -f $SRC/config.json || { echo \"snapshot missing: $SRC\" >&2; exit 1; }
n=\$(find $SRC -maxdepth 1 -name 'model-*-of-*.safetensors' 2>/dev/null | wc -l)
[ \"\${n:-0}\" -ge $EXPECTED_SHARDS ] || { echo \"snapshot incomplete: \$n shards\" >&2; exit 1; }
mkdir -p $(dirname $DEST)
if [ -f $DEST/config.json ] && [ \$(find $DEST -maxdepth 1 -name 'model-*-of-*.safetensors' 2>/dev/null | wc -l) -ge $EXPECTED_SHARDS ]; then
  echo \"$DEST complete (\$n shards) — verify only\"
  exit 0
fi
rm -rf $DEST.tmp
cp -alL $SRC/. $DEST.tmp/
mv $DEST.tmp $DEST
echo \"$DEST materialized (\$(find $DEST -name 'model-*-of-*.safetensors' | wc -l) shards, hardlinked)\"
"
