#!/usr/bin/env bash
# Materialize the HF-cache snapshots into REAL directories that satisfy start.sh's
# bind-mount contract (-v $MODEL_DIR:/model:ro). HF snapshots use relative symlinks into
# blobs/ which do not resolve inside a container mount; this script hardlinks the
# resolved files instead of copying — zero extra disk, ZERO weight bytes modified
# (hardlinks to the same blobs; provably the pristine checkpoint).
#
# Run ON EVERY NODE, once, before first serve. Idempotent: re-runs refresh in place.
# Usage: materialize-weights.sh [hf_snapshot_dir] [dest_dir]
set -euo pipefail

HF="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL_SNAP="${1:-$HF/hub/models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/da920bb0b9f4a06727223a349e55468e38352348}"
MODEL_DST="${2:-$HOME/.local/nvfp4-tp4/weights/nvidia-glm53-nvfp4}"
DRAFT_SNAP="${3:-$HF/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a}"
DRAFT_DST="${4:-$HOME/.local/nvfp4-tp4/weights/glm53-dflash2}"
# start.sh hard-requires /model/chat_template.jinja. The pinned nvidia snapshot SHIPS it
# (verified on the 09b04e5e and da920bb inventories), so no cross-lane fallback is used:
# this recipe must work with every sibling recipe's on-node artifacts absent. If a future
# pin drops the file, pass the template explicitly as arg 5 (a tracked file or a path).
CHAT_FALLBACK="${5:-}"

[ -d "$MODEL_SNAP" ] || { echo "ERROR: model snapshot missing: $MODEL_SNAP" >&2; exit 1; }
[ -d "$DRAFT_SNAP" ] || { echo "ERROR: drafter snapshot missing: $DRAFT_SNAP" >&2; exit 1; }
mkdir -p "$(dirname "$MODEL_DST")" "$(dirname "$DRAFT_DST")"

materialize() { # <src> <dst>
  local src="$1" dst="$2"
  rm -rf "$dst"; mkdir -p "$dst"
  # -L dereferences snapshot symlinks; --link-dest hardlinks identical files back to the
  # blobs (same filesystem) instead of copying. Result: a real-file tree, ~0 extra disk.
  rsync -aL --link-dest="$src/" "$src/" "$dst/"
}

echo "[materialize] model:  $MODEL_SNAP -> $MODEL_DST"
materialize "$MODEL_SNAP" "$MODEL_DST"
echo "[materialize] drafter: $DRAFT_SNAP -> $DRAFT_DST"
materialize "$DRAFT_SNAP" "$DRAFT_DST"

# start.sh passes --chat-template /model/chat_template.jinja — must exist in the model dir.
if [ ! -f "$MODEL_DST/chat_template.jinja" ]; then
  if [ -n "$CHAT_FALLBACK" ] && [ -f "$CHAT_FALLBACK" ]; then
    cp "$CHAT_FALLBACK" "$MODEL_DST/chat_template.jinja"
    echo "[materialize] chat_template.jinja copied from explicit arg 5: $CHAT_FALLBACK"
  else
    echo "ERROR: $MODEL_DST/chat_template.jinja missing — the pinned nvidia snapshot no longer" >&2
    echo "       ships it. Pass the chat template as arg 5:" >&2
    echo "         $0 <snap> <dst> <draft_snap> <draft_dst> /path/to/chat_template.jinja" >&2
    exit 1
  fi
fi

echo "[materialize] sanity (model dir):"
find "$MODEL_DST" -maxdepth 1 -type f | wc -l | xargs echo "  top-level files:"
du -sh "$MODEL_DST" "$DRAFT_DST"
echo "[materialize] OK — symlink-free real dirs; weights are hardlinks to the pristine blobs."
