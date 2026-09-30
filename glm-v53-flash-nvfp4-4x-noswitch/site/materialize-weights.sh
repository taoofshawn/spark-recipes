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
MODEL_SNAP="${1:-$HF/hub/models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/09b04e5e74bca08ca8549fc736d4cdd8624bfde3}"
MODEL_DST="${2:-$HOME/.local/nvfp4-tp4/weights/nvidia-glm53-nvfp4}"
DRAFT_SNAP="${3:-$HF/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a}"
DRAFT_DST="${4:-$HOME/.local/nvfp4-tp4/weights/glm53-dflash2}"
# Fallback chat template if the nvidia snapshot does not ship chat_template.jinja
# (start.sh hard-requires /model/chat_template.jinja). The FP8 stack's deployed zai copy.
CHAT_FALLBACK="${5:-$HOME/glm53-flash-fp8-zai/chat_template.jinja}"

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
  if [ -f "$CHAT_FALLBACK" ]; then
    cp "$CHAT_FALLBACK" "$MODEL_DST/chat_template.jinja"
    echo "[materialize] chat_template.jinja copied from fallback: $CHAT_FALLBACK"
  else
    echo "ERROR: $MODEL_DST/chat_template.jinja missing and fallback absent: $CHAT_FALLBACK" >&2
    echo "       supply the zai pinned template (rev 690b7052, sha256 0c4099f3...) as arg 5." >&2
    exit 1
  fi
fi

echo "[materialize] sanity (model dir):"
find "$MODEL_DST" -maxdepth 1 -type f | wc -l | xargs echo "  top-level files:"
du -sh "$MODEL_DST" "$DRAFT_DST"
echo "[materialize] OK — symlink-free real dirs; weights are hardlinks to the pristine blobs."
