#!/usr/bin/env bash
# Materialize the DeepSeek-V4.1-Flash-1017 HF-cache snapshot into a REAL directory that
# satisfies the launcher's bind-mount contract (-v "$MODEL_HOST_PATH":/models/...:ro).
# HF snapshots use relative symlinks into blobs/ which do not resolve inside a container
# mount; this hardlinks the resolved files — zero extra disk, ZERO weight bytes modified
# (identical to the GLM lanes' site/materialize-weights.sh pattern).
#
# Run ON EVERY NODE, once, before first serve. Idempotent: re-runs refresh in place.
#
# NOTE: unlike the GLM lanes there is NO separate drafter checkpoint — the DSpark draft
# layers (mtp.0/1/2) live INSIDE the 48-shard checkpoint (shards 44-46; Engram rows read
# from shards 47/48 at serve time), and there is no chat_template.jinja to place (the
# dsv41-feat engine carries the template; thinking default is set via
# --default-chat-template-kwargs in the launcher).
# Usage: materialize-weights.sh [hf_snapshot_dir] [dest_dir]
set -euo pipefail

HF="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL_SNAP="${1:-$HF/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170}"
MODEL_DST="${2:-$HOME/.local/ds41f-tp4/weights/deepseek-v41-flash-1017}"

[ -d "$MODEL_SNAP" ] || { echo "ERROR: model snapshot missing: $MODEL_SNAP" >&2; exit 1; }
mkdir -p "$(dirname "$MODEL_DST")"

rm -rf "$MODEL_DST"; mkdir -p "$MODEL_DST"
# -L dereferences snapshot symlinks; --link-dest hardlinks identical files back to the
# blobs (same filesystem) instead of copying. Result: a real-file tree, ~0 extra disk.
rsync -aL --link-dest="$MODEL_SNAP/" "$MODEL_SNAP/" "$MODEL_DST/"

# Launcher contract checks (mirrors upstream/deepseek_v41_cycle_serve.sh --run):
[ -f "$MODEL_DST/config.json" ] || { echo "ERROR: config.json missing in $MODEL_DST" >&2; exit 1; }
grep -q '"DeepseekV41ForCausalLM"' "$MODEL_DST/config.json" || {
  echo "ERROR: $MODEL_DST/config.json is not DeepseekV41ForCausalLM" >&2; exit 1; }
[ -f "$MODEL_DST/model-00048-of-00048.safetensors" ] || {
  echo "ERROR: shard 48 missing — checkpoint incomplete (Engram rows read from shards 47/48 at serve time)" >&2; exit 1; }

# Encoding/ and inference/ are DeepSeek reference code, not served weights — harmless on
# disk but they add ~GiB to the bind mount; report, don't delete.
echo "[materialize] model:  $MODEL_SNAP -> $MODEL_DST"
du -sh "$MODEL_DST"
echo "[materialize] shards: $(find "$MODEL_DST" -name 'model-*.safetensors' | wc -l)/48"
echo "[materialize] OK — symlink-free real dir; weights are hardlinks to the pristine blobs."
