#!/usr/bin/env bash
# Build this rank's packed Engram shards (one pread per row) into the cache mount.
# Runs ON EVERY NODE, once per checkpoint. Wraps upstream/tools/pack_engram_rows.py
# inside the serving image exactly as the sparkring quickstart prescribes (~9 min/rank,
# CPU only; sparse 101 GB files, ~48 GB allocated per rank).
#
# The packer REFUSES an existing output directory (manifests don't bind rows to a
# checkpoint revision) — for a different checkpoint select a fresh --out-dir.
# Source env values come from env.switchless (MODEL_DIR, CACHE_DIR, IMAGE).
# Measured effect (sparkring cycle-tp4-dspark5-graphs-20260910.md "Engram loader rebalance"):
# prefill 16K 1590 -> 1873 tok/s, 64K 1745 -> 2058, burst TTFT p50 11.1 -> 9.8 s.
# Usage: build-engram-packed.sh [out_dir_under_cache]   (default /cache/engram-packed)
set -euo pipefail
cd "$(dirname "$0")"

# shellcheck disable=SC1090
source "${ENV_FILE:-env.switchless}"
OUT="${1:-engram-packed}"
RANK=${RANK:?set RANK=<0..3> for this node}
: "${MODEL_DIR:?}" "${CACHE_DIR:?}" "${IMAGE:?}"

mkdir -p "$CACHE_DIR/$OUT"
docker run --rm --entrypoint python3 --memory 6g \
  -v "$MODEL_DIR:/models/DeepSeek-V4.1-Flash:ro" \
  -v "$CACHE_DIR:/cache" \
  -v "$(cd ../upstream/tools && pwd):/tools:ro" \
  "$IMAGE" \
  /tools/pack_engram_rows.py --model-dir /models/DeepSeek-V4.1-Flash \
    --out-dir "/cache/$OUT" --tp 4 --rank "$RANK" --balanced
echo "[engram-packed] rank $RANK done: $CACHE_DIR/$OUT"
