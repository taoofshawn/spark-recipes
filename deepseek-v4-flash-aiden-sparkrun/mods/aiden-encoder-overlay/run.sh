#!/bin/bash
# aiden-encoder-overlay: install the official 0731 encoder + corrected
# tokenizer wrapper into the aiden `sparkrun-vllm-ds4-gb10` image's vLLM
# package BEFORE `vllm serve` starts.
#
# This replicates the bind-mount overlay the `deepseek-v4-flash-aiden`
# docker-compose recipe uses, so sparkrun's mods (docker cp -> run.sh)
# apply the same 1:1 hardening:
#   - encoding_dsv4.py     -> deepseek_v4_encoding.py  (official 0731 effort
#     prompts: low=empty, high=Absolute maximum, max=Beyond maximum)
#   - deepseek_v4_wrapper.py -> deepseek_v4.py         (restores low/high/max
#     routing + `off`, and repairs tool-call argument JSON)
#
# Aiden 3.75/3.7 store the vLLM python env at /opt/venv (production-3.8 uses
# /opt/env). We target /opt/venv and bail loudly if the layout is unexpected.
set -euo pipefail

MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="${VLLM_SITE_PACKAGES:-/opt/venv/lib/python3.12/site-packages/vllm}"
TOK="$VLLM_ROOT/tokenizers"
ENCODING_TARGET="$TOK/deepseek_v4_encoding.py"
WRAPPER_TARGET="$TOK/deepseek_v4.py"
ENCODING_SRC="$MOD_DIR/encoding_dsv4.py"
WRAPPER_SRC="$MOD_DIR/deepseek_v4_wrapper.py"

echo "=== [aiden-encoder-overlay] VLLM_ROOT=$VLLM_ROOT ==="

for f in "$ENCODING_SRC" "$WRAPPER_SRC" "$TOK"; do
  [ -e "$f" ] || { echo "missing $f; ABORT" >&2; exit 1; }
done

cp -f "$ENCODING_SRC" "$ENCODING_TARGET"
cp -f "$WRAPPER_SRC" "$WRAPPER_TARGET"

# Drop stale bytecode so the *new* modules are imported.
find "$TOK" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

# Sanity check that hooks actually landed where vLLM will import them.
grep -q "normalize_tool_arguments" "$ENCODING_TARGET" || { echo "encoder overlay missing; ABORT" >&2; exit 1; }
grep -q "reasoning_effort" "$WRAPPER_TARGET" || { echo "wrapper overlay missing; ABORT" >&2; exit 1; }

echo "=== [aiden-encoder-overlay] OK: official 0731 encoder + tool-arg wrapper installed ==="
