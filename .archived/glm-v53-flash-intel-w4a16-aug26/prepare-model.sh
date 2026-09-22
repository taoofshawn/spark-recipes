#!/usr/bin/env bash
#
# Intel AutoRound W4A16 -> GPTQ surgery for the GLM-5.3-Flash INT4 checkpoint.
#
# The Intel repo ships `quantization_config.quant_method: "auto-round"`, which
# no GB10 vLLM fork recognizes. But the tensors are plain GPTQ
# (`packing_format: auto_round:auto_gptq`, sym, group-128) — so we materialize
# a SYNTHETIC REVISION (default: "gptq-surgery") inside the model's own HF hub
# cache entry: a snapshot dir of hardlinks from the pinned upstream snapshot
# (zero extra space) whose config.json swaps the auto-round quantization_config
# for a stock GPTQ config whose `dynamic` rules re-express every auto-round
# `extra_config` exclusion (the BF16 layers: attention, router/gate, shared
# experts, indexer, visual tower, norms, ...). A one-line refs/<rev> file makes
# that revision resolvable OFFLINE, so compose serves the HF repo ID directly:
#
#   vllm serve Intel/GLM-5.3-Flash-W4A16-AutoRound --revision gptq-surgery
#
# Serving the HF ID (instead of a local directory path) keeps /v1/models
# .root — and tool-eval-bench's "Model:" banner — at the real HF id (parity
# with the deepseek-v4-flash-vision-0rand recipe, which serves its HF ID from
# the cache the same way).
#
# This is the surgery described by @miken on the NVIDIA forum
# (topic 382041, post 5): "auto-round isn't in the GB10 forks'
# QUANTIZATION_METHODS, but the tensors are plain GPTQ (auto_round:auto_gptq,
# sym g128). Symlink the snapshot into a fresh dir and swap
# `quantization_config` in `config.json`; every `extra_config` exclusion
# becomes a `dynamic` `-:regex` skip rule."
#
# Run on BOTH nodes (TP=2 loads ranks from local disk). Idempotent: a second
# run reuses the existing hardlinks and only refreshes config.json + the ref.
#
# Requires: the HF snapshot already downloaded (see README step 0):
#   hf download Intel/GLM-5.3-Flash-W4A16-AutoRound --revision <rev>
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# Capture caller-provided values so they override the .env defaults below.
PRE_MODEL="${MODEL:-}"; PRE_REV="${MODEL_REVISION:-}"
PRE_HF="${HF_CACHE:-}"; PRE_SURG="${SURGERY_REVISION:-}"
[ -f "$SCRIPT_DIR/.env" ] && { set -a; source "$SCRIPT_DIR/.env"; set +a; }
MODEL="${PRE_MODEL:-$MODEL}"
MODEL_REVISION="${PRE_REV:-$MODEL_REVISION}"
HF_CACHE="${PRE_HF:-$HF_CACHE}"
SURGERY_REVISION="${PRE_SURG:-$SURGERY_REVISION}"

MODEL="${MODEL:-Intel/GLM-5.3-Flash-W4A16-AutoRound}"
MODEL_REVISION="${MODEL_REVISION:-5eee1846f0321058ed73745f9aa16f2aaf0fc0a0}"
HF_CACHE="${HF_CACHE:-/home/sdrew/.cache/huggingface}"
SURGERY_REVISION="${SURGERY_REVISION:-gptq-surgery}"

HUB="$HF_CACHE/hub/models--$(printf '%s' "$MODEL" | sed 's|/|--|g')"
SNAP="$HUB/snapshots/$MODEL_REVISION"
SURG="$HUB/snapshots/$SURGERY_REVISION"

[ -f "$SNAP/config.json" ] || {
  echo "FATAL: HF snapshot not found at $SNAP" >&2
  echo "  Download first: hf download $MODEL --revision $MODEL_REVISION" >&2
  echo "  (see README; does the HF cache live elsewhere? override HF_CACHE)" >&2
  exit 2
}

echo "[prepare-model] source snapshot : $SNAP"
echo "[prepare-model] surgery revision: $SURG"

mkdir -p "$SURG" "$HUB/refs"

# ---- hardlink every file except config.json (kept as a real file below) ----
linked=0
for f in "$SNAP"/*; do
  b="$(basename "$f")"
  [ "$b" = "config.json" ] && continue
  if [ ! -e "$SURG/$b" ]; then
    ln "$(readlink -f "$f")" "$SURG/$b" 2>/dev/null \
      || ln -s "$(readlink -f "$f")" "$SURG/$b"
    linked=$((linked+1))
  fi
done

# ---- rewrite config.json: auto-round config -> stock GPTQ config ------------
python3 - "$SNAP/config.json" "$SURG" <<'PY'
import json, os, sys

src, dst = sys.argv[1], sys.argv[2]
cfg = json.load(open(src))
qc = cfg.get("quantization_config") or {}

if qc.get("quant_method") != "auto-round":
    print(f"FATAL: {src} is not an auto-round config (quant_method={qc.get('quant_method')!r}); nothing to do")
    sys.exit(2)

extra = qc.get("extra_config") or {}
if not extra:
    print("FATAL: auto-round extra_config is empty; cannot derive dynamic skip rules")
    sys.exit(2)

# Keep a copy of the pristine auto-round config next to the surgery output
# (the snapshot's own config.json.autoround.bak is the PRE-quantization
# config — a different file — and is hardlinked in by the loop above).
bak = os.path.join(dst, "config.json.autoround.orig.bak")
if not os.path.exists(bak):
    with open(bak, "w") as fh:
        json.dump(cfg, fh, indent=2)

new_qc = {
    "quant_method": "gptq",
    "bits": int(qc.get("bits", 4)),
    "group_size": int(qc.get("group_size", 128)),
    "sym": bool(qc.get("sym", True)),
    "desc_act": False,
    "lm_head": False,
    "true_sequential": True,
    # every auto-round BF16 exclusion (679 rules) becomes a vLLM GPTQ
    # `dynamic` skip: "-:<regex>" -> module left unquantized (BF16 load).
    "dynamic": {f"-:{k}": {} for k in extra},
}
cfg_out = dict(cfg)
cfg_out["quantization_config"] = new_qc
with open(os.path.join(dst, "config.json"), "w") as fh:
    json.dump(cfg_out, fh, indent=2)

print(f"[prepare-model] surgery OK: {dst}")
print(f"[prepare-model]   bits={new_qc['bits']} group_size={new_qc['group_size']} "
      f"sym={new_qc['sym']} desc_act={new_qc['desc_act']}")
print(f"[prepare-model]   dynamic skip rules: {len(new_qc['dynamic'])} "
      f"(from extra_config: bit-16 exclusions)")
PY

# ---- refs file: make the synthetic revision resolvable offline --------------
# huggingface_hub resolves <revision> -> refs/<revision> -> snapshots/<hash>;
# offline mode reads exactly that. Write WITHOUT a trailing newline — the hub
# reads the file verbatim (f.read(), no strip), so a '\n' would break
# pointer resolution.
printf '%s' "$SURGERY_REVISION" > "$HUB/refs/$SURGERY_REVISION"
echo "[prepare-model] refs/$SURGERY_REVISION -> snapshots/$SURGERY_REVISION (offline-resolvable)"

# ---- sanity ----------------------------------------------------------------
python3 - "$SURG" <<'PY'
import json, os, sys

d = sys.argv[1]
cfg = json.load(open(os.path.join(d, "config.json")))
qc = cfg["quantization_config"]
assert qc["quant_method"] == "gptq", qc["quant_method"]
assert qc["sym"] is True and qc["desc_act"] is False
assert qc["bits"] == 4 and qc["group_size"] == 128
assert len(qc["dynamic"]) >= 600, f"only {len(qc['dynamic'])} dynamic rules?"
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
missing = [s for s in set(idx["weight_map"].values()) if not os.path.exists(os.path.join(d, s))]
assert not missing, f"missing shards: {missing}"
print(f"[prepare-model] sanity OK: gptq config + {len(idx['weight_map'])} tensor entries, "
      f"{len(set(idx['weight_map'].values()))} shard files present")
PY

echo "[prepare-model] next: docker compose up (worker first) — see README."
echo "[prepare-model] serve identity: $MODEL @ revision $SURGERY_REVISION"
