#!/usr/bin/env bash
# Entrypoint for model-name-proxy (OpenResty).
#
# Renders nginx.conf from the template:
#   1. generates the sub_filter response rules from $SPOOF_FROM (comma-separated
#      real served model names) and injects them at __SPOOF_RULES_*__
#   2. injects BACKEND_MODEL and expands $SPOOF_MODEL / $UPSTREAM with envsubst
#
# Env:
#   SPOOF_MODEL    name clients see                    (default: spark-model)
#   BACKEND_MODEL  ACTIVE recipe's real served name    (default: deepseek-v4-flash)
#   SPOOF_RESPONSES  "1" = rewrite response model names too (JSON + SSE);
#                    "0" = responses untouched (mode A: see the real model)
#                    (default: 1)
set -euo pipefail
SPOOF_MODEL="${SPOOF_MODEL:-spark-model}"
BACKEND_MODEL="${BACKEND_MODEL:-deepseek-v4-flash}"
UPSTREAM="${UPSTREAM:-127.0.0.1:8000}"
SPOOF_FROM="${SPOOF_FROM:-deepseek-v4-flash,glm-5.3-flash}"
SPOOF_RESPONSES="${SPOOF_RESPONSES:-1}"

TEMPLATE="/usr/local/openresty/nginx/conf/nginx.conf.template"
OUT="/usr/local/openresty/nginx/conf/nginx.conf"

# Rules for /v1/models responses: ALWAYS spoofed in both modes — the endpoint
# must keep advertising the stable generic name regardless of mode.
MODELS_RULES=""
IFS=',' read -ra NAMES <<< "$SPOOF_FROM"
for name in "${NAMES[@]}"; do
  [ -n "$name" ] || continue
  MODELS_RULES+="        sub_filter '\"${name}\"' '\"${SPOOF_MODEL}\"';"$'\n'
done
[ -n "$MODELS_RULES" ] || { echo "FATAL: SPOOF_FROM is empty" >&2; exit 1; }

# Rules for completion/chat responses (JSON + SSE): mode B only
# (SPOOF_RESPONSES=1). Mode A leaves completion responses untouched so the
# real active model is visible there; mode B rewrites them to the spoofed name.
GENERAL_RULES=""
if [ "$SPOOF_RESPONSES" = "1" ]; then
  GENERAL_RULES="$MODELS_RULES"
fi

awk -v models_rules="$MODELS_RULES" -v general_rules="$GENERAL_RULES" -v backend="$BACKEND_MODEL" '
  { gsub(/__SPOOF_RULES_MODELS__/, models_rules) }
  { gsub(/__SPOOF_RULES_GENERAL__/, general_rules) }
  { gsub(/__BACKEND_MODEL__/, backend) }
  { print }
' "$TEMPLATE" > "$OUT"
echo "[model-name-proxy] spoof=${SPOOF_MODEL} responses=${SPOOF_RESPONSES} backend=${BACKEND_MODEL} upstream=${UPSTREAM} from=${SPOOF_FROM}"
envsubst '$SPOOF_MODEL $UPSTREAM' < "$OUT" > "$OUT.tmp" && mv "$OUT.tmp" "$OUT"
exec /usr/local/openresty/nginx/sbin/nginx -g 'daemon off;'
