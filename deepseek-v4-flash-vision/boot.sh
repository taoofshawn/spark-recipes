set -euo pipefail

# Auto-detect the RoCE-v2 IPv4 GID index unless one is set in .env.
if [ -z "${NCCL_IB_GID_INDEX:-}" ]; then
  for HCA in $(echo "${NCCL_IB_HCA}" | tr ',' ' '); do
    for i in $(seq 0 15); do
      t=$(cat /sys/class/infiniband/$HCA/ports/1/gid_attrs/types/$i 2>/dev/null || true)
      g=$(cat /sys/class/infiniband/$HCA/ports/1/gids/$i 2>/dev/null || true)
      case "$t" in *"RoCE v2"*) case "$g" in *0000:0000:0000:0000:0000:ffff:*) export NCCL_IB_GID_INDEX=$i; break 2;; esac;; esac
    done
  done
fi

echo "[ds4v] NODE_RANK=${NODE_RANK} MTP=${MTP_NUM_TOKENS} GID=${NCCL_IB_GID_INDEX}"

if [ "${ASYNC_SCHED:-1}" = "1" ]; then ASYNC_ARG="--async-scheduling"; else ASYNC_ARG=""; fi
SPECULATIVE_CONFIG="{\"method\":\"dspark\",\"num_speculative_tokens\":${MTP_NUM_TOKENS:-6},\"draft_sample_method\":\"probabilistic\",\"moe_backend\":\"b12x\"}";

# DSpark draft FULL-graph capture crashes inside flashinfer's
# SparseMlaSm120DecodeDsv4 (draft decode uses next_n-token queries the kernel
# rejects; flashinfer #4752 upstream). PIECEWISE mode maps the drafter to
# NOTE: --enable-flashinfer-autotune is deliberately OFF: with flashinfer
# 0.6.18 the autotuner crashes the first decode launch
# (SparseMlaSm120DecodeDsv4 "Check failed" from an unsupported tactic); the
# default heuristic tactic boots cleanly. NEVER put '#' comments inside the
# command continuation below — after a line-continuation backslash a '#'
# starts a comment that eats the remainder of the command.
# --block-size 256: required by the DSV4 compressed-SWA KV groups (compress
# ratios of 4/128 need pages divisible by 128; block 64 cannot form a common
# block size). The flashinfer decode kernel template accepts 256-token pages
# via our patched launcher (PATCH(dsv4-pagesize) in the image; kernel
# addressing is generic in the page size).
exec vllm serve "${MODEL_PATH}" \
  --compilation-config '{"cudagraph_mode": "PIECEWISE"}' \
  --revision "${MODEL_REVISION:-main}" \
  --served-model-name "${SERVED_MODEL_NAME:-deepseek-v4-flash-vision-exp}" \
  --host 0.0.0.0 --port "${PORT:-4000}" --trust-remote-code \
  --tensor-parallel-size "${TP_SIZE:-2}" --pipeline-parallel-size 1 \
  --kv-cache-dtype "${KV_CACHE_DTYPE:-nvfp4_ds_mla}" --block-size 256 \
  --max-model-len "${MAX_MODEL_LEN:-1048576}" --max-num-seqs "${MAX_NUM_SEQS:-6}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}" \
  --gpu-memory-utilization "${GPU_MEM:-0.83}" --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":8}' \
  ${ASYNC_ARG} --enable-chunked-prefill \
  --speculative-config "${SPECULATIVE_CONFIG}" \
  --tokenizer-mode deepseek_v4 --distributed-executor-backend mp \
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":" thinking","reasoning_end_str":" response"}' \
  --default-chat-template-kwargs "{\"thinking\":${THINKING:-true},\"reasoning_effort\":\"${REASONING_EFFORT:-high}\"}" \
  --attention-backend FLASHINFER_MLA_SPARSE_DSV4 --moe-backend b12x \
  --generation-config vllm \
  --nnodes 2 --node-rank "${NODE_RANK}" --master-addr "${MASTER_ADDR}" --master-port "${MASTER_PORT:-25000}" \
  ${HEADLESS:+--headless}
