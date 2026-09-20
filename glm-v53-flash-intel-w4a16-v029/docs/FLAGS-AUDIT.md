# Flag audit — glm-v53-flash-intel-w4a16 compose flags → vLLM v0.29.0

Audited 2026-09-20 by dumping `vllm serve --help=all` (2011 lines) from the
digest-pinned base image `vllm/vllm-openai:v0.29.0-aarch64@sha256:18372a72…`
(run with `--gpus all`; a bare container fails device inference and cannot
render help). In-image probes also confirmed: vLLM `0.29.0`, dist at
`/usr/local/lib/python3.12/dist-packages/vllm`, and
`INCConfig.override_quantization_method` containing the `auto-round` branch.

| compose flag (legacy) | in v0.29.0 `--help=all`? | action |
|---|---|---|
| `--served-model-name` | yes | keep |
| `--host 0.0.0.0` / `--port` | yes | keep |
| `--trust-remote-code` | yes | keep (harmless even though `Glm5NextForConditionalGeneration` is in-tree; drop only if proven unnecessary) |
| `--tensor-parallel-size` | yes | keep |
| `--gpu-memory-utilization` | yes | keep |
| `--max-model-len` | yes | keep |
| `--max-num-seqs` | yes | keep |
| `--block-size` | yes | keep (2304) |
| `--kv-cache-dtype` | yes; choices `{auto,bfloat16,float16,fp8,fp8_ds_mla,fp8_e4m3,fp8_e5m2,fp8_inc,nvfp4,…}` | keep `fp8_e4m3` |
| `--kv-cache-memory` | **RENAMED → `--kv-cache-memory-bytes`** (env `KV_CACHE_MEMORY_BYTES`) | rename in compose |
| `--moe-backend` | yes; choices include `marlin`, `triton`, `b12x`, `flashinfer_cutlass`, `humming`, … | keep `marlin` |
| `--tool-call-parser glm47` | yes | keep |
| `--enable-auto-tool-choice` | yes | keep |
| `--reasoning-parser glm45` | yes (registry: glm45 and glm47 both → `Glm47MoeParserReasoningAdapter`) | keep `glm45` |
| `--chat-template` | yes | keep (points at baked `/models/chat_template_mm.jinja`) |
| `--default-chat-template-kwargs` | yes | keep |
| `--generation-config` / `--override-generation-config` | yes / yes | keep |
| `--enforce-eager` | yes | keep (EAGER knob) |
| `--enable-flashinfer-autotune` / `--no-enable-flashinfer-autotune` | yes (negatable form present) | keep `--no-enable-flashinfer-autotune` |
| `--async-scheduling` | yes | keep (ASYNC knob) |
| `--distributed-executor-backend mp` | yes | keep |
| `--nnodes` / `--node-rank` / `--master-addr` / `--master-port` | yes | keep |
| `--headless` | yes | keep |
| `--speculative-config` | yes | keep — `{"method":"mtp","num_speculative_tokens":3,"disable_eagle_block_drop":true}`; the `disable_eagle_block_drop` field exists only via baked patch 0001 |
| `--prefix-match-unit` | yes | keep (PMU128) |
| `--enable-prefix-caching` | yes | keep (dflash lane only; kept for parity, harmless on mtp3) |
| `--enable-prompt-tokens-details` | yes | keep |
| `--max-num-batched-tokens` | yes | keep (MNBT 8192) |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` env | **SUPERSEDED → `--prefix-cache-retention-interval`** CLI flag, default 0 (#52216) | drop the env; default already 0 (flag not emitted) |
| `--prefix-cache-retention-interval` | yes | not emitted (default 0 is what we want) |

**Boot-time env carried from the legacy compose** (not CLI flags): `NCCL_*`
fabric rows, `HF_HOME`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`,
`VLLM_EXECUTE_MODEL_TIMEOUT_S` (verify name still read in v0.29.0 at bring-up;
the official recipe uses `VLLM_ENGINE_READY_TIMEOUT_S=3600` instead — if the
legacy var is ignored, switch), `PYTORCH_CUDA_ALLOC_CONF` (also set in-image),
`TORCH_CUDA_ARCH_LIST`/`FLASHINFER_CUDA_ARCH_LIST=12.1a` (baked in-image ENV),
`TORCHINDUCTOR_CACHE_DIR`/`TILELANG_CACHE_DIR` (JIT cache persistence).

**Bring-up watch items from this audit:**
1. `VLLM_EXECUTE_MODEL_TIMEOUT_S` vs `VLLM_ENGINE_READY_TIMEOUT_S` — confirm
   which timeout env the v0.29.0 engine honors; set the recipe accordingly.
2. MRV2 is now the default model runner — `VLLM_USE_V2_MODEL_RUNNER` is
   obsolete; CUDA-graph/KV accounting changed (#53306), so the 13.5 GB KV pin
   may size the pool differently. Gate: boot log KV pool ≈1.9M tokens @1M ctx.
