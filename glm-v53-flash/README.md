# glm-v53-flash - GLM-5.3-Flash NVFP4 (2× DGX Spark, vLLM)

> **DRAFT: not tested** — this recipe has NOT been launched on this cluster yet.
> Boot markers, KV-pool sizes, and tok/s numbers in this README are upstream /
> NVIDIA-forum measurements and must be re-verified here. Research, upstream
> evaluation, and the "where to watch for updates" list live in
> [`research.md`](./research.md).

Serves **GLM-5.3-Flash** (arch `glm5_next` — 320B total / 18B active hybrid
MoE: 34 KDA linear-attention layers + 11 DeepSeek sparse-MLA layers, NoPE,
native MTP draft head, vision) on the 2-node DGX Spark cluster, TP=2, using
the **NVFP4** checkpoint and **vLLM**. This is a vLLM recipe (unlike the
Qwen SGLang recipe): the repo's vLLM stack is the validated path for this
model on GB10 — with eight SM121 kernel patches baked into a local image
(`patches/`).

## Configuration overview

| knob | value |
|---|---|
| Engine | vLLM day-0 image `vllm/vllm-openai:glm53-flash-arm64-cu130` (digest-pinned) |
| Image | locally built `glm53-nvfp4-sm121:local` (8 patch layers, see `patches/`) |
| Checkpoint | `LibertAIDAI/GLM-5.3-Flash-NVFP4` (194.6 GB; revision pinned) |
| KV cache | `fp8_e4m3` (patched NoPE-MLA path) |
| KV pool | 507K tokens @ 4.14 GiB/rank (default, 3/3 reliable) · 672K @ 5.5 GiB (local-weights record) |
| Spec decode | MTP 3 (in-checkpoint BF16 draft head; 3 = measured TP2 winner) |
| `gpu_memory_utilization` | 0.85 (0.78/0.80 starve the bf16 KV at 131K+) |
| `max_num_seqs` | 6 · `block-size` 2304 (kpool page invariant) · `--moe-backend marlin` |
| Context | 262,144 (TP2 ceiling — the model-native 1M needs TP4 / 4 nodes) |
| Thinking | **off** server-side (agent-safe default — see Tool-calling notes) |
| Serve | port `4000`, served model `glm-5.3-flash` |

Measured upstream references on 2× DGX Spark: **21.8–28.3 tok/s** single-stream
decode (vLLM NVFP4, MTP3/4, fp8 KV), ~1150 tok/s prefill, TTFT ~0.2–0.7 s.
This recipe's own numbers still need to be measured here (not launched yet).

---

## Prep (both nodes)

```bash
cd ~/code/spark-recipes/glm-v53-flash/
docker compose --env-file .env --env-file .env.node0 config -q   # sanity on each node

# Download the checkpoint (195 GB) on BOTH nodes — serving is offline:
hf download LibertAIDAI/GLM-5.3-Flash-NVFP4 --revision aa28e1f54130286c95fee10d0705c74ce8743734
```

The first `docker compose up` builds the patched image on each node. The build
needs network for the FlashInfer nightly + NCCL/cutlass pip re-pins (a few
minutes after the ~9.7 GB base pull).

## Run

> ⚠️ **GPU contention rule:** this recipe uses all reserved GPUs. Tear down
> any currently-running model container (DeepSeek, MiMo, Qwen) before starting.

```bash
# Both nodes, mandatory pre-launch (GB10 UMA page-cache starvation — MemFree,
# not MemAvailable, is what the NVRM allocator counts):
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
# Optional: keep the page cache small through the ~195 GB load (helps the KV
# slab allocate; the compose also runs a best-effort in-container loop):
sudo ./tools/cache_flusher.sh &

# Node 1 (follower, rank 1): start FIRST
docker compose --env-file .env --env-file .env.node1 up -d --build

# ~25-30 s later: Node 0 (leader, rank 0, API server)
docker compose --env-file .env --env-file .env.node0 up -d --build
```

Ready in ~15–25 min (195 GB weight load + warmup). Never benchmark right after
boot; first boots are slower. If a rank dies silently 1–2 min after
"Initial free memory … reserved N GiB", the KV slab was too big — drop
`KV_CACHE_MEMORY` to `4445787956` (see research.md "GB10 KV memory wall").

## Verify

```bash
# API up + model id + context (use /health for liveness — /v1/models returns
# 200 even with a dead engine):
curl -s http://127.0.0.1:4000/health
curl -s http://127.0.0.1:4000/v1/models   # -> "glm-5.3-flash", max_model_len 262144

# Boot markers to confirm (head + worker logs):
#   Loading model weights took ...          (no pe_dim=64 assert — patched image)
#   Initial free memory ... reserved        (KV slab pinned, no NV_ERR_NO_MEMORY)
#   flashinfer 0.6.18.dev20260819           (nightly, not 0.6.17 — NaN fix)
#   Version: 2.30.7  /  Version: 4.6.2     (NCCL + cutlass re-pins)
#   Uvicorn running on ...                  (ready)

# 6-way concurrent tool-carrying load test (catches silent FP4 corruption /
# degenerate loops):
python3 tools/load_test_glm.py
#  -> VERDICT: PASS
```

## Tool-calling notes (read before wiring agents)

- `--tool-call-parser glm47` — the plain `glm` parser **fails silently**
  (empty `content` + `tool_calls: null`). `--reasoning-parser glm45` splits
  the template's thinking block out of `content`.
- `THINKING=false` server-side (`--default-chat-template-kwargs
  '{"enable_thinking": false}'`). The day-0 stack has thinking+tools issues
  (SGLang #36669 shows `!` degeneration under multi-tool agentic prompts on
  this model family); the off default is the agent-safe lane. Clients can opt
  in per request with `"chat_template_kwargs": {"enable_thinking": true}`.
- `max_tokens` includes reasoning tokens when thinking is on.
- The checkpoint is multimodal; keep `LANGUAGE_MODEL_ONLY=0` (default) to
  preserve vision. Set `LANGUAGE_MODEL_ONLY=1` for a text-only endpoint that
  skips the ~15.7 GB multimodal processor (frees memory under pressure).

## Known issues & gotchas (day-0/day-1; from upstream + forum)

| symptom | cause / fix |
|---|---|
| `pe_dim must be 64 for fp8_ds_mla` at warmup | NoPE-MLA vs the stock SM12x sparse backend — **must use the patched image** (patch layer 1); stock `glm53-flash-arm64-cu130` does not boot GLM-5.3 on GB10 |
| serves but every reply is garbage / a repeated token loop | **silent FP4 MoE corruption** on sm_121 — auto-selected `FLASHINFER_CUTLASS` NvFp4 backend; fix `--moe-backend marlin` (model loads, `/health` green — this one lies) |
| NaN logits on 64–256-row batches | FlashInfer 0.6.17 FA2 MLA scheduler NaN on SM121 — patched image ships the 0.6.18 nightly |
| `ncclCommInitRank: internal error` / CuTeDSL `cute-to-nvvm` ICE | the nightly's transitive downgrades — re-pinned in the image (NCCL 2.30.7, cutlass-dsl 4.6.2) |
| decode past ~24K context → `EngineDeadError` | `persistent_topk` oversubscribes GB10's SM budget (needs 128KB smem; SM121 has ~101KB) — fixed by the `sparse_attn_indexer_kpool.py` overlay (patch layer 8). A 20K gate proves nothing; test with a 28-32K prompt |
| `NV_ERR_NO_MEMORY` in dmesg, worker dies, KV "succeeds" then first-touch death | GB10 KV wall — MemAvailable lies; pin `KV_CACHE_MEMORY` (never ride GMU), drop_caches ritual, don't exceed ~6 GiB/rank on TP2 |
| worker "Connection reset by peer" + head hangs at "Init torch distributed begin" | stale-head rendezvous — always `docker compose down` on BOTH nodes between relaunches; start worker first |
| `max-num-batched-tokens` < 2048 → silent segfault in warmup | DSA indexer `index_topk=2048` invariant — the compose refuses it |
| `No available shared memory broadcast block found in 60 seconds` | benign — FlashInfer autotuning (CPU ~150%), not a hang |
| boot death right after weight load | MTP draft head (+~5 GB) at GMU 0.85 trips UMA OOM without the KV pin — keep `KV_CACHE_MEMORY` set |
| `--load-format instanttensor` | do NOT enable in TP2 — the fast loader is unstable multi-node (rank dies silently ~60-90 s after load); safetensors is the shipped path |

## Files

```
docker-compose.yml      # build + launch (GID auto-detect, model resolve, vllm serve)
.env                    # shared config (cluster IPs, NICs, HF/JIT cache paths)
.env.node0 / .env.node1 # per-node overrides (NODE_RANK, ROCE_IP)
patches/
  Dockerfile            # FROM vllm/vllm-openai:glm53-flash-arm64-cu130 (digest-pinned) + 8 patch layers
  patch_v7.py           # indexer top-k init + pool clamp                          [tonyd2wild]
  patch_v8_fp8.py       # fp8 KV cache for the NoPE-MLA path on SM12x             [tonyd2wild]
  sparse_attn_indexer_kpool.py  # persistent_topk SM121 gate (24K-context crash)  [tonyd2wild]
tools/
  cache_flusher.sh      # host-side GB10 page-cache guard during load             [tonyd2wild]
  load_test_glm.py      # 6-way concurrent tool-carrying load test (degenerate-loop check)
README.md               # this file (run/build)
research.md             # research notes + future-work handoff for agents
```

---

# Audit trail

- **2026-08-28 — initial recipe (DRAFT, not tested).** GLM-5.3-Flash NVFP4 on
  2× DGX Spark via vLLM (the repo's native engine), translated from
  tonyd2wild's world-first deploy into the repo's docker-compose conventions
  (port 4000, offline HF cache on both nodes, worker-first start, GID
  auto-detect). Consolidates tonyd2wild's v1→v8 patch ladder (SM121 NoPE-MLA,
  FlashInfer 0.6.18, NCCL/cutlass re-pins, PDL off, indexer hardening, fp8 KV,
  persistent_topk gate) into one reproducible Dockerfile. v9 (InstantTensor)
  deliberately excluded (unstable in TP2). Defaults: MTP3 (measured TP2
  winner), KV pin 4.14 GiB (3/3 reliable; 5.5 GiB local-weights record
  documented as the upgrade). Not yet launched — full upstream evaluation +
  watch-list: `research.md`.
