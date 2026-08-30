# glm-v53-flash - GLM-5.3-Flash NVFP4 (2× DGX Spark, vLLM)

> **Status: tested on this cluster (2026-08-28).** Boot markers, KV-pool size,
> and tok/s below are measured here, not just upstream. See the audit trail for
> this deployment's numbers and the one recipe fix it required. Research,
> upstream evaluation, and the "where to watch for updates" list live in
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
| Image | locally built `glm53-nvfp4-sm121:local` (10 patch layers, see `patches/`) |
| Checkpoint | `RedHatAI/GLM-5.3-Flash-NVFP4` (184 GiB compressed-tensors W4A4; revision pinned — switched from LibertAIDAI 2026-08-30, see audit trail) |
| KV cache | `fp8_e4m3` (patched NoPE-MLA path) |
| KV pool | 507K tokens @ 4.14 GiB/rank (default, 3/3 reliable) · 672K @ 5.5 GiB (local-weights record) |
| Spec decode | MTP 3 default · **DFlash2 opt-in** (`DFLASH2=1`, 2.15x upstream — see below; non-commercial drafter license) |
| DFlash2 KV pool | 581,040 tokens @ 262K context, profiler-sized (upstream 2026-08-28 correction; see DFlash2 section) |
| `gpu_memory_utilization` | 0.85 (0.78/0.80 starve the bf16 KV at 131K+) |
| `max_num_seqs` | 6 · `block-size` 2304 (kpool page invariant) · `--moe-backend marlin` |
| Context | 262,144 (TP2 ceiling — the model-native 1M needs TP4 / 4 nodes) |
| Thinking | **on, `high`** server-side (all-recipes parity — and now a real toggle via `THINKING`, see Tool-calling notes) |
| Serve | port `4000`, served model `glm-5.3-flash` |

Measured on THIS cluster: **23–28 tok/s** single-stream, ~51–62 tok/s
aggregate at C6 (first boot, cold JIT — see audit trail). Upstream DFlash2
references: 46.9 tok/s single-stream at 74.1% acceptance (2.15x MTP), 56.2
tok/s aggregate @ C5, zero failures.

---

## Prep (both nodes)

```bash
cd ~/code/spark-recipes/glm-v53-flash/
docker compose --env-file .env --env-file .env.node0 config -q   # sanity on each node

# Download the checkpoint (184 GiB) on BOTH nodes — serving is offline:
hf download RedHatAI/GLM-5.3-Flash-NVFP4 --revision 36c184c6cda000a481711306df5adde42f63321a
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
  the template's thinking block into the `reasoning` field (note: this
  stack exposes it as `reasoning`, not `reasoning_content`).
- **Thinking is now a real toggle (was structurally always-on).** The
  checkpoint's shipped `chat_template.jinja` has **no `enable_thinking`
  gate**: the generation prompt always opens a think block and an effort
  header is always emitted (`reasoning_effort` undefined → **max**). Patch
  layer 10 vendors the upstream-fixed template (commit `53912b4`) and wires
  it via `--chat-template`, so `THINKING=true|false` (default true — the
  pre-fix behavior) and per-request
  `"chat_template_kwargs": {"enable_thinking": false}` now work. Requests
  without any effort param were observed to **degenerate** (800 tokens,
  empty `reasoning` and empty `content`), so the recipe always pins an
  effort server-side regardless.
- Sampling defaults (temp 1.0, top_p 0.95) are now **explicit recipe
  flags**: `--generation-config vllm` + `--override-generation-config
  '{"temperature":1.0,"top_p":0.95}'` (env `TEMPERATURE`/`TOP_P`). The
  RedHatAI checkpoint's `generation_config.json` ships ONLY temperature, so
  without these top_p drifted to vLLM's 1.0 (the old LibertAIDAI checkpoint
  shipped 0.95). `--generation-config vllm` affects sampling only — the
  multi-EOS stop set still loads from the model config. Values match the
  DeepSeek recipes. Note the sampling kwargs are also accepted per-request
  via the OpenAI `temperature`/`top_p` fields, which override these.
- `max_tokens` includes reasoning tokens when thinking is on (a short
  answer with `high` effort cost ~330 completion tokens vs 41 without).
  Keep it generous for agentic use.
- **Thinking off = +8% draft acceptance, but leaks reasoning prose.**
  Upstream measured `enable_thinking: false` as faster (reasoning traces
  draft worse), and it is now a real toggle — but with thinking off GLM
  emits untagged reasoning-prose into `content`, which some agent
  harnesses mis-parse. One more reason the default stays on.

## DFlash2 fast drafting (opt-in, `DFLASH2=1`)

The upstream repo added **DFlash2** — inco.ai's block-diffusion drafter
(vLLM PR #52816 port) — and it is **proven at TP2 on our exact lane** (fp8 KV,
marlin, block-size 2304): **46.9 tok/s single-stream at 74.1% acceptance
(2.15x MTP-4)**, 56.2 tok/s aggregate @ C5 with zero failures, and the
drafter costs zero KV pool (it slot-shares the MLA tensors). MTP stays the
default because the drafter checkpoint is licensed
**CC-BY-NC-ND-4.0 — non-commercial use only** (the base model is MIT).

```bash
# Both nodes:
hf download incoai/GLM-5.3-Flash-DFlash2 --revision 7d74cdd881ed7e32c31175984a67823127b66cfe   # 2.34 GB

# Then in .env (or per-node env): DFLASH2=1
# SPEC_TOKENS must be 7 (drafter block 8 - 1) — the compose refuses anything else.
# KV pin drops to 3221225472 (3 GiB, upstream's shipped DFlash2 value) unless
# you set KV_CACHE_MEMORY. Do NOT raise it to chase KV headroom: upstream
# withdrew its 7 GiB "ceiling" (2026-08-28) — pinned pools skip the activation
# reservation and die on the first long prompt. Profiler-sized is 581,040
# tokens @ 262K. If you must pin higher, validate with a >=28K-token prompt.
```

Boot signatures to confirm: `Using Eagle3 auxiliary layers from config:
(6, 15, 25, 34, 43)`, `Warming up spec-decode rejection sampler kernels
(vocab=154880, num_spec=7, ...)`, and `/metrics` acceptance in the 0.6-0.8
range (acceptance ~0.15 = broken aux capture — do not serve). First inference
JIT-compiles drafter kernels (~10 tok/s) — measure warm.

Cautions (upstream issue #7, still open): a second 2x GB10 pair reports only
28-31 tok/s at 0.35 acceptance with an overlay built FROM v9/InstantTensor —
our stack is FROM the stable v8-equivalent, but benchmark before trusting
46.9 as guaranteed. Do not combine with `--load-format instanttensor`.

### Upstream open problems (2026-08-28, docs/OPEN-PROBLEMS.md) — affects us

- **KV pin trap:** `--kv-cache-memory` makes vLLM skip subtracting the measured
  activation peak — allocates, warms, answers short prompts, then **dies on the
  first long request**. Reproduced at 7.5 GiB, 300K ctx, 700K ctx, and 12 GiB.
  Upstream withdrew its 7 GiB ceiling figure rather than restate it.
- **TP worker rank profiles 4-5 GiB less KV headroom than the head** (min across
  ranks binds the pool) — explains why pool figures drift between boots; not a
  config error, looks like an upstream vLLM issue.
- **`--load-format instanttensor`** is fast (~40-100 s loads) but silently
  unstable multi-node: a rank dies ~1 min post-load in 4/4 upstream TP2 boots.
  We already avoid it.
- **UVM livelock:** with swap active, the kernel can page vLLM out mid-load →
  unrecoverable spin (`UVM GPU` kthread hot, 96% GPU util at ~10 W). Upstream
  requires `vm.swappiness=0` on every node (does not survive reboot; put it in
  `/etc/sysctl.d/`) and a `swapoff -a && swapon -a` before launch.
- **vision is not speculated:** image requests work but get no DFlash2 speedup
  (drafter takes text-only draft inputs).
- **CUDA graphs trap (vllm#53030):** piecewise-graph `BatchDescriptor` collision
  silently pins acceptance at exactly 1.00 — check
  `vllm:spec_decode_num_accepted_tokens_per_pos_total` after any graph-enabled
  boot. We run `--enforce-eager`, unaffected.

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
  Dockerfile            # FROM vllm/vllm-openai:glm53-flash-arm64-cu130 (digest-pinned) + 10 patch layers
  patch_v7.py           # indexer top-k init + pool clamp                          [tonyd2wild]
  patch_v8_fp8.py       # fp8 KV cache for the NoPE-MLA path on SM12x             [tonyd2wild]
  sparse_attn_indexer_kpool.py  # persistent_topk SM121 gate (24K-context crash)  [tonyd2wild]
  chat_template_mm.jinja# fixed template honoring enable_thinking (53912b4)       [tonyd2wild]
  overlay-dflash2/      # DFlash2 drafter port (vLLM PR #52816 + GLM glue), inert unless DFLASH2=1 [tonyd2wild]
tools/
  cache_flusher.sh      # host-side GB10 page-cache guard during load             [tonyd2wild]
  fleet_watchdog.sh     # /health-probing watchdog; orchestrated 2-node relaunch  [tonyd2wild, adapted]
  load_test_glm.py      # 6-way concurrent tool-carrying load test (degenerate-loop check)
README.md               # this file (run/build)
research.md             # research notes + future-work handoff for agents
```

---

# Audit trail

- **2026-08-30 — checkpoint switched: `RedHatAI/GLM-5.3-Flash-NVFP4` (W4A4
  compressed-tensors) replaces `LibertAIDAI/GLM-5.3-Flash-NVFP4` (ModelOpt
  weight-only).** Upstream made the same switch (2026-08-29/30, commits
  `7497e96b`/`5a4df199`) after ajclark reported in upstream issue #10 that
  ModelOpt NVFP4 builds emit **intermittent corrupted token IDs** — silent
  until a corrupted token lands inside a tool-call block and desyncs the
  parser — confirmed on 2× GB10 / SM121 TP2 DFlash2, i.e. our exact lane;
  RedHatAI of the same model is clean (vllm#54150, originally SM120). This
  supersedes the 2026-08-28 "keep LibertAIDAI" quant verdict, which predates
  the corruption evidence. Swap is drop-in: same flags (`--moe-backend marlin`,
  fp8 KV, DFlash2 k=7), rev pinned `36c184c6`. Trade-off: activations also
  quantized to FP4 (W4A4) — expect a few points lower on hard reasoning — and
  RedHatAI ships `chat_template.jinja` but NOT `chat_template_mm.jinja`; our
  `TEMPLATE_FIX` layer already supplies the vision template via
  `--chat-template`. Bonus: ~2× faster load (11 shards vs 120). **NOT yet
  measured here** — on the next bounce: verify boot + `/v1/models`, confirm
  `reasoning_content` split (the standing `glm45` vs `deepseek_r1` VERIFY
  item), and rerun a tool-call stress test (`tools/load_test_glm.py`) to
  confirm the corruption class is gone.
- **2026-08-28 — upstream KV-ceiling correction adopted (drop 7 GiB pin).**
  Upstream withdrew its published 7 GiB / 727,583-token DFlash2 ceiling
  (commit 53853387): pinned `--kv-cache-memory` pools skip the measured
  activation-headroom subtraction and die on the first LONG prompt (verified
  upstream at 7.5 GiB, 300K ctx, 700K ctx, 12 GiB). Deep-dive profiler-sized
  figures at 262K context: 581,040 tokens with DFlash2 (verified through a
  28,818-token prompt) / 965,166 without a drafter. Our DFLASH2=1 pin
  reverted from 7516192768 (7 GiB) to upstream's shipped 3221225472 (3 GiB);
  MTP default 4445787956 unchanged. Also adopted `docs/OPEN-PROBLEMS.md`
  findings (rank-1 KV asymmetry, InstantTensor instability, UVM livelock /
  `vm.swappiness=0`, vision-not-speculated, CUDA-graph acceptance trap) into
  the DFlash2 section — see "Upstream open problems".
- **2026-08-28 — upstream sync: DFlash2 (opt-in) + enable_thinking template
  fix.** Adopted from tonyd2wild's repo update (overlay-dflash2, commits
  64c92e9/3238536/53912b4/a5c4b19): (1) patch layers 9-10 — the DFlash2
  drafter port (inert unless `DFLASH2=1`; registry + Eagle3 aux-capture +
  GLM5-Next drafter KV group that slot-shares MLA tensors) and the fixed
  chat template wired via `--chat-template` (`TEMPLATE_FIX=1`), which makes
  `THINKING` a real toggle (default true = pre-fix behavior, all-recipes
  parity); (2) `tools/fleet_watchdog.sh` adapted for this 2-node compose
  layout (probes `/health`, NOT `/v1/models` — the latter returns 200 with a
  dead engine; orchestrates worker-first relaunch); (3) verified our
  `sparse_attn_indexer_kpool.py` gate matches upstream `a5c4b19`
  (multi_processor_count >= 78 -> persistent_topk, else top_k_per_row_decode)
  — no change needed. DFlash2 numbers (46.9 tok/s C1, 74.1% acc, 2.15x) are
  upstream measurements; not yet measured here. Drafter license
  CC-BY-NC-ND-4.0 keeps it opt-in (MTP3 default unchanged). Issue #7
  (reproduction gap with a v9-based overlay) noted in the DFlash2 section.
- **2026-08-28 — first successful launch on this cluster (2× GB10, TP2).**
  Boot markers verified on head + worker: `MARLIN` NvFp4 MoE backend engaged
  (no FP4-corruption silent loop), `fp8_e4m3` KV cache, `Initial free memory
  110.73 GiB, reserved 4.14 GiB for KV Cache` (KV slab pinned, no phantom
  backing death), `GPU KV cache size: 527,879 tokens` (≈507K expected),
  `NCCL 2.30.7` + cutlass-dsl 4.6.2 re-pins held, MTP3 spec decode active,
  no `pe_dim=64` assert, `/v1/models` → `glm-5.3-flash`, `max_model_len
  262144`, `/health` 200. Measured (first boot, cold JIT): weight load
  181.29 GiB in ~795 s, engine init (profile/KV/warmup) ~150 s; 6-way
  concurrent load test **PASS** (all streams clean, tool calls working;
  single-stream decode ~23–28 tok/s, aggregate ~51–62 tok/s at C6) — matches
  upstream 21.8–28.3 tok/s. **Recipe fix required before this boot:** patch
  layer 1 in `patches/Dockerfile` had malformed Python (single-quote
  multiline strings → `SyntaxError` at `cuda.py` step 2); rewritten to
  triple-quoted strings, content unchanged (commit 5fbd006). A fresh warm boot
  is expected to be faster than these cold-start numbers.
- **2026-08-28 — thinking defaults fixed (all-recipes parity).** Found while
  verifying the HF commit history: our pin (`aa28e1f5`) already contains the
  upstream chat-template sync and parser notes (verified: snapshot files
  byte-identical to HF HEAD). But the template has **no `enable_thinking`
  gate** — the original `THINKING=false` default-chat-template-kwargs was a
  no-op, and no-effort requests degenerate (800 tokens, empty output).
  Changed the server default to `--default-chat-template-kwargs
  '{"reasoning_effort": "high"}'` (env `REASONING_EFFORT`, default `high`);
  temp 1.0 / top_p 0.95 already flow from the checkpoint's
  `generation_config.json` (confirmed in the boot log). omp client
  registration updated (`reasoning: true`, effort levels low/high/max,
  default high). Applied on the NEXT container restart (not yet bounced at
   the time of writing).
- **2026-08-30 — sampling made explicit (RedHatAI switch follow-up).** The
  RedHatAI checkpoint's `generation_config.json` ships ONLY `temperature:
  1.0` (no `top_p`), so top_p silently drifted from the old LibertAIDAI
  checkpoint's 0.95 to vLLM's 1.0. Made all four inference defaults explicit
  recipe env: `THINKING=true`, `REASONING_EFFORT=high`, `TEMPERATURE=1.0`,
  `TOP_P=0.95`, wired via `--generation-config vllm` +
  `--override-generation-config '{"temperature":1.0,"top_p":0.95}'`
  (sampling only — multi-EOS stop set still loads from the model config,
  verified in vLLM source: `try_get_generation_config` reads the file in
  both `auto` and `vllm` modes). Matches the other recipes' values.
