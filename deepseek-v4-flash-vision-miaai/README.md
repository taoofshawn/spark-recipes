# deepseek-v4-flash-vision-miaai — native DeepSeek-V4-Flash-Vision-Exp

**Adoption** of [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
(vendored at upstream HEAD `d97c808`, 2026-09-01) into this repo's
docker-compose conventions. Native multimodal: a 32-layer ViT + Aligner with
OpenAI `image_url` support — **not** the caption-shim approach.

## What it serves

- **Model:** `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` (revision
  `86f746b3…` — weights identical to HEAD `e46e16bf…`, the cached snapshot on
  both nodes; later commits are README/eval only)
- **Served name:** `deepseek-v4-flash` (cluster convention; upstream serves
  `deepseek-v4-flash-vision-exp`)
- **Port:** 8000 | **Context:** 1M | **KV:** `nvfp4_ds_mla` (2,331,430-token
  pool @ GMU 0.83) | **DSpark:** k=6 | **Vision:** images only (no video —
  the official weights have no video encoder; GIF decodes as a still frame)

## Architecture (from upstream)

```
  client ──► :8000 vllm (Anemll dspark-vllm-gx10:0.1.1, TP=2, 2 nodes)
                 ▲ boot-time hotfix chain (./patches/, all read-only mounts)
```

- **Image:** `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` digest-pinned
  (`sha256:a83948…`), pulled — no local build.
- The command block applies the upstream hotfix chain at every boot: encoder
  copy + reasoning-effort mapping (issue #21), NVFP4-MLA + spin-wait + six
  kernel `.sh` hotfixes, then the `.py` chain (vision support, empty-encoder,
  partial-prefill, decode fairness, hybrid SWA, Triton specialization,
  suppress-stops, tool truncation, issue-191 fail-closed tool contract).
  `patches/` is vendored verbatim from upstream.
- Optional opt-ins (default off, matching upstream): GB10 hybrid NVFP4
  plugin (`ENABLE_VLLM_GB10_PATCH=1`), Responses-API compat, XGrammar
  termination, thinking-budget, assistant-final, issue-141 sparse-MLA chunk.
  Issue-191 fail-closed tool contract is the one opt-in that is ON by default
  here (see the 2026-09-06 changelog).

## Vision fix included: `bias_vl` routing (upstream issue #175 / PR #179)

The Vision-Exp checkpoint ships a vision-specific MoE router bias
(`.ffn.gate.bias_vl`) on 46 layers. Upstream bug: the parameter loaded but was
never read, so image tokens routed through the **text** MoE path in all 43
layers (and collapsed onto one fixed expert set in the 3 hash-MoE layers).
Fixed in PR #179 (`101e6f8`): image placeholder rows now call `fused_topk_bias`
with `bias_vl` and **no** hash table; text rows keep text bias + `tid2eid`.
CUDA-graph capture takes the stock text path.

**This recipe carries the fix** — it is baked into the vendored
`patches/hotfix-dsv4-vision-exp.py` (remaps `ffn.gate.bias_vl` →
`e_score_correction_bias_vl`, routes image rows with it, skips the hash table)
plus the `patches/vision_exp/` module, and the compose runs that hotfix at
boot unconditionally. Verified upstream after the fix: red JPEG → 117 image
tokens → "Red"; Earth-in-hands PNG → 365 tokens → accurate description.

## Upstream update pass (2026-09-06, vendored sync to HEAD `957890ac`)

- **Issue #191 fail-closed tool contract — ADOPTED, default ON.** Our profile
  is literally the issue's failing configuration (its env block: Vision-Exp
  `86f746b3`, MTP=6, `MAX_NUM_SEQS=6`, batched 8192, inflight 1,
  `nvfp4_ds_mla`, Anemll 0.1.1, `tool-call-parser deepseek_v4`): the gate
  campaign measured **3/145** concurrent strict `tool_choice` requests (c=4)
  returning HTTP 200 with a contract violation (named/required cardinality or
  argument schema), rooted in low-effort reasoning outrunning `max_tokens`
  (`finish_reason=length` → zero or a salvaged partial call). Upstream merged
  the patcher (#209 folded into #214): contract check on the head's serving
  layer → bounded re-generation (`DSPARK_ISSUE191_TOOLCALL_RETRIES`, default
  2) with thinking-off fallback on the last retry
  (`DSPARK_ISSUE191_TOOLCALL_THINKOFF_FALLBACK`, default 1) → HTTP 500 instead
  of a wrong 200 (`DSPARK_ISSUE191_TOOLCALL_MODE=log` available for a
  measurement pass). Author's live evidence on the same lane: 4× 145/145 with
  the hotfix on, 0 HTTP 500. Vendored `hotfix-vllm-issue191-toolcall-failclosed.py`
  and wired after the always-on issue-55 truncation hotfix (the patcher pins
  the post-issue55 `serving.py` bytes; ours runs unconditionally — verified).
  Takes effect on the next recreate. This is the one upstream opt-in we set to
  1: the alternative is silently contract-violating 200s on agent traffic.
- **Vision image-cap fix (upstream PR #231) — SYNCED.** Upstream's
  `vision_exp/processor.py` hardcoded `get_supported_mm_limits() →
  {"image": 16}`, so vLLM clamped every request to `min(16, --limit-mm-per-prompt)`
  (raising the CLI past 16 did nothing; 17+ images 400'd without the
  "Set --limit-mm-per-prompt" hint). It now returns `None` and the CLI governs.
  Our effective cap was already `min(16, image=10)` = 10 and remains 10; the
  sync just removes the hidden 16 ceiling. No config change.
- **Issue136 XGrammar chain (#210) — SYNCED** (still opt-in off). The patcher
  is now a two-file atomic transaction: the three vLLM #52805 hunks (issue
  #136) + the vLLM #53046 hunk (draft-window FSM validation at
  reasoning-end, issue #210), with `stock`/`patched`/`partial` chain states,
  rollback and `--status`. Still behind `DSPARK_ENABLE_ISSUE136_XGRAMMAR_HOTFIX=0`.
- **`NCCL_GIN_ENABLE` passthrough — ADDED** (unset default). Upstream measured
  (ENVS.md, 2026-09-05, on the 2-node lane): exact `0` (CPU-driven
  comm-init) drops NCCL comm-init from ~2 min to ~13 s at ~97% GPU memory
  pressure with no bandwidth change at serving message sizes. Bootstrap-speed
  knob; set it on a recreate if cold starts matter.
- **Reviewed and intentionally NOT adopted** (all upstream opt-ins, default 0,
  no measurement on our kit): DSpark block-k unlock (#215 — k=6 is the
  validated value on this image; the trained `dspark_block_size=5` path is
  unmeasured), DSpark SWA-prefix repopulation (#223 — deployed clean on
  0731/k=5 per #235 but unmeasured on Vision-Exp, and prefix-cache hits
  quantize to `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` boundaries), RoPE
  sparse-SWA fix (#222 — upstream itself requires a 128K+ long-context A/B
  before defaulting on), DSML recovery (#225), C128A prefill metadata cache
  (#233 — perf micro-opt, no A/B data), issue144 effort-directive prefix-cache
  alignment (#227 — fixes 0% hits under high/max; our default lane is low —
  note for clients that force `high`/`max` into long prefixed
  conversations), MXFP4 indexer K cache (#226 — N/A: our KV is
  `nvfp4_ds_mla`, and it needs the rejected DEEPGEMM_SM121_ALIAS companion).
  Forum cross-check (post #106, 09-05): an A/B of three fresh DSpark hotfixes
  found none of them moves prose decode.
- **Watch, no change:**
  - **#216** `cudaErrorNotPermitted` in `vision_exp/apply.py:124`
    (encoder/masked-write path), killing EngineCore after long uptime on the
    exact 0.1.1/`nvfp4_ds_mla`/MTP=6 stack — two independent reports (one on
    the non-abliterated `86f746b3` checkpoint with our exact flag set),
    maintainers could not repro; leading hypothesis: Docker's systemd cgroup
    driver + host `daemon-reload` revoking hook-injected GPU devices.
    **Checked this cluster (read-only, 09-06):** cgroup=systemd,
    nvidia-container-toolkit **1.20.0**; the running container's scope unit
    already lists `/dev/char/195:{0,254,255}` + `498:{0,1}` in
    `50-DeviceAllow.conf` — i.e. unit-level, survives reload; the exposed
    state is not present here. No compose change added; if it ever appears,
    the documented fix (NVIDIA doc option 2) is explicit `devices:` for
    `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`, `/dev/nvidia-uvm-tools`.
  - **#217** post-r3 live A/B (2× GB10 TP=2, `f5665e8`): cap 2 beat cap 1 —
    decode spread 1.66× vs 2.01× (8K) and 3.4–3.6× vs 5.9–6.0× (32K), TTFT
    spread 2.35× vs 4.09×, wave wall ~10% faster, slowest lane 10.5–12.7 vs
    9.3–9.8 tok/s. **But** #211's gate26 cap-1 number (1.60–1.76×) numerically
    equals #217's cap-2 result and the probe definitions aren't published —
    the two datasets are irreconcilable. Kept `DSPARK_MAX_INFLIGHT_PREFILLS=1`
    (upstream default); revisit if the gate26 probe is published.
  - **#237** agent reasoning infinite-repetition loops (Vision-Exp, VS Code
    agent with screenshot prompts) — open, no merged fix (PR #152 proposed a
    bounded loop-breaker; same family as #82). Suppress-stops is already
    active here.
  - Forum post #108 (09-06): a malformed image that 400s on this stack can
    take the engine down (not just a 400); our empty-encoder hotfix didn't
    cover that class; no upstream fix yet — proxy-side guard is the interim
    option users are using.
  - Cross-check alternative: tonyd2wild's own Vision-Exp stack
    (`DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark`, k=5,
    Patch 4/5 for a locally built image — not portable to Anemll 0.1.1, but
    its 2-node numbers are a sanity reference: code ~47 tok/s, prose ~38
    tok/s at 1M, needle-in-haystack to 500K).
- **No pin changes:** HF checkpoint unchanged (only `.eval_results/*.yaml`
  added at HEAD `6821d6ad`; weights identical to pinned `86f746b3`), image
  digest for `0.1.1` unchanged (`sha256:a83948…`; GHCR still only has
  0.1.0/0.1.1).

## Upstream update pass (2026-09-04, vendored sync to HEAD `bc2ef473`)

- **issue27 r3 (PR #211/#212)**: upstream walked the in-flight partial-prefill
  cap default back to **1** — the r1/r2 counting was buggy, so the earlier
  "2 is better" A/B measured a cap that never engaged. Synced
  `hotfix-dsv4-issue27-partial-prefill-concurrency.py` (r3 derives the count in
  exact parity with the stock scheduler set) and reverted
  `DSPARK_MAX_INFLIGHT_PREFILLS` 2→1 to match upstream's shipped default.
- **issue117 shm ring buffer, now default-on upstream**: stability backport
  (vLLM #45224 crash-hang class) for this exact image. Vendored
  `hotfix-vllm-issue117-shm-ring-buffer.py`, wired into the always-on boot
  chain (skip via `DSPARK_SKIP_ISSUE117_RECHECK_HOTFIX=1`).
- **Vision hotfix host-sync removal (issues #203/#204, merged)**:
  `vision_exp/apply.py` + `image_processor.py` now classify routing kind once
  per forward via a shared model-local cell instead of once per MoE layer
  (`.sum().item()` host sync × 43 layers gone). Decodes image workloads
  measurably faster; no config change.
- **issue138 responses-history patcher updated** (now also carries the opt-in
  Codex `agent_message` method-rewrite; still opt-in, default off — skip).
- Reviewed and deliberately NOT adopted (upstream opt-ins, default 0, no
  measurement on our kit): codex-agent-message, responses-store,
  adaptive-prefill-chunk, replicate-markov-head, sp-indexer,
  deepgemm-sm121-mqa-alias, TP3 launcher/padding (3-node only).

## References

- Forum: [DeepSeek v4 Flash Vision Exp is Released as Open Weights — post #55
  (fix announcement)](https://forums.developer.nvidia.com/t/deepseek-v4-flash-vision-exp-is-released-as-open-weights/381911/55)
- Upstream issue: [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark#175 —
  `bias_vl` loaded but never read](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/issues/175)
  → fixed in [PR #179](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/pull/179)
- Upstream [issue #191](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/issues/191)
  (strict tool contracts, our exact profile) → fail-closed hotfix adopted 2026-09-06.
- Alternate Vision-Exp stack (cross-reference, not portable to this image):
  [tonyd2wild/DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark)
- Upstream repo (README, docs/ENVS.md, docs/PATCHES.md, docs/SETUP.md carry the
  full constraint matrix — read them before tuning anything).

## Deploy

```bash
# 0) one-time on BOTH nodes: model snapshot cached (TP=2 reads on every rank)
hf download deepseek-ai/DeepSeek-V4-Flash-Vision-Exp --revision 86f746b36186f0e567729a5c06a8c918caba82a9

# 1) worker (node 1) FIRST, then leader ~35 s later — the :25000 TCP store
#    must be up before rank 1 connects.
#    worker: docker compose --env-file .env --env-file .env.node1 up -d
#    leader: docker compose --env-file .env --env-file .env.node0 up -d

# 2) verify
curl http://127.0.0.1:8000/v1/models     # -> "id":"deepseek-v4-flash", max_model_len 1048576
# vision smoke: image_url chat completion (see upstream smoke script)
```

## Cluster deviations from upstream

| knob | upstream | here | why |
|---|---|---|---|
| `PORT` / `VLLM_PORT` | 8888 | 8000 | cluster convention |
| `SERVED_MODEL_NAME` | deepseek-v4-flash-vision-exp | `deepseek-v4-flash` | user requirement — one name across recipes |
| `GPU_MEMORY_UTILIZATION` | 0.80 default | 0.83 | upstream README-measured value (ViT takes more weight RAM than 0731) |
| start order | `start-deepseek…sh` orchestrator | compose `.env.node0/1` | repo convention; worker first |
| JIT caches | on HF volume | node-local `/vllm-cache` | repo convention (issue #27 family) |

## GPU contention

Serves on all 2 GPUs per node. Tear down any other model container before
starting. One recipe at a time.
