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
- **Port:** 4000 | **Context:** 1M | **KV:** `nvfp4_ds_mla` (2,331,430-token
  pool @ GMU 0.83) | **DSpark:** k=6 | **Vision:** images only (no video —
  the official weights have no video encoder; GIF decodes as a still frame)

## Architecture (from upstream)

```
  client ──► :4000 vllm (Anemll dspark-vllm-gx10:0.1.1, TP=2, 2 nodes)
                 ▲ boot-time hotfix chain (./patches/, all read-only mounts)
```

- **Image:** `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` digest-pinned
  (`sha256:a83948…`), pulled — no local build.
- The command block applies the upstream hotfix chain at every boot: encoder
  copy + reasoning-effort mapping (issue #21), NVFP4-MLA + spin-wait + six
  kernel `.sh` hotfixes, then the `.py` chain (vision support, empty-encoder,
  partial-prefill, decode fairness, hybrid SWA, Triton specialization,
  suppress-stops, tool truncation). `patches/` is vendored verbatim from
  upstream.
- Optional opt-ins (all default off, matching upstream): GB10 hybrid NVFP4
  plugin (`ENABLE_VLLM_GB10_PATCH=1`), Responses-API compat, XGrammar
  termination, thinking-budget, assistant-final, issue-141 sparse-MLA chunk.

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
curl http://127.0.0.1:4000/v1/models     # -> "id":"deepseek-v4-flash", max_model_len 1048576
# vision smoke: image_url chat completion (see upstream smoke script)
```

## Cluster deviations from upstream

| knob | upstream | here | why |
|---|---|---|---|
| `PORT` / `VLLM_PORT` | 8888 | 4000 | cluster convention |
| `SERVED_MODEL_NAME` | deepseek-v4-flash-vision-exp | `deepseek-v4-flash` | user requirement — one name across recipes |
| `GPU_MEMORY_UTILIZATION` | 0.80 default | 0.83 | upstream README-measured value (ViT takes more weight RAM than 0731) |
| start order | `start-deepseek…sh` orchestrator | compose `.env.node0/1` | repo convention; worker first |
| JIT caches | on HF volume | node-local `/vllm-cache` | repo convention (issue #27 family) |

## GPU contention

Serves on all 2 GPUs per node. Tear down any other model container before
starting. One recipe at a time.
