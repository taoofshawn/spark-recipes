# deepseek-v4-flash-vision-0rand — native DeepSeek-V4-Flash-Vision-Exp (vLLM 0.28.1rc1 B12X)

**Adoption** of
[0rand/DeepSeek-v4-flash-ver-2sparks-vllm-029-0rand](https://github.com/0rand/DeepSeek-v4-flash-ver-2sparks-vllm-029-0rand)
(upstream @ `74e9f11`, 2026-09-09) into this repo's docker-compose
conventions. The upstream repo is a thin config-driven launcher; the runtime
is a **from-scratch vLLM build on the PilcoTHINK Dockerfile lane** with native
DSV4 vision + native DSpark + B12X baked in — no monkey-patch mods, no
encoder-file copies, no runtime hotfix chain (unlike `vision-miaai`).

The image is **Dickson's build of that lane** (`dicksondickson/vllm_spark_dsv4`
@ `38db013b`), which additionally bakes in the upstream **DeepSeek V4 tool-call
parser fix** (`vllm/parser/deepseek_v4.py` from vLLM `d98c8c03`, #55954) — the
measured delta behind Dickson's 93/100 vs the earlier 0rand image's 86/100
tool-eval-bench score (see changelog).

## What it serves

- **Model:** `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` (revision
  `86f746b3…` — 48 shards, byte-identical to HEAD `6821d6ad` and to the
  `e46e16bf` tree cached as `refs/main` on both nodes)
- **Served name:** `deepseek-v4-flash` (cluster convention)
- **Port:** 8000 | **Context:** 1M | **KV:** plain `fp8` | **DSpark:** k=6 |
  **Vision:** images only (no video encoder in the official weights)

## Stack

| Component | Pin |
|---|---|
| Image | `dicksondickson/vllm_spark_dsv4:0.29-b12x@sha256:899d174e…` (digest-pinned, pushed 2026-09-10) — Dickson's build of the PilcoTHINK `38db013b` lane; tag says 0.29, actual vLLM is `0.28.1rc1.dev475+g6fbb00b18` (`6fbb00b1`, native DSV4 vision + DSpark upstream) + the upstream `d98c8c03` tool-call parser fix baked in |
| vLLM | `6fbb00b1` (0.28.1rc1.dev475 + `.d20260907`) |
| PyTorch | 2.13.0+cu130 (CUDA 13.0.2 base) |
| FlashInfer | 0.6.18 @ `27d5b029` (DSV4 dual-cache dispatch C4+C128) |
| B12X | 1.3.0 @ `06b4de7c` |
| NCCL | SM121 source-built (`compute_121,code=sm_121`) |
| o_proj repair | AST-audited SM12x `(1,128,128)` + packed-scale 3D views, baked in |

Image provenance at runtime: `/workspace/build-metadata.yaml` inside the
container. Build instructions live upstream (PilcoTHINK
[`DGX_Spark_vllm_Dockerfile/0.28/DSV4F-Vision-exp`](https://github.com/gpdev-Pilcothink/DGX_Spark_vllm_Dockerfile/tree/main/0.28/DSV4F-Vision-exp))
— this recipe consumes the published image, no local build.

## Upstream's deliberate serving choices — do NOT regress these

1. **`--kv-cache-dtype fp8` (plain).** `fp8_ds_mla` / `nvfp4_ds_mla` need the
   patched gist stack; on this image they fail with `No valid attention
   backend`. (The "Using DeepSeek's fp8_ds_mla KV cache format" log line is
   normal — the FLASHINFER_MLA_SPARSE_DSV4 backend picks that format
   internally.)
2. **`--attention-backend FLASHINFER_MLA_SPARSE_DSV4`** (NOT `B12X`).
3. **`VLLM_USE_AOT_COMPILE=0` + `VLLM_USE_BREAKABLE_CUDAGRAPH=1`** — the
   image's preflight (`verify_deepseek_vision_dspark.py`) hard-requires both;
   compose pins them regardless of image defaults.
4. **`gpu_memory_utilization` ≤ 0.877.** GB10 reports ~106.77/121.69 GiB free
   at boot; 0.9 crashes (`Free memory ... less than desired GPU memory
   utilization`). **0.87** is tight-but-valid and the upstream shipped value;
   0.85 is the conservative fallback (verified valid).
5. **DSpark k must be divisible by `n_predict=3`.** The spec validator rejects
   k=5 (`must be divisible by n_predict=3`); k=3 passes the validator but was
   measured to degrade the thinking process (forum post #171: "illegal"
   forcing to 3 "brain-damages the thinking process"); k=6 is upstream's
   shipped value and the 93/100 tool-eval receipt config.
6. **`--skip-mm-profiling` + `--limit-mm-per-prompt`** — RAM guard at boot.
   We ship `image=10` (see deviations). Max 384 image tokens/image.
7. **`max_num_batched_tokens` 4096** — upstream's shipped best-settings value
   (~3M-token KV pool @ GMU 0.87, 93/100 tool-eval-bench receipt). Their perf
   table measured **2048** as acceptance optimum (~3.9M pool, +1.5× tg t/s at
   c8); **1024** gives the biggest pool (4M). All three are one `.env` line.
   vLLM warns `max_num_scheduled_tokens is set to 4096 based on speculative
   decoding` — expected.
8. **`max_num_seqs` 8** (their 16-seq config trades KV for throughput; 8 is
   the shipped default and matches the agent-serving profile).
9. **`--enabled` multi-node flags come from the launcher, not the recipe.**

## Our requirement pins (differs from upstream defaults)

Upstream serves `REASONING_EFFORT=max` with no explicit temp/top_p (vLLM
default 1.0/1.0 per this checkpoint's `generation_config.json`) and no
thinking default flag. This recipe pins the repo-wide sampling contract —
same values as the aiden and GLM recipes:

| knob | value | why |
|---|---|---|
| `temperature` | 1.0 | official eval config (model card: temp 1.0 / top_p 0.95) |
| `top_p` | 0.95 | official eval config; explicit pin so a checkpoint refresh can't drift it back to 1.0 |
| `thinking` | `true` | thinking ON by default (all-recipes parity) |
| `reasoning_effort` | `high` | agentic sweet spot (372268 #520: +7 passes for +5.5% wall time; max = no win + safety regression) |

Mechanism: `--generation-config vllm --override-generation-config
'{"temperature":1.0,"top_p":0.95}'` (both flags exist at this vLLM rev) +
`--default-chat-template-kwargs '{"thinking":true,"reasoning_effort":"high"}'`
(the native `deepseek_v4` tokenizer wrapper reads exactly
`thinking`/`reasoning_effort` from `chat_template_kwargs` — verified in
`vllm/tokenizers/deepseek_v4.py` at `6fbb00b1`). Clients still override per
request via `chat_template_kwargs` / the OpenAI `reasoning_effort` field.

## What the baked-in runtime does (from upstream + forum receipts)

- **Native vision, no proxy/shims.** `image_url` content parts work; upstream
  demoed a vision smoke with a dragon portrait. The vLLM tokenizer ships
  `merge_tool_messages()` (port of DeepSeek's reference function) — images in
  **tool results** are re-homed into the user message automatically, so
  agent harnesses can send images in `role:"tool"` (forum #177, verified by
  OllieOllie on this exact image; DeepSeek's formal contract is
  "images in USER messages only").
- **DSpark acceptance** ~0.70/0.49/0.28 per position, avg 44–76%; ~45-50 t/s
  1-seq, ~105-110 t/s 8-seq (upstream receipts 2026-09-08..09).
- **tool-eval-bench 93/100** (hardmode, 59 pass / 10 partial / 0 fail) at
  k=6, batch 4096, seqs 8, GMU 0.87, temp 1.0 — post #176/177 after the k=3
  regression was fixed.
- Boot ~10 min on their kit (weights 48/48 in ~29 s, then CUDA-graph capture).

## Cluster deviations from upstream

| knob | upstream | here | why |
|---|---|---|---|
| `PORT` | 8100 | 8000 | cluster convention |
| `SERVED_MODEL_NAME` | deepseek-v4-flash | `deepseek-v4-flash` | cluster convention (also matches name-proxy `BACKEND_MODEL`) |
| `HF_HOME` mount | `/root/.cache/huggingface` (eugr launcher) | `/cache/huggingface` + `HF_HOME=/cache/huggingface` | compose convention; same tree on both nodes |
| JIT caches | eugr launcher cache mounts | node-local `/vllm-cache` (VLLM_CACHE_ROOT + FlashInfer/TileLang/Triton/B12X dirs) | repo convention; both ranks write concurrently |
| `MAX_NUM_BATCHED_TOKENS` | 4096 | 4096 | same (upstream's shipped best) |
| `LIMIT_MM_PER_PROMPT` | 8 (README) / 3 (.env.sample — inconsistent upstream) | `image=10` | operator convention: 5-image snapcompact payloads + parity with GLM recipe's 10 |
| start order | eugr `start-cluster.sh` | compose worker-first (`.env.node0/1`) | repo convention |
| draft default | `probabilistic` | `probabilistic` | same |
| thinking/effort/temp/top_p | `max`, no pins | `true`/`high`/1.0/0.95 | user requirement — repo parity pins |
| API keys | none | `${VLLM_API_KEY:-}` absent — set in `.env` if needed (image supports `VLLM_API_KEY`/`DSPARK_API_KEYS`; not wired here to keep the diff minimal) | — |

Note: upstream requires **no** `--revision` (they pin the hub tree in
`HF_CACHE_DIR`); we pin `DSPARK_REVISION=86f746b3…` because the repo
convention is explicit revision + `--revision`. Both nodes already carry the
complete tree (48 shards, 0 `*.incomplete`, `refs/main` → `e46e16bf`).

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
# vision smoke: image_url chat completion (see upstream README)
```

The image is ~10.8 GiB and pulls from Docker Hub; first boot JITs B12X/CUTLASS
kernels into the node-local `/vllm-cache` volume (cold boot slower than the
warm one; upstream reports ~10 min).

## Optional: vision-grounding fix (`FIX_MM_PREFIX_SPAN`)

Upstream PR #1 (merged 2026-09-12) fixes a V2-model-runner bug where DSV4
vision mm-prefix bidirectional attention spans were derived from the
IMAGE-embed runs instead of the full sentinel block — a measured y-axis
grounding bias (ball-sweep y 416/501/501 → 250/350/750). The fix is vendored
verbatim under `mods/fix-dsv4-mm-prefix-span/` (Python-only patch of
`attn_utils.py` + `model_states/default.py`; idempotent, dry-run guarded) and
applied at container boot when enabled. **Default OFF** (upstream default
too):

1. set `FIX_MM_PREFIX_SPAN=1` in `.env`
2. recreate the container on BOTH nodes — worker first (start order)

If the patch cannot apply to a future image (context drift), boot fails
loudly rather than serving silently unpatched.

## References

- Upstream repo: [0rand/DeepSeek-v4-flash-ver-2sparks-vllm-029-0rand](https://github.com/0rand/DeepSeek-v4-flash-ver-2sparks-vllm-029-0rand)
  — README + `.env.sample` carry the measured knobs and the "do not regress"
  list; mirror that if upstream updates.
- Image build lane: [gpdev-Pilcothink/DGX_Spark_vllm_Dockerfile](https://github.com/gpdev-Pilcothink/DGX_Spark_vllm_Dockerfile)
  (0.28/DSV4F-Vision-exp; PILCOTHINK's tool-parser patch included, forum #180/#185).
- Image build (Dickson): [dicksondickson/vllm_spark_dsv4](https://hub.docker.com/r/dicksondickson/vllm_spark_dsv4)
  — Dickson's build of the `38db013b` lane, bakes in the upstream `d98c8c03`
  tool-call parser fix (forum #194).
- Forum: [DeepSeek v4 Flash Vision Exp is Released as Open Weights](https://forums.developer.nvidia.com/t/deepseek-v4-flash-vision-exp-is-released-as-open-weights/381911)
  — 0rand's posts from 09-08/09-09: image push (#151), k=3 thinking regression
  (#171), k=6 + 93/100 receipt (#173/#176), tool-result image re-homing (#177);
  parser-fix provenance (#185), Dickson image switch (#194/#214), locked
  benchmark comparison (#212).
- Sibling recipe: [`../deepseek-v4-flash-vision-miaai`](../deepseek-v4-flash-vision-miaai)
  (Anemll 0.1.1 + NVFP4-MLA + hotfix chain — the quality-reference lane;
  this 0rand lane is the speed/context lane with identical weights).
- Model card: [deepseek-ai/DeepSeek-V4-Flash-Vision-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp)
  (eval config: `max` effort, temp 1.0 / top_p 0.95 — pins above follow it,
  effort differs per operator requirement).

## GPU contention

Serves on all 2 GPUs per node. Tear down any other model container before
starting. One recipe at a time.

## Ops notes (running hazards — forum receipts)

- **Prefill headroom:** this image is stable to ~850k-token prefill
  (forum 381911 #246/#250, 2026-09-11). The historical >65k-prefill OOM
  was load-time RAM (tensor unpack + cudagraph profiling, #249) — if a
  boot OOMs, prefer safe-tensors weights before blaming GMU.
- **Client max-output-tokens footgun (reproduced, #234):** a client env
  `COPILOT_PROVIDER_MAX_OUTPUT_TOKENS=131072` causes thinking loops on
  DS4-Vision-Exp specifically; 32768 clears it.
- **GB10 clock latch (thread 382897):** decode collapsing to a few tok/s
  after days of uptime can be GPUs latched <1 GHz; recovery requires
  UNPLUGGING the Spark — a reboot does not clear it.
- **GB10 UMA has no cgroup accounting/PSI/OOM events (thread 383003):**
  memory death can present as a silent freeze/power-off; `blackbox` /
  `sparkview` exist for post-mortems.

Dated changelog and update-pass findings live in [`research.md`](research.md)
(AGENTS.md convention: the README is the active-running doc only).
