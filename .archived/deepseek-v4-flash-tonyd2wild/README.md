# deepseek-v4-flash-tonyd2wild - 1M token context (NVFP4 DS-MLA KV)

A self-contained recipe for running DeepSeek-V4-Flash-0731 on a 2-node DGX Spark
cluster, built from [tonyd2wild's DSpark stack](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark).

## Configuration overview

| knob | value |
|---|---|
| Image | locally built `vllm-dspark-runtime:dspark-nvfp4-stage-c` (4-stage overlay on `ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready`) |
| Model runner | v1 (`--distributed-executor-backend mp`) |
| KV cache | `nvfp4_ds_mla` |
| DSpark spec tokens | 5 (`MTP_NUM_TOKENS=5`) |
| `max_num_seqs` | 6 (measured-best for 1M) |
| `max_num_batched_tokens` | 8192 |
| `max_cudagraph_capture_size` | `seqs×(k+1)` = 36 |
| `gpu_memory_utilization` | 0.78 (0.80 "boots-then-dies" on this stack) |
| sampling | `--generation-config vllm`, no override |
| tokenizer | `deepseek_v4` mode with `fastokens` shim (`VLLM_USE_FASTOKENS=1`) |
| thinking default | `true` (server `reasoning_effort=high`; clients can override per request) |
| context length | 1M (1048576) |
| serve | port `4000`, served model `deepseek-v4-flash` |

> ⚠️ Keep this recipe's native backend wiring — `nvfp4_ds_mla` KV with the v1
> runner via `--distributed-executor-backend mp`, and B12X MoE via
> `VLLM_USE_B12X_MOE=1`. These flags belong to this specific image; mixing in
> flags from other recipes/images fails at startup.

---

## 2026-08-30 — Patch A + fused-Markov (campaign-2026-08-20 STACK winner)

Adopted the upstream DSpark tuning campaign's final config:
`VLLM_DSPARK_DRAFT_CAPTURE_SIZES=1` (Patch A) + `VLLM_DSPARK_FUSED_MARKOV_ARGMAX=1`
at `gpu_memory_utilization 0.78` / `max_num_seqs 6`.

- **Patch A** is a genuine fork bug fix, not a knob: the DSpark proposer shared
  the target model's cudagraph capture sizes, which under spec-decode round to
  multiples of `1+k` (k=5 → 6/12/18), so a batch-1 draft dispatched on the
  6-bucket, clipped to the 4 draft rows, and the draft MoE processed **20 draft
  tokens/step for a single stream instead of 5**. Patch A installs a
  drafter-private capture-size view `{1,2,4}` (env-gated, default off = byte
  identical to stock). **+3.0% single-stream decode** (chat +6.4%, code +6.5%),
  every stable quality probe byte-identical (6/6 MATCH, garble 30/30).
- **Fused-Markov argmax** (`VLLM_DSPARK_FUSED_MARKOV_ARGMAX=1`): one Triton
  kernel per draft position instead of three. Neutral single-stream, **+5-9% c4
  aggregate**. Quality intact.
- **KV/memory accounting** (campaign): baseline 0.76 = 1.32M tokens; STACK
  (Patch A + fused-Markov) at 0.78 = **1.38M tokens**; 0.80 alone = 1.79M but is
  the flagged physical edge (driver OOM retries at profiling; this recipe's README
  historically said 0.80 "boots-then-dies"). Patch A costs ~2.5 GB + fused-Markov
  ~1.5 GB at profiling, which the 0.76→0.78 bump absorbs. **Keep 0.78.**
- This also reconciles the compose back to the README-documented lane:
  previous compose carried the unverified C12 profile (GPU_MEM 0.83 /
  `max_num_seqs 12`, marked "verify on this cluster"). Reverting to 0.78/6 for a
  validated bring-up. Re-enable C12 only after a gated run on this cluster.
- **Apply doc** (upstream): bind-mount `patches/A-drafter-sizes/v1/spec_decode/
  dspark_proposer.py` → `/opt/env/lib/python3.12/site-packages/vllm/v1/spec_decode/
  dspark_proposer.py:ro` on BOTH nodes; set both env toggles identically across
  ranks (mismatch → NCCL hang). Vendored copy is byte-identical to upstream
  (1152 lines).
- Not adopted from the campaign: Patch B (real draft probs for T>0) — the
  full-vocab gathers cost what the acceptance gain earns (throughput flat);
  E4 defer-target-capture (no gain); dual-HCA (interconnect provably not the
  bottleneck); k=4/k=3 (strictly lose to k=5).

## 2026-08-23 — fastokens shim (VLLM_USE_FASTOKENS=1)

Adopted the crusoecloud `fastokens` Rust BPE backend (10x+ faster tokenization,
TTFT win on long prompts) behind the existing-but-unconsumed
`VLLM_USE_FASTOKENS` env var that the vendored `envs.py` already defines.

- **Why a runtime hook:** this image is a vLLM 0.21.x fork, and 0.21 only ships
  `--tokenizer-mode fastokens` — which would REPLACE the `deepseek_v4` mode and
  drop the reasoning-effort / tool-arg overlays this recipe depends on. vLLM's
  env-var path (`VLLM_USE_FASTOKENS=1` → `fastokens.patch_transformers()`,
  keeping `deepseek_v4` intact) only exists on vLLM main. So the recipe applies
  the same process-global, idempotent patch at interpreter start via a
  bind-mounted `sitecustomize.py` (`./sitecustomize.py` → `/opt/env/.../site-packages/`).
  The existing `detokenizer.py` overlay already looks up
  `tokenizers.decoders.DecodeStream` on the module, so the shim's DecodeStream
  rebind is honored without changes.
- **Package install:** `fastokens>=0.2.0` is pip-installed at boot when the env
  var is on (kept out of the vendored upstream Dockerfile; hard-fails if the
  install fails so the shim is never silently skipped). Opt out:
  `VLLM_USE_FASTOKENS=0`. Package verified to have `manylinux_2_28_aarch64`
  wheels (works on GB10).
- **Validation:** `fastokens.patch_transformers()` smoke-tested against
  transformers 4.57.6 (the version range in this image). Runtime verification
  on the cluster: boot log shows `[fastokens] ...` and no
  `Error in sitecustomize`; tokenize/TTFT delta at large context not yet
  measured — benchmark before/after with the repo's usual curl/jq harness.

## 2026-08-23 (later) — fastokens hook refinement + inert env-var cleanup

Follow-ups from the first cluster boot with `VLLM_USE_FASTOKENS=1`:

- **`sitecustomize.py` now warns instead of raising.** The strict ImportError
  printed `Error in sitecustomize` once per first boot — the compose boot
  script's "is fastokens installed?" probe runs before the package exists.
  `sitecustomize` runs in every python process, so it is the wrong layer for a
  fatal guard; the authoritative guard stays in the compose boot script (it
  installs `fastokens` and hard-fails the boot if the install fails). A
  missing/too-old package now logs a greppable `[fastokens] WARNING` and the
  process continues without the shim.
- **Dropped `VLLM_TRITON_MLA_SPARSE` and `VLLM_SKIP_INIT_MEMORY_CHECK` from the
  compose env.** Both logged `Unknown vLLM environment variable` at boot and
  have no references anywhere in this image's vLLM code — they were inert
  leftovers from an older/newer upstream config. (They remain in the vendored
  `upstream/` files, which are untouched.)

## 2026-08-21 — default thinking on + reasoning_effort=high

Server-side defaults now match the other DeepSeek recipes: `THINKING=true` and a
new `REASONING_EFFORT=high` knob (passed as
`--default-chat-template-kwargs '{"thinking":true,"reasoning_effort":"high"}'`).
Clients can still override per request via `chat_template_kwargs`. aiden and
aiden-sparkrun already shipped thinking=true + reasoning_effort=high; this
closes the last gap.

## 2026-08-20 (later) — tool-arg normalize: narrow the auto-unwrap (fix false positive)

`normalize_tool_arguments` in `encoding_dsv4.py` used to unwrap a wrapper key
whenever it was present, e.g. `{"endpoint":"/x","parameters":{"page":2}}` became
`{"endpoint":"/x","page":2}` — corrupting calls where `parameters`/`input`/
`arguments` is a **legitimate nested parameter**, not a malformed wrapper.

Now the unwrap only happens when the wrapper key is the **only** real top-level
key (besides spurious metadata such as `name`/`type`/`id`). Common single-wrapper
repair still works: `{"arguments":{"city":"Paris"}}` → `{"city":"Paris"}`, and
truncated-JSON repair is unchanged. Applied to all three copies of
`encoding_dsv4.py` (tonyd2wild, aiden, aiden-sparkrun) to keep them in sync.

## 2026-08-20 — upstream review: JIT-cache hardening + engine-ready timeout

Re-reviewed `tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark`
against the vendored pin (`d728faee`, 2026-07-31). It has 13 new commits
(PRs #14/#17/#19/#21/#24/#28 and issue sweeps). Most were already backported
into this recipe (Patch 5 stop-suppression, PR #17 tool parser); the
tokenizer/encoder rewrite in PR #24 was **not** adopted because it drops the
tool-argument repair functions this recipe deliberately carries. Two fixes
from upstream were worth adopting here:

1. **Node-local JIT/compile caches (upstream issue #27, adopted).** This recipe
   previously put `VLLM_CACHE_ROOT` inside the HF cache tree
   (`/cache/huggingface/vllm-cache`). If the HF cache is ever mounted on a
   shared/NFS path, seven concurrently-written caches (vLLM compile, DeepGEMM,
   FlashInfer workspace, TileLang, TorchInductor, Triton, torch extensions)
   corrupt one another in confusing ways — including an ABI-mismatched
   FlashInfer `sampling.so` loaded silently because
   `FLASHINFER_DISABLE_VERSION_CHECK=1`. The compose now mounts a dedicated
   node-local `/vllm-cache` volume (host `JIT_CACHE_DIR`, default
   `~/.cache/vllm-dspark`) and points all seven caches at it.
2. **Engine-ready timeout 600→3600 s (upstream PR #19, adopted).** The stock
   600 s timeout can trip `torch.distributed` teardown on a warm restart of
   this 155 GiB model, killing the second startup with no useful error. Now
   `VLLM_ENGINE_READY_TIMEOUT_S=3600` matches the sparkrun profile.

Not adopted (documented so future reviews don't re-litigate): PR #24's
`deepseek_v4_encoding.py` rewrite (drops tool-arg repair), GLOO/TP socket
ifname fallback (this recipe already sets them from `.env`), `LOCAL_MODELS_DIR`
(we serve from the HF cache), KV-dtype override and server-side chat-template
kwargs (already covered by this recipe's knobs), and the upstream sparkrun
recipe (this recipe is compose-based).

1. **Reasoning-effort + tool-argument hardening overlay** - `encoding_dsv4.py` and
   `deepseek_v4_wrapper.py` bind-mounted to
   `../vllm/tokenizers/deepseek_v4_encoding.py` and `deepseek_v4.py`.
   - This runtime's stock encoder only accepts `high`/`max` reasoning_effort (`low`
     asserts; forum 372268 P510). These overlays restore the official 3-level
     (low/high/max) `REASONING_EFFORT_PROMPTS` + routing, so effort levels behave
     like the aiden recipe.
   - The same files add tool-argument JSON repair/normalization
     (`normalize_tool_arguments`, `repair_tool_arguments_json`, `parse_tool_arguments`,
     `dsml_param_to_python`, `normalize_parsed_dsml_tool_args`,
     `prepare_openai_tool_call_for_execution`) so wrapper-key / truncated / malformed
     tool args survive as valid JSON. Sourced from the aiden encoder fix
     (`aiden-encoder-toolarg-fix`).
2. **DSpark draft MoE backend** - added `"moe_backend":"b12x"` inside the
   `speculative-config` (forum 378824 P12, srivatsa1): the NVFP4 draft must run its
   MoE on `b12x` (not the default `flashinfer_b12x`) or the MXFP4 oracle rejects
   drafts and acceptance collapses (~1.0-1.15 tok/step). This is the flag part of
   that fix (measured 16.9->64.4 tok/s). The companion source-level
   draft-quantization / fail-closed-loader patches from that thread are **not**
   applied here - a possible follow-up if acceptance still sags.
3. GitHub audit: main repo has 4 commits since the vendored pin - all docs/tooling
   except a `sparkrun/` deploy harness that reproduces our exact config. No new
   serving patches to adopt. Other `tonyd2wild` repos use different profiles
   (FP8 / MTP-lane / other-context) and contradict these NVFP4-1M-DSpark
   invariants.

---

# Building & deploying from scratch on new DGX Sparks

Everything below is what it takes to go from a pair of fresh GB10 Sparks to a
serving `deepseek-v4-flash` on :4000. It reflects a real deployment; the gotchas
marked ⚠️ are ones actually hit.

## 0) Prerequisites (each node)

- Docker with NVIDIA GPU support (`docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 true` works).
- **Disk**: image is ~22.7 GB; the model snapshot is ~155 GB (48 shards) — HF cache needs room on **both** nodes.
- Outbound internet to `ghcr.io` (base image) and `huggingface.co` (model, only if cache absent).
- Both nodes cabled on the same RoCE fabric (ConnectX-7/QSFP) and the same control-plane Ethernet network.
- **Passwordless SSH from the head node (rank 0) to the worker (rank 1)** — the build script rsyncs to the worker over SSH.

Roles: **node0 = leader** (rank 0, runs the API server), **node1 = follower** (rank 1, `--headless`).

## 1) Clone the recipe (both nodes)

```bash
git clone https://github.com/taoofshawn/spark-recipes.git ~/code/spark-recipes
cd ~/code/spark-recipes && git checkout deepseek-v4-flash-tonyd2wild
```

The committed tree already has executable bits on all `upstream/*.sh` scripts,
so no `chmod` is needed (⚠️ older clones without the `100755` bits fail with
`./build-dspark-vllm-runtime.sh: Permission denied` — fix with
`chmod +x upstream/*.sh upstream/scripts/*.sh`).

## 2) Discover interfaces and set per-node IPs

The committed `.env` / `.env.node0` / `.env.node1` contain one cluster's values
by default. For new hardware, find the names on each node:

```bash
# Ethernet (control plane) — the "BROADCAST,UP" en* ports
ip -br addr show | grep -E '^en'
# RoCE (data plane) — maps each ethernet port to its RDMA device
ibdev2netdev        # or: rdma link show
```

Typical mapping: `enp1s0f0np0` ↔ `rocep1s0f0`, `enP2p1s0f0np0` ↔ `roceP2p1s0f0`.
The RoCE **IPv4 address** of the leader goes in `MASTER_ADDR` (`.env`) and
`ROCE_IP` (`.env.node0`); the follower's goes in `.env.node1`.

| file | fields to change on new hardware |
|---|---|
| `.env` | `MASTER_ADDR` (leader RoCE IP), `ETH_IF`, `ETH_IF2`, `IB_PORTS` |
| `.env.node0` | `ROCE_IP` (leader) |
| `.env.node1` | `ROCE_IP` (follower) |

`NCCL_IB_GID_INDEX` is left empty — the compose auto-detects the RoCE-v2 IPv4
GID at boot (override only if detection ever picks the wrong one).

## 3) Configure the build environment (`upstream/.env.dspark`)

The build script (`build-dspark-vllm-runtime.sh`) requires `WORKER_HOST` in
`upstream/.env.dspark` to rsync the build to the worker (⚠️ without it you get
`WORKER_HOST: WORKER_HOST must be set in …/.env.dspark`). A cluster-specific
copy is committed — edit it for new hardware:

```bash
cd upstream
# key values to change per cluster:
#   WORKER_HOST=<worker ssh hostname>  (e.g. spark-6d14.shawndo.intra)
#   MASTER_ADDR=<leader RoCE IP>
#   VLLM_HOST_IP=<leader RoCE IP>      WORKER_VLLM_HOST_IP=<worker RoCE IP>
#   NCCL_IB_HCA=<roce ports>           NCCL_SOCKET_IFNAME=<eth ifaces>
#   HF_CACHE=/home/<user>/.cache/huggingface   WORKER_HF_CACHE= (same path on worker)
```

⚠️ If the head has never SSH'd to the worker under that hostname, add the host key first:

```bash
ssh-keyscan -H spark-6d14.shawndo.intra >> ~/.ssh/known_hosts
ssh spark-6d14.shawndo.intra hostname   # confirm passwordless login works
```

## 4) Ensure the 0731 model is cached (BOTH nodes)

TP=2 reads the snapshot independently on each node — a missing cache on the
worker is a classic failure. The revision is pinned by `MODEL_REVISION` in
`docker-compose.yml`.

```bash
cd ~/code/spark-recipes/deepseek-v4-flash-tonyd2wild
HF_MODEL=$(grep "MODEL_PATH:" docker-compose.yml | awk '{print $NF}')
HF_REVISION=$(grep "MODEL_REVISION:" docker-compose.yml | awk '{print $NF}')
hf download $HF_MODEL --revision $HF_REVISION
```

The container mounts `${HF_CACHE}` (in `.env`, default
`/home/sdrew/.cache/huggingface`) at `/cache/huggingface`. If the container ever
runs as uid 1000 (e.g. the upstream launcher scripts) and the cache is root-owned,
startup fails — `sudo chown -R 1000:1000 <hf-cache>` fixes that. This compose runs
as root, so it is not needed here.

## 5) Build the image (head node; CPU build, no GPU needed)

There is **no prebuilt public image** — the final image is a 4-stage overlay
build on top of a public base (`ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready`):

```bash
cd ~/code/spark-recipes/deepseek-v4-flash-tonyd2wild/upstream
docker pull ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready   # ~5 GB, once
./build-dspark-vllm-runtime.sh
```

What it does:
1. builds `vllm-dspark-runtime:mia-raf-pr1` (overlay: 18 patched vLLM source files) — 1–2 min cached
2. `…-nvfp4-a` → `…-nvfp4-b` → **`vllm-dspark-runtime:dspark-nvfp4-stage-c`** (NVFP4 KV stages, ~seconds each)
3. verifies imports + prints `dspark nvfp4 stage-c image ok 0.21.1rc1.dev339+…`
4. rsyncs `upstream/` to the worker and rebuilds there (`WORKER_BUILD=1`; disable with `WORKER_BUILD=0`). First worker run pulls the base image again (~5 GB).

Alternative: build once and ship the finished image:
`docker save vllm-dspark-runtime:dspark-nvfp4-stage-c | ssh <worker> docker load`.

## 6) Verify the build on both nodes

```bash
docker images vllm-dspark-runtime:dspark-nvfp4-stage-c        # expect ~22.7 GB on BOTH
# Patch 4 (0731 shared-expert fix) — without it 0731 decode ~halves (acceptance ~26%):
docker run --rm --entrypoint grep vllm-dspark-runtime:dspark-nvfp4-stage-c \
  -n "shared_experts.gate_up_proj" \
  /opt/env/lib/python3.12/site-packages/vllm/v1/spec_decode/dspark.py
# Expected: lines 33-34  ("shared_experts.gate_up_proj", ".shared_experts.w1", 0) / (…".w3", 1)
```

## 7) Start (worker first, then leader ~30–35 s later)

```bash
# Node 1 (follower/headless) — start FIRST
docker compose --env-file .env --env-file .env.node1 up -d
# ~30–35 s later, Node 0 (leader)
docker compose --env-file .env --env-file .env.node0 up -d
```

API serves at `http://HEAD_NODE_IP:4000/v1` (served model `deepseek-v4-flash`).

## 8) Confirm it is healthy

First boot takes **~7–8 min** (model load ~3 min + warmup/compile). Watch the leader:

```bash
docker logs -f ds4-dspark
# wait for: "Application startup complete"
curl -s http://127.0.0.1:4000/v1/models | python3 -m json.tool
# expect: "id": "deepseek-v4-flash", "max_model_len": 1048576
```

Health markers in the boot log (all confirmed on a working deploy):

| marker | meaning |
|---|---|
| `Using nvfp4_ds_mla data type to store kv cache` | NVFP4 DS-MLA KV active |
| `Resolved architecture: DeepSeekV4DSparkModel` | model recognized |
| `world_size=2 … backend=nccl` / `DP group leader … world_size=2` | both nodes joined TP=2 |
| `Using 'B12X' Mxfp4 MoE backend` | the speed-critical MoE path (do not disable) |
| `num_spec_tokens=5` | DSpark k=5 |
| `GPU KV cache size: ~1.55–1.6M tokens` | KV pool (per-boot variance ~15%) |
| `Application startup complete` | API up |

Then send a real request and confirm SpecDecoding metrics appear:
`docker logs --tail 40 ds4-dspark | grep "SpecDecoding metrics"`.

⚠️ **Warm-up / cold-start**: a fresh boot is ~30% slower until a few hundred
tokens of real traffic pass through, and the warm state decays after idle —
never benchmark straight after boot or after a quiet period.

## files

| File/dir | Purpose |
|---|---|
| `.env` | Shared config — must be customized per cluster |
| `.env.node0` / `.env.node1` | Per-node overrides (rank, headless, RoCE IP) |
| `docker-compose.yml` | compose file |
| `upstream/` | **Vendored upstream repo** (build scripts, patches, docs) — see `upstream/VENDORED-AT.md` |

## Key knobs

| var | default | what it controls |
|---|---|---|
| `MTP_NUM_TOKENS` | 5 | DSpark `num_speculative_tokens` (k=5 validated; k=3 ≈ −24%) |
| `MAX_NUM_SEQS` | 6 | concurrency cap (6 is measured-best at 1M; 12 is riskier) |
| `GPU_MEM` | 0.78 | keep ≤0.78 on this stack |
| `MAX_MODEL_LEN` | 1048576 | 1M is the true YaRN ceiling |
| `THINKING` | true | server `thinking` default; clients can override per request |
| `REASONING_EFFORT` | high | server `reasoning_effort` default (low/high/max; high is the agentic sweet spot) |
| `VLLM_USE_FASTOKENS` | 1 | fastokens shim (10x+ faster tokenize, TTFT win at large context); installed at boot if missing |
| `PORT` | 4000 | serve port |

`MAX_NUM_BATCHED_TOKENS=8192` and `max-cudagraph-capture-size=seqs×(k+1)` are
derived per the upstream's validated profile — don't touch without re-measuring.

## Troubleshooting (real hits)

| Symptom | Cause / fix |
|---|---|
| `./build-dspark-vllm-runtime.sh: Permission denied` | scripts not executable — `chmod +x upstream/*.sh upstream/scripts/*.sh` (fixed in the committed tree) |
| `WORKER_HOST: WORKER_HOST must be set in …/.env.dspark` | build env missing — create/fix `upstream/.env.dspark` (committed copy exists) |
| `Host key verification failed` (during build rsync) | add host key on head: `ssh-keyscan -H <worker> >> ~/.ssh/known_hosts` |
| 0731 decode ~33 tok/s / acceptance ~26% | Patch 4 missing in the image — re-verify step 6 |
| boots then dies under traffic at `GPU_MEM=0.80` | keep `GPU_MEM` ≤ 0.78 |
| `No available shared memory broadcast block found in 60 seconds` | benign — other rank still compiling; resolves on its own |
| `Truncating max_cudagraph_capture_size to 32` | benign on this vLLM; DSpark captures at 24 (valid k+1=6 multiple) |
| `torch.compile is turned on, but the model … does not support it` | contradictory benign warning — compile actually runs (~22 s, AOT-cached) |
| `min_p and logit_bias parameters won't work with speculative decoding` | expected under DSpark |
| `SymmMemCommunicator: Device capability 12.1 not supported` | expected on GB10; falls back to PYNCCL |
| Missing module/path during build or first run | this upstream occasionally references modules living in **another repo by the same author** (`tonyd2wild`) — search his other public repos first |

## Benchmarking caveats

- **Use `stream: false`** and read `usage.completion_tokens` — under spec-decoding,
  streamed deltas measure *steps/s*, not tok/s (up to ~4× under-report).
- **Warm the engine** — fresh boot is ~30% slow; warm state decays after idle.
- KV pool is per-boot (varies ~15%); the 1.5M figure is not a fixed property.

## Attribution / upstream

Built from **tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark**,
pinned at commit `d728faee9f5a8d5ebafe7bc44bca6c5d8d0d192f` (2026-07-31), fully
vendored under `upstream/` (see `upstream/VENDORED-AT.md` for refresh steps and
license). Patches: 1/2/2b (DSpark concurrency), 3 (k=5 garble fix), 4 (0731
shared-expert loader) — all baked into the image at build time.
