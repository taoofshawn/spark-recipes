# deepseek-v41-flash-tp4-switchless — operations README

DeepSeek-V4.1-Flash **1017 release**
([`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
@ `1bae0e10`, MIT) served on the **4-node switchless ConnectX-7 ring** with SGLang
(dsv4.1 canary branch) TP4/EP1, DSpark speculative decoding (draft layers are INSIDE the
checkpoint — no separate drafter), 1M-token context, Engram n-gram tables on NVMe.
Vendored from [knapcio's 4x-DGX-Spark TP4 stack](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4)
@ `aff52fa` (vendored at [upstream/VENDORED-AT.md](upstream/VENDORED-AT.md); the MiaAI-Lab
stack it replaces is archived at `upstream-miaai-archive/` — swap rationale in
research.md 2026-10-26).

**Transport:** the site's shared four-PF switchless NCCL contract — same lib
(`~/.local/lib/nccl-patched-4pf`, SHA `8de68def…`), same netplan (all four PCIe views),
same HCA list and env flags as both GLM lanes. One `verify-fabric-p2.sh` preflight serves
all three lanes. The launcher overlays the host NCCL onto the container's pip NCCL
(`NCCL_OVERLAY_PIP=1`, ring-mode default) and refuses a stock library.

**Endpoint (after bring-up):** `http://10.69.43.170:8000/v1` (the head's client
network, `wlP9s9`; the 10.69.42.x mgmt fabric serves SSH only and refuses :8000) — model name
`deepseek-v41-flash`, 1,048,576-token context. No auth/TLS — trusted LAN only.
Thinking **ON by default** (`SGLANG_DEFAULT_THINKING=true` in the site env's
`EXTRA_CONTAINER_ENV`; upstream's default is off — this is the GLM-lanes parity
delta, owner-set 2026-10-23; the adapter's explicit-kwarg injection is REQUIRED for the
parser to split the block). Clients override per request:
`"chat_template_kwargs": {"thinking": false}` disables it; `enable_thinking` is an
accepted alias. Reasoning effort is the V4.1 numeric budget: tiers
`low=50 / high=75 (xhigh alias) / max=100`, server default **75**
(`SGLANG_DSV41_REASONING_EFFORT=75`); per request via the OpenAI `reasoning_effort`
field (tier, or a 0–0.99 float = budget/100) or
`"chat_template_kwargs": {"reasoning_effort": N}`. Sampling defaults when a request
omits them: temperature 1.0 / top_p 1.0 / top_k whole-vocab (the checkpoint ships no
`generation_config.json`). Requests that **explicitly** enable thinking and ask for
fewer than `DSV41_MAX_NEW_TOKENS_FLOOR=16384` tokens are raised to it, so the
reasoning segment cannot starve the answer. A request that only inherits the
default keeps the `max_tokens` it sent — say `{"thinking": true}` if you want the
floor (fixed 2026-10-23: the inherited default used to authorize the raise, which
inflated fixed-budget throughput probes; bench rows from `fix1` are not comparable
with earlier rows). Clients that want fixed-length generation for measurement MUST
keep sending `max_tokens` and no thinking kwarg, or send `{"thinking": false}`.
Behind the model-name proxy clients keep
sending `spark-llm`; at switch time set `BACKEND_MODEL=deepseek-v41-flash` and add it to
`SPOOF_FROM` (model-name-proxy/.env).

## Cluster map

| rank | node | mgmt IP | role |
|---|---|---|---|
| 0 | `spark-0f0b` | 10.69.42.170 | leader — API server (`:8000`), `start-cluster.sh` runs here |
| 1 | `spark-6d14` | 10.69.42.171 | worker |
| 2 | `spark-6d90` | 10.69.42.172 | worker |
| 3 | `spark-6d24` | 10.69.42.173 | worker |

Ring fabric: the shared four-PF netplan (10.10.1.x–10.10.4.x + P2 views), MTU 9000.
GPU contention rule: only ONE lane serves at a time — the active GLM lane must be torn
down first (`tp4ctl down` / `./start.sh stop` in its recipe dir).

## Locations

| path (from repo root) | what |
|---|---|
| `deepseek-v41-flash-tp4-switchless/upstream/` | vendored knapcio tree @ `aff52fa` (launcher, image builder + staged canary tree, adapters, tests); `runtime/sglang-canary/` is gitignored — staged by `scripts/fetch-sglang-canary.sh` |
| `deepseek-v41-flash-tp4-switchless/upstream-miaai-archive/` | the replaced MiaAI-Lab stack (`64b02dea`), kept for fix provenance/rollback reading — do NOT resurrect |
| `deepseek-v41-flash-tp4-switchless/site/env.switchless` | the site env (cluster transport + model pins + serving values) |
| `deepseek-v41-flash-tp4-switchless/site/materialize-miaai-weights.sh` | per-node weight materialization (hardlinks from the default HF cache; no download; name is historical — the destination path it writes is the one the site env points at) |
| `deepseek-v41-flash-tp4-switchless/site/start-cluster.sh` | cluster control (doctor/build/pack/serve/stop/status/logs/smoke) — renders the site env into `.env.tp4-switchless` and execs the upstream launcher over the SSH mesh |
| `deepseek-v41-flash-tp4-switchless/site/build-cluster.sh` | per-node image build WITHOUT the launcher's worker-stage rsync (the `--delete` rsync into `WORKER_DIR` destroys the weights bind source — 2026-10-23 gotcha); streams `git archive` per node and verifies the in-image adapter byte-for-byte |

On the nodes (paths kept from the MiaAI bring-up — renaming would force weight
re-materialization + docker-volume recreation for no operational gain): the head's
`~/.local/ds41f-miaai/stack/` = vendored tree + rendered `.env.tp4-switchless`;
`~/.local/ds41f-miaai/{weights,engram}/` = materialized checkpoint + packed Engram
shards; workers' `~/dsv41-miaai/` = build tree; `~/.local/lib/nccl-patched-4pf/` = the
SHARED four-PF NCCL build (never modify; both GLM lanes use it). HF cache holds
`deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e10` (≈475 GiB × 4 nodes — one local copy
per node is REQUIRED: Engram rows are read from shards 47/48 at serve time, NFS will
bound throughput).

## Image

Base image `lmsysorg/sglang:dev-dsv41` pinned to digest
`sha256:3dbc3130…` (verified pullable on spark-0f0b 2026-10-21; the digest rides
`BUILD_ARGS` in `site/env.switchless` because the vendored Dockerfile's
`ARG BASE_IMAGE` otherwise resolves the mutable tag). The image adds the pinned
SGLang dsv4.1 canary tree (staged, not tracked), the sglang-kernel 0.4.7 /
sgl-deep-gemm 0.2.0 wheels, the b12x/b12x_next MoE line, the SM120 flash-MLA
override, and the build-time patches: knapcio's image-placeholder sanitize plus the
site's SWA tail-eviction backport (#35515). Site mods and lineage:
[upstream/VENDORED-AT.md](upstream/VENDORED-AT.md).

Build LOCALLY via `site/build-cluster.sh [commit-ish]` FROM THE HEAD NODE (builds the
head + every worker and verifies each image carries the commit's
`adapter/encoding_compat.py` byte-for-byte). The first build stages
`runtime/sglang-canary/` via the fetch script on the head (pinned refs; workers get the
rsynced copy). Do NOT use the launcher's own `build` for image swaps — its worker stage
rsyncs `--delete` into `WORKER_DIR` (the weights bind source; the 2026-10-23 incident).
The launcher's `serve` auto-triggers it only when an image is missing — build first.

## Bring-up (after a stack change)

1. **Preflight:** `bash glm-v53-flash-fp8-tp4-switchless/noswitch-prep/scripts/verify-fabric-p2.sh`
   must print `ALL CHECKS PASS` (shared netplan + NCCL lib verification).
2. **Image build:** `bash deepseek-v41-flash-tp4-switchless/site/build-cluster.sh`
   (head + workers; adapter sentinel verified per node).
3. **Packed Engram shards:** `deepseek-v41-flash-tp4-switchless/site/start-cluster.sh pack`
   — the upstream `pack` command, once per checkpoint. Idempotent per layer: already-complete
   shards are skipped ("layer N: already complete"), incomplete/mismatched ones rebuilt
   (the shards on disk came from the MiaAI packer; the launcher's packer converges them).
4. **Validate:** `deepseek-v41-flash-tp4-switchless/site/start-cluster.sh doctor` — the
   launcher's connectivity/GPU/weights/ring preflight on all four ranks.
5. **Serve:** `./start-cluster.sh serve` — worker ranks then head; the launcher refuses
   a failed ring preflight before replacing containers.
6. **Gates (within minutes of `/health` 200):**
   - `/v1/models` advertises `deepseek-v41-flash`, 1,048,576-token context.
   - Coherent response + forced tool call (round-trip through the dsv4.1 parser).
   - Corruption probe: emoji/CJK generation, count U+FFFD — must be 0.
   - Boot log per rank: 0 NCCL WARNs, `SWITCHLESS_RING_ONLY` marker present in the
     container's NCCL, the patch tags `# [encoding_dsv41-placeholder-sanitize]` and
     `# [swa-radix-tail-evict]` present in the container's patched sources, and the
     adapter's boot warning line `…requests that explicitly enable thinking below 16384
     tokens are raised to it…`.
   - `SpecDecoding`/DSpark acceptance > 1 (speculative decoding active).
   - Record the actual `max_total_num_tokens` line (pool sized from free memory).
   - Retention (the swap's reason): after two agent-style turns with an idle gap, a
     follow-up request must report `cached_tokens ≈ full prompt` — the A→B harness in
     research.md 2026-10-26. If the cap-mode 2–4 s GPU wipe or the churn-case 0-hit
     returns, capture the boot config (`swa_tokens`, `c4_size`, ratio) BEFORE touching
     anything — that is the diagnostic signal the 2026-10-25/26 investigation was missing.

## Switching (GLM lanes ↔ this lane)

Runbook: [`../../LANE-SWITCHING.md`](../../LANE-SWITCHING.md) — the one-doc procedure;
this lane's rows are in "The lanes" table below it. Lane-specific:

- **Teardown of this lane:** `./start-cluster.sh stop`. The site wrapper removes the
  stopped containers — relaunch is a fresh `serve`, model/cache bind mounts stay on disk.
  A FAILED boot leaves stopped containers in `Created/Exited` — `docker rm -f` per rank
  before relaunch (the launcher refuses existing names).
- **Proxy:** `BACKEND_MODEL=deepseek-v41-flash`; add `deepseek-v41-flash` to
  `SPOOF_FROM` in `model-name-proxy/.env`, then `docker compose --env-file .env up -d`
  (leader node). One-line change, same as every other switch.
- **No netplan/NCCL decisions at switch time:** the four-PF netplan + shared
  `nccl-patched-4pf` lib serve all three lanes. `verify-fabric-p2.sh` before every boot.
- **Branches are per-lane** (node checkout follows whichever lane is serving; `main`
  never used on nodes).

## Day-2 ops

- Start/stop/status/logs: `site/start-cluster.sh` (renders env → execs the upstream
  `start.sh`). The launcher enforces: non-blocking doctor at serve, GPU-busy refusal
  (`FORCE=1` to override), ring config validation + head preflight BEFORE replacing
  containers, local-weights completeness checks per node, and the NCCL marker grep.
- **Sizing:** pinned KV pool 8,000,000 tokens (`MAX_TOTAL_TOKENS`) under
  `MEM_FRACTION_STATIC=0.80` with `MAX_RUNNING_REQUESTS=16`; `CHUNKED_PREFILL_SIZE=4096`
  is safe at any context WITH `DSV41_INDEXER_CHUNKED=1` (armed in
  `EXTRA_CONTAINER_ENV`). Changing admission/memory settings requires re-checking
  startup + MemAvailable + retraction behavior.
- **Warm restarts:** state/logs live under the launcher's per-profile dirs
  (`state-tp4`/`logs-tp4` semantics via `STATE_DIR`/`LOG_DIR` in the rendered env) —
  relaunches against the same paths stay warm; a stale Triton/JIT cache after an image
  swap can poison a boot (clear the state dir if a boot hangs in capture).
- Update procedure: [upstream/VENDORED-AT.md](upstream/VENDORED-AT.md) (fetch-script pin
  changes cascade into the image; the sglang re-pin plan in research.md §6 item 8
  supersedes incremental updates when it runs).
- Do not edit files on the nodes directly — recipe changes flow through the repo
  (topic branch → push → user merges → `git pull` on nodes; agents never open PRs).

References: knapcio
[repo README](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4#readme)
· [switchless-ring doc](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/blob/main/docs/switchless-ring.md)
· sibling GLM recipes `glm-v53-flash-fp8-tp4-switchless/` +
  `glm-v53-flash-nvfp4-tp4-switchless/` (fabric + NCCL provenance)
