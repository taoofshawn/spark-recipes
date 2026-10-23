# deepseek-v41-flash-tp4-switchless — operations README

DeepSeek-V4.1-Flash **1017 release**
([`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
@ `1bae0e10`, MIT — weights-only successor of the sparkring-pinned `DeepSeek-V4.1-Flash`
@ `dba1be0a`; config.json + 48-shard layout byte-identical) served on the **4-node
switchless ConnectX-7 ring** with vLLM TP4 (`dsv41-feat` branch + FlashInfer 0.7.0rc1),
DSpark k=5 speculative decoding (draft layers are INSIDE the checkpoint — no separate
drafter), 1M-token context, Engram n-gram tables on NVMe. Adapted from the
[FujitsuPolycom sparkring](https://github.com/FujitsuPolycom/sparkring)
`deepseek-v41-flash-cycle` vLLM lane @ `ad1853fc` (vendored at
[upstream/VENDORED-AT.md](upstream/VENDORED-AT.md)).

**Transport:** the site's shared four-PF switchless NCCL contract — same lib
(`~/.local/lib/nccl-patched-4pf`, SHA `8de68def…`), same netplan (all four PCIe views,
`10.10.0.0/21`), same HCA list and env flags as both GLM lanes. One `verify-fabric-p2.sh`
preflight serves all three lanes. The sparkring upstream's own transport (SparkRing
patched NCCL 2.30.7, 2 PFs, donor-image extraction) is the documented rollback
(VENDORED-AT.md "Transport decision").

**Endpoint (after bring-up):** `http://10.69.42.170:8000/v1` — model name
`deepseek-v41-flash`, 1,048,576-token context. No auth/TLS — trusted LAN only.
Thinking **ON by default** (`SGLANG_DEFAULT_THINKING=true` in the site env's
`EXTRA_CONTAINER_ENV`; upstream's default is off — this is the GLM-lanes parity
delta, owner-set 2026-10-23). Clients override per request:
`"chat_template_kwargs": {"thinking": false}` disables it; `enable_thinking` is an
accepted alias. Reasoning effort is the V4.1 numeric budget: tiers
`low=50 / high=75 (xhigh alias) / max=100`, server default **75**
(`SGLANG_DSV41_REASONING_EFFORT=75`); per request via the OpenAI `reasoning_effort`
field (tier, or a 0–0.99 float = budget/100) or
`"chat_template_kwargs": {"reasoning_effort": N}`. Sampling defaults when a request
omits them: temperature 1.0 / top_p 1.0 / top_k whole-vocab (the checkpoint ships no
`generation_config.json`); thinking requests whose `max_tokens` is below
`DSV41_MAX_NEW_TOKENS_FLOOR=16384` are raised to it so the reasoning segment cannot
starve the answer. Behind the model-name proxy clients keep
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
| `deepseek-v41-flash-tp4-switchless/upstream/` | vendored sparkring subset @ `ad1853fc` (image builder, 7 runtime patches, cycle launcher + 2 site mods, serving contract JSON) |
| `deepseek-v41-flash-tp4-switchless/site/env.switchless` | the site env (cluster transport + model pins + serving values) |
| `deepseek-v41-flash-tp4-switchless/site/materialize-weights.sh` | per-node weight materialization (hardlinks from the default HF cache; no download) |
| `deepseek-v41-flash-tp4-switchless/site/start-cluster.sh` | cluster control (check/serve/stop/status/logs) — wraps the upstream per-rank launcher over the SSH mesh, workers 3,2,1 first |
| `deepseek-v41-flash-tp4-switchless/site/build-engram-packed.sh` | once-per-rank packed Engram shard builder (measured +15% prefill upstream) |

On the nodes: `~/code/spark-recipes` (same branch convention as the GLM lanes — the
per-lane working branch carries all recipes); `~/.local/ds41f-tp4/` = recipe staging
(`upstream/`) + materialized weights + cache; `~/.local/lib/nccl-patched-4pf/` = the
SHARED four-PF NCCL build (never modify; both GLM lanes use it). HF cache holds
`deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e10` (≈475 GiB × 4 nodes — one local copy
per node is REQUIRED: Engram rows are read from shards 47/48 at serve time, NFS will
bound throughput).

## Image (the highest-risk item)

The pinned upstream base image no longer exists on Docker Hub; this site serves
`local/sparkring-deepseek-v41:overlay6` — rebuilt on substituted base
`vllm/vllm-openai:nightly-dev-arm64-cu13.0.1-591bb95` plus a pip-installed branch-era
DeepGEMM (`8b1392b`). Full lineage and rationale: [upstream/VENDORED-AT.md](upstream/VENDORED-AT.md)
site mod #3 and research.md 2026-10-18.

No public image exists — build locally on an idle Spark per
[upstream/README.md](upstream/README.md) (`build-image.sh`), then
`docker save local/sparkring-deepseek-v41:overlay6 | ssh <rank> docker load` to the
other three. Pins: vLLM `dsv41-feat @ e47aa780…`, FlashInfer `0.7.0rc1 @ 07869c61…`,
receipt in [upstream/image-receipt.json](upstream/image-receipt.json). Pin the built
image's ID as `IMAGE_ID` in `site/env.switchless` — the launcher refuses a mismatch.
The full extension build needs an idle node (a CUTLASS TU takes >7 GiB of compiler RSS).

## Bring-up (first time)

1. **Weights materialization (every node):**
   `bash deepseek-v41-flash-tp4-switchless/site/materialize-weights.sh`
   — hardlinks the HF-cache snapshot @ `1bae0e10` into
   `~/.local/ds41f-tp4/weights/deepseek-v41-flash-1017/` (zero copy, zero modification;
   verifies `DeepseekV41ForCausalLM` + shard 48 present).
2. **Image build + distribution** (above), `IMAGE_ID` pinned in `site/env.switchless`.
3. **Packed Engram shards (every node, once per checkpoint):**
   `RANK=<r> bash deepseek-v41-flash-tp4-switchless/site/build-engram-packed.sh`
   (~9 min/rank, CPU only; ~48 GB allocated per rank; refuses an existing out-dir).
4. **Preflight:** `bash glm-v53-flash-fp8-tp4-switchless/noswitch-prep/scripts/verify-fabric-p2.sh`
   must print `ALL CHECKS PASS` (shared netplan + NCCL lib verification).
5. **Validate:** `DRY=1` not supported — use
   `deepseek-v41-flash-tp4-switchless/site/start-cluster.sh check` (offline `--check`
   of the upstream launcher on all four ranks; prints the exact docker commands).
6. **Serve:** `./start-cluster.sh serve` — ranks 3,2,1 then 0; ~8 min to readiness from
   local NVMe (~4 min weights, ~1 min DSpark draft, ~2 min graphs + autotune).
7. **Gates (within minutes of `/health` 200):**
   - `/v1/models` advertises `deepseek-v41-flash`, 1,048,576-token context.
   - Coherent response + forced tool call (`deepseek_v41` parser round-trip).
   - Corruption probe: emoji/CJK generation, count U+FFFD — must be 0.
   - Boot log per rank: `Engram DISK mode: layer …` lines (one per Engram layer),
     `BALANCED column assignment, rank r owns hash columns […]` (six distinct columns
     per rank — identical `off=` values on every rank mean the rank-offset fix did not
     mount), `PACKED single-read shard`, `Application startup complete`.
   - `SpecDecoding metrics` mean acceptance length > 1 (DSpark k=5 active).
   - Record the actual `GPU KV cache size` line (the pool is sized from free memory —
     upstream measured 10.94 GiB / 2,182,642 tokens at the 430K limit + 0.83 util;
     the 1M default's pool is a bring-up observation, not yet a receipt).

## Switching (GLM lanes ↔ this lane)

Runbook: [`../../LANE-SWITCHING.md`](../../LANE-SWITCHING.md) — the one-doc procedure;
this lane's rows are in "The lanes" table below it. Lane-specific:

- **Teardown of this lane:** `./start-cluster.sh stop` (rank 0 FIRST — a worker that
  starts while an old head still listens on the rendezvous port joins the old head and
  hangs; upstream gotcha). Stop preserves nothing here: the site wrapper removes the
  stopped container (`docker stop --time 30` + `docker rm`) — relaunch is a fresh
  `serve`, model/cache bind mounts stay on disk.
- **Proxy:** `BACKEND_MODEL=deepseek-v41-flash`; add `deepseek-v41-flash` to
  `SPOOF_FROM` in `model-name-proxy/.env`, then `docker compose --env-file .env up -d`
  (leader node). One-line change, same as every other switch.
- **No netplan/NCCL decisions at switch time:** the four-PF netplan + shared
  `nccl-patched-4pf` lib serve all three lanes. `verify-fabric-p2.sh` before every boot.
- **Branches are per-lane** (node checkout follows whichever lane is serving; `main`
  never used on nodes).

## Day-2 ops

- Start/stop/status/logs: `site/start-cluster.sh` (wraps upstream
  `deepseek_v41_cycle_serve.sh --check|--run` per rank). The upstream launcher enforces:
  MemAvailable ≥ 100 GiB before launch (reboot the rank first — memory fragmentation on
  GB10 costs real throughput), patch MD5 verification, image-ID match, no existing
  container name.
- **Sizing:** 1M context is configured but upstream's measurements ran through 430,080
  tokens (400K needle pass); a full-1M request is NOT yet evidenced. `max_num_seqs 8`
  is the recorded profile (16 probed OK as an admission option at C16). Changing
  admission/memory settings requires re-checking startup + MemAvailable + preemption.
- **Warm restarts:** caches (vLLM/FlashInfer/Triton/TVM-FFI) live in
  `~/.local/ds41f-tp4/cache/` (bind-mounted at `/cache`) — relaunches against the same
  paths stay warm.
- Update procedure: [upstream/VENDORED-AT.md](upstream/VENDORED-AT.md). Any upstream
  `recipes/deepseek-v41-flash-cycle.json` serving change needs a review against
  `site/env.switchless` before inheriting.
- Do not edit files on the nodes directly — recipe changes flow through the repo
  (topic branch → push → user merges → `git pull` on nodes; agents never open PRs).

References: sparkring
[quickstart](https://github.com/FujitsuPolycom/sparkring/blob/main/profiles/deepseek-v41-flash-cycle/README.md)
· [image builder](https://github.com/FujitsuPolycom/sparkring/blob/main/runtime/deepseek-v41-gb10/README.md)
· [benchmark record](https://github.com/FujitsuPolycom/sparkring/blob/main/performance/records/deepseek-v41-flash/cycle-tp4-dspark5-graphs-20260910.md)
· sibling GLM recipes `glm-v53-flash-fp8-tp4-switchless/` +
  `glm-v53-flash-nvfp4-tp4-switchless/` (fabric + NCCL provenance)
