---
name: bring-up-spark-recipe
description: Use when bringing a recipe in the spark-recipes repo up on the 2-node DGX Spark cluster — starting/staging a recipe directory, tearing down the currently-serving model, rebuilding the patched image, launching worker-then-head, and confirming health. Triggers: "bring up <recipe>", "start <recipe> on the sparks", "deploy the glm/qwen/deepseek recipe", "latest recipe not running", "fresh head/worker after a model change", DFlash2/DFlash enablement.
---

# Bring Up a Spark Recipe

Deploy a `spark-recipes` recipe onto the fixed 2-node DGX Spark (GB10) cluster and
leave it serving healthily on port 4000. This is the standing operational loop for
this repo; it encodes the cluster's hard-won constraints so a fresh agent can go
from "repo checkout" to "model serving" without re-discovering them.

Two fixed nodes (address by DNS hostnames, NEVER the `192.168.0.x` RoCE IPs — those
time out from the management network):

| role | host | RoCE IP |
|---|---|---|
| head / leader / rank 0 (API server) | `spark-0f0b.shawndo.intra` | `192.168.0.170` |
| worker / follower / rank 1 (headless) | `spark-6d14.shawndo.intra` | `192.168.0.171` |

The recipe lives on BOTH nodes at `~/code/spark-recipes/<recipe>/` (a git checkout
of `taoofshawn/spark-recipes`).
## Scope boundary

This skill ONLY brings a recipe up and keeps it serving — it does NOT research,
review, or update recipe source code, and it does NOT create branches, commit, or
push changes to the repo. Updating a recipe against its upstream sources is the
`recipe-update` skill's job; run that first if the recipe needs refreshing, then
bring the refreshed recipe up here.

## When to use / not

Use for any docker-compose recipe: `glm-v53-flash`, `deepseek-v4-flash-aiden`,
`deepseek-v4-flash-tonyd2wild`, `mimo-v25-dflash-tonyd2wild`. The one sparkrun
recipe (`deepseek-v4-flash-aiden-sparkrun`) is managed with `sparkrun run`,
not this flow — but the teardown/health steps still apply.

## Core principle

**Worker (rank 1) starts first; head (rank 0) ~25-30 s later.** The multi-node TCP
store on `master_addr:25000` must be up on the head's port before the follower
connects; wrong order → `DistStoreError: 1/2 clients` / `Connection reset by peer`.
Always `docker compose down` on BOTH nodes between relaunches (stale-head
rendezvous otherwise hangs the head at "Init torch distributed begin").

## The loop (docker-compose recipes)

### 1. Assess current state (both nodes, parallel)

```bash
ssh spark-0f0b.shawndo.intra 'hostname; docker ps -a --format "{{.Names}}\t{{.Status}}\t{{.Image}}"; cd ~/code/spark-recipes && git rev-parse --abbrev-ref HEAD && git log --oneline -3 && git status --short'
ssh spark-6d14.shawndo.intra 'same...'
```

Note: the spark `git` branches are independent of the workstation repo. The node may
be on any branch (`main`, a recipe branch, a stale branch). If the recipe you want
isn't checked out, `git checkout <recipe-branch> && git pull origin <recipe-branch>`.

**GPU contention rule:** every vLLM recipe uses all reserved GPUs. Only one recipe
serves at a time. If any other model container is Up (e.g. `qwen38fn-vllm`), tear it
down.

### 2. Tear down the running model + stale container (both nodes)

```bash
ssh spark-0f0b.shawndo.intra 'docker stop <other-container> 2>/dev/null; docker rm -f <other-container> 2>/dev/null; docker rm -f <recipe-container> 2>/dev/null; cd ~/code/spark-recipes && git checkout <recipe-branch> && git pull origin <recipe-branch>'
ssh spark-6d14.shawndo.intra 'same'
```

### 3. Verify the model checkpoint(s) are cached (offline serving)

Each recipe's `.env`/compose names the HF models + pinned revisions it needs on BOTH
nodes. Check the HF cache dirs exist and are non-empty:

```bash
du -sh ~/.cache/huggingface/hub/models--<owner>--<model>
ls ~/.cache/huggingface/hub/models--<owner>--<model>/snapshots/
```

If a drafter/aux checkpoint is missing (e.g. `incoai/GLM-5.3-Flash-DFlash2`), download
it on BOTH nodes with the local `hf` CLI (`~/.local/bin/hf`):
`~/.local/bin/hf download <owner>/<model> --revision <rev>`.

BLOCK if the checkpoint is absent — serving is offline; it will not download at
runtime. State exactly which revision is missing and that it must be cached on both
nodes.

### 4. Set the target config in the recipe's `.env`

Each recipe enables its flagship features via `.env` variables with documented
defaults. Read the recipe README + `.env` to see the real knobs. For
`glm-v53-flash`, enabling DFlash2 (the latest-commit feature) means:

```
DFLASH2=1
SPEC_TOKENS=7        # DFlash2 block size 8 - 1; the compose FATAL-errors otherwise
```

The compose `command` block hard-enforces these invariants and FATAL-exits with a
clear message if violated — treat those messages as authoritative. Sync the edited
`.env` to BOTH nodes with `scp`.

### 5. Cache ritual (both nodes, before every launch)

GB10 is unified memory; heavy weight-load IO grows the page cache and starves the
NVRM allocator (it counts **MemFree, not MemAvailable**). Run on both nodes right
before `up`:

```bash
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches
```

Optional during the long load: `sudo ./tools/cache_flusher.sh &` (recipes ship it).

### 6. Validate compose config (each node with its own .env.nodeN)

```bash
cd ~/code/spark-recipes/<recipe> && docker compose --env-file .env --env-file .env.node0 config -q   # head
docker compose --env-file .env --env-file .env.node1 config -q                                        # worker
```

### 7. Build + launch — worker first, then head

```bash
# worker (rank 1): start FIRST
ssh spark-6d14.shawndo.intra 'cd ~/code/spark-recipes/<recipe> && docker compose --env-file .env --env-file .env.node1 up -d --build'
# ~30 s later: head (rank 0, API server)
ssh spark-0f0b.shawndo.intra 'cd ~/code/spark-recipes/<recipe> && docker compose --env-file .env --env-file .env.node0 up -d --build'
```

`--build` rebuilds the patched image (the recipes bake SM121 kernel patches into a
local image). The first build pulls the ~10 GB digest-pinned base image + re-pins
NCCL/cutlass/FlashInfer, then runs the patch chain — minutes. Run each as a
background SSH job and watch them complete.

DO NOT launch both `up` simultaneously; the worker must begin its rendezvous listen
first. If the worker fails FAST (container `Restarting` / a FATAL log), fix before
starting the head — a dead worker makes the head hang waiting to rendezvous.

### 8. Watch boot and verify health (don't stop until serving)

Weight load + engine init + kernel warmup takes ~15-25 min (GLM: 195 GB weights).
Poll the head log for the readiness sequence and confirm the model actually serves:

```bash
# Boot progress
ssh spark-0f0b.shawndo.intra 'docker logs --tail 5 glm53-nvfp4'
# Health (use /health — /v1/models returns 200 even with a dead engine)
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:4000/health
curl -s http://127.0.0.1:4000/v1/models
```

Health = `/health` returns 200 AND `/v1/models` shows the right `id` + `max_model_len`.
Confirm the recipe's boot markers in the log (README lists them per recipe; e.g.
GLM: `Using 'MARLIN' NvFp4 MoE backend`, `GPU KV cache size`, `Using Eagle3
auxiliary layers from config`, the rejection-sampler warmup line,
`Uvicorn running` / `Application startup complete`).

### 9. Feature-specific validation for the flagged feature

For DFlash2 (glm-v53-flash), confirm in the head log that the LATEST commits are
active, not a stale build:

- `Resolved architecture: DFlash2DraftModel` and
  `speculative_config=SpeculativeConfig(method='dflash', ... num_spec_tokens=7)`
  — proves the drafter is wired and SPEC_TOKENS=7 took.
- `Using Eagle3 auxiliary layers from config: (6, 15, 25, 34, 43)` — the raw
  `target_layer_ids [5,14,24,33,42]` after the runner's +1; proves aux capture glue.
- `reserved 3.0 GiB memory for KV Cache` — the DFlash2 3 GiB pin (upstream withdrew
  the 7 GiB ceiling); tells you the latest KV-pin commit is running.
- `Warming up spec-decode rejection sampler kernels (vocab=154880, num_spec=7, ...)`.

Then measure acceptance (broken aux capture shows as acceptance pinned near 1.0 /
position-0 under ~50%; healthy is position-0 acceptance ~0.6-0.8, per-position
decaying gently):
- Read `vllm:spec_decode_num_draft_tokens_total` + `...accepted_tokens_total` and
  `..._per_pos_total ` from `/metrics` before and after a warm generation, and
  compute the deltas. Skip the first inference (it JIT-compiles drafter kernels).
- `http://127.0.0.1:4000/metrics`.

### 10. Run the recipe's load test

Each recipe ships a tool-carrying load test (e.g. `python3 tools/load_test_glm.py`)
that normal smokes can't substitute for — it's the only thing that catches silent
FP4-corruption loops / degenerate token repetition under concurrency. Run it on the
head, expect a `VERDICT: PASS`.

## Troubleshooting quick reference

| symptom | cause / fix |
|---|---|
| `FATAL: <feature> requires SPEC_TOKENS=7` | compose invariant; set the var in `.env` (see step 4) |
| build fails at a `py_compile` / patch step | stale or wrong path in `patches/Dockerfile`; the patch script edits `vllm/models/...` (singular) while an older check may reference `model_executor/models/...` — fix the path, rebuild |
| `DistStoreError: 1/2 clients` / `Connection reset by peer` | start order — worker first; `down` BOTH nodes first |
| worker dies 1-2 min after "Initial free memory … reserved N GiB" | KV slab too big — drop `KV_CACHE_MEMORY` (README: default is the only safe value; do NOT raise pins) |
| boot dies right after weight load | MTP/drafter head trips UMA OOM without the KV pin — keep `KV_CACHE_MEMORY` set |
| health 200 but `/v1/models` empty/wrong | engine still initializing or a different model; keep polling |
| `No available shared memory broadcast block found in 60 seconds` | BENIGN — other rank compiling; resolves |
| `min_p and logit_bits won't work with speculative decoding` | expected under DFlash/MTP |
| `SymmMemCommunicator: Device capability 12.1 not supported` | expected on GB10; falls back to PYNCCL |
| every reply is garbage / repeated-token loop | silent FP4 MoE corruption — check `--moe-backend marlin` is set (auto FLASHINFER_CUTLASS corrupts on SM121) |

Recipe invariants are per-image and NOT transferable between recipes: start order,
`gpu_memory_utilization`, `max_num_seqs`, KV dtype, backend names, `--moe-backend`.
Never "tidy" env vars / flags that look duplicated — they are image-specific.

## Common mistakes

- Starting the head before the worker → rendezvous hang.
- Bringing up both nodes but tearing down between relaunches is skipped → stale-head
  hang at "Init torch distributed begin".
- Confirming health with `/v1/models` alone → returns 200 with a dead engine; use
  `/health`.
- Forgetting to `--build` → serves the OLD baked image without the new patch layers.
- Editing `upstream/` vendored files (tonyd2wild recipe) → recipe-level files in the
  parent dir are the customization layer; `upstream/` is for reference.
- Benchmarking right after boot → fresh boots are ~30% slower until warmup.

## Supporting file

`scripts/deploy_recipe.sh` — automates steps 2, 4-7 (teardown, config check, cache
ritual, worker-first build+launch) for a recipe directory. Review it before first
use; it hard-codes the two node hostnames and this repo's branch + `.env` conventions.
