---
name: adopt-recipe
description: >-
  Use when creating a brand-new recipe in the spark-recipes repo from an NVIDIA
  forum thread and/or a GitHub repo — reviewing the linked sources and generating
  a new docker-compose recipe directory on a fresh branch for review. Triggers:
  "adopt <link>", "create a recipe for <model>", "new recipe from <thread/repo>",
  "port <author>'s recipe", "add <model> to spark-recipes". Does NOT update an
  existing recipe (recipe-update) and does NOT bring anything up on the sparks
  (bring-up-spark-recipe).
---

# Adopt a Recipe

Create a NEW `spark-recipes` recipe from an external source — an NVIDIA DGX Spark
forum thread and/or a GitHub repo: review the links, extract the serving stack,
and generate a docker-compose recipe directory in this repo's house style, on a
new branch, **left uncommitted** for the user to review.

## Scope boundary

This skill CREATES a new recipe directory and stops there. It does NOT:

- update an existing recipe against its upstreams — that is `recipe-update`;
- bring the recipe up on the sparks, build images, or verify a live endpoint —
  that is `bring-up-spark-recipe`, run AFTER the recipe lands;
- commit, push, open a PR, or merge anything. The deliverable is an untracked
  directory on a fresh topic branch.

If the target looks like a variant of a recipe this repo already carries (same
model + same quant, just newer pins), CHECK WITH THE USER before doing anything
else — they may still want a separate recipe for the same model (different
quant, different lane, or some other difference). Only fall back to
`recipe-update` if the user says so.

## Inputs

| input | required | fallback |
|---|---|---|
| forum thread URL or thread ID | one of the two links | proceed with what you have; note the gap in `research.md` |
| GitHub repo URL | one of the two links | " |
| recipe name | no | derive from the model + distinguisher (step 3) |

## The loop

### 1. Review the target links (before writing anything)

Mine the sources the way AGENTS.md's "review how to query each source" section
prescribes. Forum = Discourse JSON API (no auth; browser-ish User-Agent;
paginate `/t/<id>.json` until you have every post). GitHub = raw README/docs +
the API for commits/PRs/issues. Extract — each item WITH a receipt (post
number, file path, commit SHA, log line):

| extract | feeds |
|---|---|
| model repo + pinned revision + quant method (+ license) | `MODEL` / `MODEL_REVISION`; whether a surgery/adaptation step is needed |
| serving image + digest/tag + vLLM version | `IMAGE`; which flag set applies |
| spec-decode method + k + drafter checkpoint + revision | the compose spec-config block |
| KV dtype / pin / pool, GMU, MAX_SEQS, MNBT, block size | the tuning rows |
| MoE/attention backend + why | boot-fatal on the wrong value |
| patches / overlays / mods the stack carries | what must be vendored into the recipe dir |
| measured claims (tok/s, TEB, acceptance) with their protocol | receipts for `research.md`; never adopt "in theory" numbers |
| boot markers + known failure modes | the README boot-marker section |

Also note the author handle(s) — the naming convention leans on them.

### 2. Decide the shape

Docker-compose recipe (this skill's output): `.env` + `.env.node0`/`.env.node1`
+ `docker-compose.yml` + `README.md` + `research.md`, exactly like
`glm-v53-flash-intel-w4a16` and the archived compose recipes. Upstream runtime
mods live inside the container: bind-mounted `:ro` overlays, or baked into a
locally built image (the `dflash2-pmu128/` pattern). Do NOT produce a sparkrun
recipe unless explicitly asked.

### 3. Name it

Name given → use it verbatim as the subdirectory (lowercase-hyphenate it if it
isn't already). No name → follow the existing pattern
`<model>-<distinguisher>`, where the distinguisher is the forum author's
handle, the quant, or the defining feature:

    glm-v53-flash-intel-w4a16      # model-quant
    deepseek-v4-flash-tonyd2wild   # model-author
    mimo-v25-dflash-tonyd2wild     # model-feature-author
    qwen-3.8-flash-next-tonyd2wild # model-author

Check the directory doesn't collide with an existing one (`glob` the repo root
first). If two candidates are equally good, pick one, say so, and record the
alternatives in `research.md`.

### 4. Branch, then create the directory

`main` = stable recipes only; direct pushes to `main` are blocked (AGENTS.md).
Branch from a fresh `main`:

    git checkout main && git pull origin main
    git checkout -b adopt-<recipe-name>
    mkdir <recipe-name>

### 5. Generate the recipe (house style)

Model the files on `glm-v53-flash-intel-w4a16/` — the live reference. Copy its
STRUCTURE, not its values. Every non-cluster value must trace to a reviewed
source; every cluster value is THIS cluster's, verbatim (AGENTS.md table):

- `.env` — shared config: `MASTER_ADDR=192.168.0.170`, `PORT=8000`,
  `MPORT=29521`, host cache paths (`MODEL_HOST_PATH` / `HF_CACHE` /
  `CACHE_HOST_PATH`), `ETH_IF=enp1s0f0np0` / `ETH_IF2=enP2p1s0f0np0` /
  `IB_PORTS=rocep1s0f0,roceP2p1s0f0`, `NCCL_SUBNET=192.168.0.0/24`, `IMAGE`,
  `MODEL` + `MODEL_REVISION`, `SERVED_MODEL_NAME`, lane/spec vars, `GMU`,
  `MAX_LEN`, `MAX_SEQS`, `MNBT`, `BLOCK_SIZE`, `KV_DTYPE`,
  `KV_CACHE_MEMORY`, `MOE_BACKEND`, `EAGER`, the sampling pins (step 6),
  `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`, and the NCCL/RoCE env block
  (copy it from the reference recipe's compose `environment:` section).
- `.env.node0` — `NODE_RANK=0`, `HEADLESS=`, `ROCE_IP=192.168.0.170`.
- `.env.node1` — `NODE_RANK=1`, `HEADLESS=1`, `ROCE_IP=192.168.0.171`.
- `docker-compose.yml` — the invariants below.
- `README.md` — ACTIVE-RUNNING doc only: model + quant, profiles, one-time
  provisioning (model/drafter `hf download` pins, any surgery script), image
  build, deploy (worker FIRST, head ~30–35 s later), verify (`/health` then
  `/v1/models`), boot markers, tuning rows, known issues, references.
  Historical changelog and TODO/watch material NEVER go here.
- `research.md` — provenance (which post/repo/file each value came from),
  measured receipts, watchlist, dated changelog blocks.

compose invariants (reference recipe + AGENTS.md): `entrypoint: []`;
`restart: "no"`; `network_mode: host`; `ipc: host`; `shm_size: 32g`; memlock
ulimits + `cap_add: IPC_LOCK`; `gpus: all`; `/dev/infiniband` device; a
healthcheck that exits 0 when `HEADLESS` is set, with a generous
`start_period`; volumes for the model dir / HF cache / cache root plus `:ro`
patch overlays; `CUDA_VISIBLE_DEVICES=0` (one GPU per node, TP=2 across nodes);
and a `command` block that auto-detects the RoCE v2 GID index from sysfs
(fail-closed — it renumbers across reboots), preflights the model dir, builds
the vLLM arg list, appends `--distributed-executor-backend mp --nnodes 2
--node-rank … --master-addr … --master-port …` (+ `--headless` on the worker),
and `exec vllm "$${ARGS[@]}"`.

**HF-cache invariant (never adopt upstream's cache/model paths):** the
host-side HF cache source MUST stay the spark user's default HF cache —
`HF_CACHE=/home/sdrew/.cache/huggingface`, the same location every other
recipe downloads into — regardless of how upstream lays out its model paths
(per-recipe model dirs, `DRAFT_MOUNT_DIR`, a relocated in-image `HF_HOME`,
etc.). Map upstream's model/drafter layout INTO that shared cache: download
with pinned revisions on BOTH nodes and resolve snapshot dirs at boot; do NOT
mount a separate per-recipe cache or repoint the host side of `HF_HOME`.
`HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` are mandatory in every recipe's
compose env — serving is offline by convention, and an upstream config that
expects runtime downloads is wrong for this cluster, not a knob to adopt.

If the checkpoint needs a surgery/adaptation step (quant-config rewrite,
drafter wiring), vendor it as an idempotent, fail-closed script in the recipe
dir and document it as README step 0.

### 6. Apply the standard defaults

These override whatever the upstream ran — unless the upstream value is
physically incompatible with the checkpoint/image, in which case keep
upstream's value and flag the deviation in README + research.md:

| setting | value |
|---|---|
| backend serve port | `PORT=8000` — repo invariant. Clients reach the model through the model-name proxy on **:4000** with model `spark-llm`; the proxy's `BACKEND_MODEL` must equal `SERVED_MODEL_NAME` |
| `SERVED_MODEL_NAME` | `glm-5.3-flash` for any GLM-5.3-Flash variant; `deepseek-v4-flash` for any DeepSeek-V4-Flash variant; otherwise a similar lowercase-hyphenated name derived from the parent model (e.g. `mimo-v2.5`) |
| temperature | `TEMPERATURE=1.0` |
| top_p | `TOP_P=0.95` |
| thinking | `THINKING=true` |
| reasoning effort | `REASONING_EFFORT=high` |

Pin them the way the reference recipe does — `--generation-config vllm` +
`--override-generation-config` + `--default-chat-template-kwargs` — so a
checkpoint/template refresh can't silently drift them. Use the model family's
equivalent mechanism where it differs (GLM: chat-template kwargs; DSv4: the
encoder's three-level reasoning-effort prompts).

### 7. Finish uncommitted — and stop

    git status    # recipe dir untracked on adopt-<recipe-name>; nothing staged

Report: branch name, directory, what was adopted from where (with receipts),
which defaults were overridden and why. Do NOT commit, push, PR, merge, build,
or launch.

## Common mistakes

- Committing (or pushing/PRing) the recipe — the deliverable is UNCOMMITTED.
- Writing changelog/watch material into README.md — that belongs in
  `research.md`; README is the active-running doc only.
- Copying upstream cluster IPs, NIC names, GMU/KV/backend values as-is —
  cluster wiring is THIS cluster's (AGENTS.md); tuning knobs are per-image and
  must come from the reviewed source for THIS image.
- Forgetting `entrypoint: []` → the command block is appended to the image's
  `vllm serve` argv and the container dies at argparse.
- Hard-coding a GID index instead of the sysfs auto-detect loop.
- Serving on any port other than 8000, or drifting `SERVED_MODEL_NAME` from
  the parent-model convention — both break the model-name proxy contract.
- Adopting a measured claim without a receipt (post number / commit / log line).
- Repointing the HF cache away from the spark user's default
  `/home/sdrew/.cache/huggingface` (upstream model-dir/`HF_HOME` layouts), or
  dropping `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` — all recipes share ONE
  host cache and serve offline (HF-cache invariant, step 5).
- Inventing hardware or a third node. Two fixed nodes; values in AGENTS.md.
