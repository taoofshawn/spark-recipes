# AGENTS.md

Operational guide to this repository for agentic/automated contributors. Read this first; it
condenses what took this project's maintainers weeks of forum mining, cluster debugging, and
repeated deploy cycles to learn.

## What this repository is

A collection of **self-contained recipes for serving LLMs on a 2-node DGX Spark (GB10) cluster**
via vLLM, with DSpark speculative decoding, at 1M-token context. Every recipe is a drop-in
package: config files, a launch mechanism, and the runtime patches/overlays needed to make the
specific model work on this specific hardware.

Two models are covered:

- **DeepSeek-V4-Flash-0731** (`deepseek-ai/DeepSeek-V4-Flash-0731`) — the main workhorse.
  Served as `deepseek-v4-flash` on port `4000`.
- **MiMo-V2.5 + DFlash** (Xiaomi) — a second, independent recipe. Not part of the DeepSeek
  cluster of recipes.

The upstream source of truth for most of this is the
[NVIDIA DGX Spark forum](https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10/719)
plus a few community GitHub repos. Recipes are vendored/backported from those; links and
attribution are in each recipe directory's README.

## Cluster topology (two fixed nodes)

This repo is deployed on a specific pair of nodes. Do not invent other hardware in commits.

| role | host | RoCE IP | notes |
|---|---|---|---|
| node 0 (leader, rank 0, API server) | `spark-0f0b.shawndo.intra` | `192.168.0.170` | also called "head" |
| node 1 (follower, rank 1, headless) | `spark-6d14.shawndo.intra` | `192.168.0.171` | also called "worker" |

- Both nodes must have the repo checked out at `~/code/spark-recipes` and the model cached in
  `~/.cache/huggingface` (serving is **offline**: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`).
- Inter-node networking: **RoCE/InfiniBand** for the NCCL data plane
  (`IB_PORTS=rocep1s0f0,roceP2p1s0f0`) and **Ethernet** for control plane
  (`ETH_IF=enp1s0f0np0`, `ETH_IF2=enP2p1s0f0np0`). These are the actual NIC names on this
  cluster; the files always say "match YOUR NICs" because they were originally written for
  other hardware — here the committed values ARE correct.
- Each node has 2 GPUs; recipes use `CUDA_VISIBLE_DEVICES=0` (one GPU per node, TP=2 across
  nodes).
- The RoCE IPv4 **GID index is auto-detected at container boot** (it renumbers across
  reboots). This detection loop is duplicated in each docker-compose `command` block — keep it
  consistent across recipes.
- **GPU contention rule:** both DeepSeek-V4 recipes use all reserved GPUs. Tear down any
  currently-running model container before starting another. Only one recipe may serve at a time.

## Repository layout

```
README.md                       # short index; stable recipes in main, in-progress on branches
deepseek-v4-flash-aiden/                    # docker-compose recipe (the "reference" compose)
deepseek-v4-flash-aiden-sparkrun/           # sparkrun port of aiden (no rebuild, docker-pull)
deepseek-v4-flash-eugr-sparkrun/            # sparkrun port of eugr's B12X source build
deepseek-v4-flash-tonyd2wild/               # docker-compose recipe (NVFP4 DS-MLA KV stack)
  └── upstream/                             # VENDORED upstream repo — do NOT edit (see below)
mimo-v25-dflash-tonyd2wild/                 # docker-compose recipe (MiMo-V2.5 + DFlash)
```

### The three DeepSeek recipes: pick the right one

All three serve the same model/port (deepseek-v4-flash :4000, 1M context) but differ in HOW the
runtime is obtained and which backend flags apply. **Flags and env vars are NOT interchangeable
between them** — each belongs to a specific image/build; mixing them fails at startup.

| recipe | mechanism | image / backend | status notes |
|---|---|---|---|
| `deepseek-v4-flash-aiden` | docker-compose | prebuilt `aidendle94/sparkrun-vllm-ds4-gb10` (digest-pinned, "3.75"), `FLASHINFER_MLA_SPARSE_DSV4` attention | the battle-tested baseline; most documented |
| `deepseek-v4-flash-aiden-sparkrun` | sparkrun recipe (`.yaml`) | same aiden image, `builder: docker-pull` (critical — without it sparkrun picks the eugr source builder) | sparkrun-native management, no rebuild |
| `deepseek-v4-flash-eugr-sparkrun` | sparkrun recipe | `vllm-node-b12x` built from source via eugr (`build_args: --exp-b12x`), `B12X_MLA_SPARSE` attention | needs a ~20–40 min source build on first run; repo's only source-built recipe |
| `deepseek-v4-flash-tonyd2wild` | docker-compose | locally built `vllm-dspark-runtime:dspark-nvfp4-stage-c` (4-stage overlay), `nvfp4_ds_mla` KV | most patched; full from-scratch build guide in its README |

`aiden-sparkrun` is the same as `aiden` but managed through sparkrun (cluster abstraction). If a
change targets one, the other usually needs the equivalent change.

## How the recipes work

### docker-compose recipes (`deepseek-v4-flash-aiden`, `...-tonyd2wild`, `mimo-v25-dflash-tonyd2wild`)

- **`.env`** = shared config (`MASTER_ADDR`, `PORT=4000`, `ETH_IF`, `ETH_IF2`, `IB_PORTS`).
- **`.env.node0` / `.env.node1`** = per-node overrides (`NODE_RANK=0|1`, `HEADLESS=1` on worker,
  `ROCE_IP`).
- `docker-compose.yml` = image + env + a big `command` block that, at boot: auto-detects the
  NCCL GID index → applies runtime mods/patches inside the container → starts Ray (mimo only) →
  launches `vllm serve`.
- **Start order: worker (node 1) FIRST, then leader (node 0) ~30–35 s later.** This is
  repeatedly emphasized; the multi-node TCP store (`master_addr:25000`) must be up before the
  follower connects.
- Verify: `curl http://127.0.0.1:4000/v1/models` should show `"id":"deepseek-v4-flash"` and
  `"max_model_len":1048576`.

### sparkrun recipes (`deepseek-v4-flash-aiden-sparkrun`, `deepseek-v4-flash-eugr-sparkrun`)

- Single `.yaml` with `defaults` (templating values), `env`, `command`, and `executor_config`.
- Run from the leader: `sparkrun run <path>.yaml` (dry-run with `-n`, force rebuild with
  `--force-build`, watch `sparkrun logs <id>`, stop with `sparkrun stop <id>`).
- **Templating gotcha:** sparkrun substitutes `{key}` from `defaults` as a single unit, so
  JSON-valued flags (`speculative_config`, `compilation_config`, …) must live as single
  `defaults` values used as `'{key}'` — not eugr-style `{{...}}` escapes.
- **`model_revision` must appear in BOTH the top-level field and `defaults`** (top-level drives
  model-distribution lookup; only `defaults` keys get `{placeholder}`-substituted).
- The recipe yaml files carry lengthy explanatory comments — they are the best in-file
  documentation of WHY things are set the way they are.

## The recurring overlays: why the same fix files appear in every DSv4 recipe

DeepSeek-V4-Flash-0731's **reasoning-effort levels and tool-call handling are broken in every
bundled vLLM encoder** found in the wild (pre-0731 encoders collapse `low`→`high`, assert on
`low`, or emit malformed tool-arg JSON). Every recipe ships a fix in one of two forms:

1. **Bind-mount overlays** (docker-compose recipes): files in the recipe dir are mounted
   `:ro` directly OVER files inside the image's vLLM install, e.g.
   `./encoding_dsv4.py:/opt/venv/lib/python3.12/site-packages/vllm/tokenizers/deepseek_v4_encoding.py:ro`.
2. **mods** (sparkrun recipes): a `mods/<name>/run.sh` that `cp`'s the same fix files into the
   container before `vllm serve`.

The fixed files are named the same across recipes and should stay in sync:

| file | what it fixes |
|---|---|
| `encoding_dsv4.py` | official 0731 three-level reasoning-effort prompts (`low`= "", `high`="Absolute maximum…", `max`="Beyond maximum…") + tool-argument JSON repair |
| `deepseek_v4_wrapper.py` | tokenizer wrapper routing: `low`→low, `high`→high, `max`/`xhigh`→max, `none`/`off`→chat |
| `detokenizer.py` | Open-PR backport: don't evaluate client `stop` strings inside the reasoning segment (prevents silent `content:null` answers) |
| `deepseekv32_tool_parser.py` | (tonyd2wild only) streaming tool-parser fix — suppress spurious `"tool_calls": []` on content deltas |

Tool-arg hardening functions to grep for: `normalize_tool_arguments`, `repair_tool_arguments_json`,
`parse_tool_arguments`, `dsml_param_to_python`. If you touch `encoding_dsv4.py` or
`deepseek_v4_wrapper.py`, update ALL recipes that carry copies.

**Image path differences matter:** aiden 3.75/3.7 keep vLLM at `/opt/venv/lib/python3.12/...`;
production-3.8 and the tonyd2wild/ eugr images use `/opt/env/...`. Every overlay/mod/executor
config hard-codes one of these. Wrong path = overlay silently not applied.

## Key technical vocabulary

- **DSpark** — NVIDIA's speculative-decoding method used everywhere here
  (`--speculative-config '{"method":"dspark",...}'`). `num_speculative_tokens` (a.k.a.
  `MTP_NUM_TOKENS`, k) is usually 4–5; **k ≤ 5 or a multiple of 5** (the drafter emits exactly 5
  per pass). `draft_sample_method=probabilistic` is in the recipes but is largely a no-op for
  DSpark (reads equal on the `NO_DRAFT_PROBS` path) — don't "fix" it.
- **B12X** — the custom fused-MoE / linear / sparse-MLA backend family (Blackwell-optimized,
  NVFP4). `VLLM_USE_B12X_MOE=1` and `—moe-backend b12x` are the speed-critical switches;
  removing them silently tanks decode to ~29 tok/s (falls back to DEEPGEMM_MXFP4). Boot-log
  marker: `Using 'B12X' Mxfp4 MoE backend`.
- **NVFP4 / fp8 KV caches** — `kv-cache-dtype nvfp4_ds_mla` (tonyd2wild) vs `fp8` (aiden/eugr).
  These flags are image-specific.
- **gpu_memory_utilization / GPU_MEM** — tuned per stack: aiden `0.83` (a KV lever, don't
  lower lightly), tonyd2wild ≤ `0.78` (`0.80` "boots-then-dies"), eugr `0.85`, mimo `0.83`.
  Values are NOT transferable between recipes.
- **GMU is a KV-cache lever, and ~155 GiB of weights are the floor** — lowering it shrinks the
  KV pool and can break 1M context; but if a server crashes under traffic, lowering GMU (e.g.
  to 0.78) is the documented first knob.
- **max_num_seqs** — concurrency cap; 6 is the validated agent-serving value for DSv4 (12 is
  "riskier"; mimo uses 6). **max_num_batched_tokens** — the newer aiden-sparkrun profile uses
  2048 (better decode fairness under agent traffic; the older aiden compose still ships
  16384), tonyd2wild 8192, eugr 10240. Not common between recipes.
- **AOT/JIT** — `VLLM_USE_AOT_COMPILE=1`; cold starts recompile (~7–8 min boot) because only
  the HF cache is persisted as a volume. Warm-up penalty: fresh boot is ~30% slower until a few
  hundred tokens of traffic pass; never benchmark right after boot.
- **Spec-decode benchmarking caveat:** use `stream:false` and read
  `usage.completion_tokens`; streamed deltas measure steps/s, not tok/s (up to ~4× under-report).

## Vendored upstream — `deepseek-v4-flash-tonyd2wild/upstream/`

This directory is a **full, unmodified copy** of tonyd2wild's upstream repo, pinned at a commit.
- **Do not edit files under `upstream/` for local customization** — that lives in the parent
  recipe's `docker-compose.yml` / `.env` / README.
- Refresh procedure (VENDORED-AT.md): re-clone or `git fetch` and checkout the new commit, then
  re-verify that **Patch 4** (`shared_experts.gate_up_proj` lines in `dspark.py`) is still baked
  in, then rebuild.
- **Patch 4 matters:** without it, 0731 decode roughly halves (acceptance drops to ~26%).
  Verify with `grep -n "shared_experts.gate_up_proj"` on the built image.

## Git / workflow conventions

- `main` = stable recipes only. **In-progress recipe work goes on branches** (per README). The
  remote currently also has `aiden-aug15-updates` (unmerged tuning refresh: k=5 + greedy draft,
  GMU 0.83).
- Commit messages are descriptive one-liners; PRs merge topic branches into `main` (see git
  log history: `Deepseek v4 flash tonyd2wild (#3)`, `... aiden sparkrun (#9)`, etc.).
- Recipe directories document their own changelog inline (the tonyd2wild README has a
  dated "audit trail" section; aiden README has an "upgrading to production-3.8" revert
  write-up). Keep that pattern — it is how regressions get explained later.
- `.gitignore` ignores `.worktrees/`, `__pycache__`, `*.pyc`. No CI, no tests, no tooling in the
  repo. The "tests" are curl health checks and boot-log markers documented in each README.
- One inconsistency to know about: some READMEs still say `git checkout <recipe-branch>` (e.g.
  `deepseek-v4-flash-aiden-sparkrun`, `deepseek-v4-flash-tonyd2wild`), but those branches were
  merged into `main` and deleted. **Everything lives on `main` now** (plus
  `aiden-aug15-updates`); `git checkout main` is correct.

## Typical maintenance workflow & where to look for updates

This repo is a rolling forward-port of community work. The standing task — "review recipe X,
adopt anything new from its sources" — is a research pass followed by an evaluate-and-merge
pass. This section tells you exactly where to look and how to query each source programmatically.
The sources below are ranked by how much they have historically mattered for these recipes.

### 1) The NVIDIA Developer Forum (forums.developer.nvidia.com) — highest-value source

It runs **Discourse**, so it exposes a public JSON API (no auth required for reads). The repo's
root README already calls this forum "the source for most/all of this information."

**Relevant categories** (the exact slugs/IDs below are verified live):

| category | URL / JSON path | ID |
|---|---|---|
| DGX Spark / GB10 User Forum (parent) | `/c/accelerated-computing/dgx-spark-gb10` → `.json` | 719 |
| DGX Spark / GB10 (main board) | `/c/accelerated-computing/dgx-spark-gb10/dgx-spark-gb10/721.json` | 721 |
| DGX Spark / GB10 Projects (recipes) | `/c/accelerated-computing/dgx-spark-gb10-projects/723.json` | 723 |
| DGX Spark / GB10 Announcements | (`/c/...` under 719) | 722 |

**API recipes an agent should use:**

```python
# Latest topics in the DGX Spark boards
GET https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10/721.json?page=0&order=activity
GET https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10-projects/723.json

# Full-text search (Discourse syntax: after:<date>, category:<id>, order:latest, #tag)
GET https://forums.developer.nvidia.com/search.json?q=deepseek%20v4%20flash
GET https://forums.developer.nvidia.com/search.json?q=deepseek%20v4%20flash%20after:2026-08-01
GET https://forums.developer.nvidia.com/search.json?q=dspark%20category:721%20order:latest

# A single thread's posts (20 per page; paginate with ?page=1,2,...). Use .json after the numeric id.
GET https://forums.developer.nvidia.com/t/372268.json
GET https://forums.developer.nvidia.com/t/372268.json?page=2

# Newest posts forum-wide (monitor for replies to tracked threads)
GET https://forums.developer.nvidia.com/posts.json
```

Send a browser-ish `User-Agent` header; the API is plain, unauthenticated JSON. `requests` is fine.
To keep up with a thread you already follow: fetch `/t/<id>.json`, read `posts_count`, and page
until you have the posts newer than your last check (each post has `post_number` and `created_at`).

**The threads that matter for these recipes** (IDs are stable; titles verified):

| thread ID | why it matters |
|---|---|
| `372268` | "DeepSeek v4 Flash (Aiden Recipe from Reddit), 1M token ses…" — **the aiden recipe parent thread**; 700+ posts, still active. Source of the aiden image, encoder/tool-arg fixes, GMU notes. |
| `376220` | "Instructions for running Deepseek-v4-flash with DSpark using Eugr's repo" (post 18 = bernisse's B12X solution) — parent of the eugr recipe. |
| `378824` | "DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark" — **tonyd2wild's own announcement thread**; patch/version updates land here. |
| `378890` | "Agent Serving on 2× DGX Spark with DeepSeek V4 Flash 0731: KV-cache, …" — the agent-serving tuning profile (smaller prefill chunks). |
| `379863` / `376884` | 1×/2× Spark DSpark tuning, tok/s claims, acceptance metrics. Useful cross-checks when a number in a README looks stale. |

Search terms worth running on every review: `deepseek v4 flash`, `dspark`, `b12x`, `nvfp4`,
`sparkrun`, `mimo v2.5`, `dflash`. Also check `category:721 after:<last-review-date>` — new
topics in the DGX Spark board — because recipe-relevant info frequently lands in threads whose
titles do not contain "deepseek".

### 2) tonyd2wild's GitHub repos — direct source for the tonyd2wild recipe

**Primary:** `tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark` —
fully vendored in `deepseek-v4-flash-tonyd2wild/upstream/` at a pinned
commit (see `upstream/VENDORED-AT.md`). Review pass:

```bash
# What changed upstream since our vendored pin?
git ls-remote https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark HEAD
# diff the pinned commit vs HEAD (sample it via the GitHub API or a temp clone)

# Open PRs, issues, and recent commits (GitHub API, no auth needed for public repos, 60 req/h)
GET https://api.github.com/repos/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark/pulls?state=open
GET https://api.github.com/repos/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark/issues?state=open
GET https://api.github.com/repos/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark/commits?per_page=20
```

The README's own changelog ("2026-08 improvements") shows the audit this repo already did
against upstream: watch for new **patches** (`patches/` dir), new **docs** (root-level `*.md`
update notes), and **benchmark/script** changes. Whenever upstream bumps a fix, re-check that
the corresponding overlay in our recipe (`encoding_dsv4.py`, `deepseek_v4_wrapper.py`,
`detokenizer.py`, `deepseekv32_tool_parser.py`) still matches.

**Sibling repos by the same author matter too.** Several times a missing module or patch was
found in a *different* tonyd2wild repo. His DGX-Spark-serving family includes
`DeepSeek-v4-Flash-DSpark-60-tok-s-900K-ctx-2x-DGX-Spark`, `DeepSeek-v4-Flash-0731-Vision-DSpark-1M-NVFP4-KV-2x-DGX-Spark`,
`DeepSeek-v4-Flash-DSpark-Abliterated-*-2x-DGX-Spark`, `MiMo-V2.5-*-DGX-Spark`, plus tooling
repos (`The-Sparky-Command-Center`, `2Wild-Coding-Agent-Latency-Monitor`). The tonyd2wild
README's troubleshooting table explicitly tells agents to search "his other public repos first"
when a module is missing.

### 3) eugr's repo — direct source for the eugr recipe + sparkrun semantics

**`eugr/spark-vllm-docker`** (branch `b12x`) is where the eugr recipe's runtime and build come
from. It is also where the **sparkrun** recipe-format semantics originate (the repo is 450+ stars
and very active — multiple commits and PRs per week). Check its open PRs and issues too; several
have already fed this repo's recipe yaml (e.g. JSON-valued-arg substitution fixes, new DSv4
recipes, `--apply-vllm-pr` support). `eugr/llama-benchy` is his benchmark suite (useful for
verifying throughput claims). Search: `GET /repos/eugr/spark-vllm-docker/pulls?state=open`.

Also relevant: **`spark-arena/sparkrun`** (the sparkrun CLI/tool itself, docs at sparkrun.dev)
and third-party sparkrun recipe registries (`styles01/sparkrun-recipes`,
`brainchillz/sparkrun-dspark-registry`) — good for spotting new flags or config patterns.

### 4) Container images and model checkpoints — the "what changed" sources

- **Aiden image:** `aidendle94/sparkrun-vllm-ds4-gb10` on Docker Hub. Check new tags:
  `GET https://hub.docker.com/v2/repositories/aidendle94/sparkrun-vllm-ds4-gb10/tags?page_size=25`.
  Our recipes pin `production-3.75` by digest; the README documents exactly why `production-3.8`
  (vLLM 0.21.1) was reverted — `production-hybrid-*` tags have since appeared, so review those
  when they land. An image bump cascades through: digest, overlay mount paths (`/opt/venv` vs
  `/opt/env`), `HF_HOME`, attention-backend env names, and `VLLM_USE_V2_MODEL_RUNNER`.
- **Model checkpoint:** `deepseek-ai/DeepSeek-V4-Flash-0731` on HuggingFace (pinned revision
  `9e165c30…`). Check for a newer revision:
  `GET https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4-Flash-0731` (compare
  `lastModified`; look at the `siblings` to see if newer `encoding/encoding_dsv4.py` exists).
  Any model update must be mirrored in `MODEL_REVISION`/`model_revision` in every recipe and
  re-verified against the newer official encoder.
- **vLLM upstream:** DSpark work crosses into `vllm-project/vllm` (e.g. the merged DSpark PR the
  upstream `UPSTREAM_V024_STATUS.md` tracks). Only adopt upstream vLLM changes if the newer
  runtime boots on GB10/SM120 with the custom backends — stock vLLM often cannot (see the same
  doc).

### 5) Running the review: an end-to-end loop

1. **Snapshot current state:** `git log -5`, note `upstream/VENDORED-AT.md` pin, note which
   overlays each recipe carries.
2. **Forum pass:** search `deepseek v4 flash` + `dspark` (+ new-topics sweep of cat 721 since
   the last scan). Pull each returned thread's newest posts. Look for: new image tags, new
   patches, new tuning knobs with measured numbers, and regression reports (especially the ones
   that could hit this cluster's config).
3. **GitHub pass:** for tonyd2wild primary + the eugr repo, list open PRs/issues + recent
   commits; grab the diff of anything touching DSpark/encoder/loader/batching.
4. **Image/model pass:** check Docker Hub tags for aiden, HF model `lastModified`/siblings.
5. **Evaluate before adopting:** every candidate must (a) apply to our pinned image/revision,
   (b) not contradict the recipe's validated invariants (start order, port 4000, offline serving,
   GMU/KV dtype/backend wiring), and (c) be backed by a measured claim. Reject "in theory"
   improvements that touch the fragile cross-recipe knobs.
6. **Adopt surgically:** recipe-level changes go in the recipe dir (not `upstream/`); overlays
   copied between recipes stay in sync; document why in the commit message like the existing
   history does.

### 6) Reporting conventions

Follow the repo's existing style when you make changes: terse PR/commit titles with a `(#N)`
PR number (`... sparkrun (#9)`), and for notable tuning work add a dated changelog block in the
recipe README (the tonyd2wild "2026-08 improvements — audit trail" section is the template) that
states what changed, the measured before/after, and any gotchas hit.

## Common failure modes (from real deployments; all in READMEs)

| symptom | cause / fix |
|---|---|
| `DistStoreError: 1/2 clients` / `Connection reset by peer` on restart | wrong start order — leader (rank 0) first so TCP store on `:25000` is up |
| model "not found" offline | HF cache missing on the worker (TP reads snapshot on both nodes) or `HF_HOME`/`HF_CACHE` mismatch (3.8 image moved it to `/cache/huggingface`) |
| `./build-dspark-vllm-runtime.sh: Permission denied` | missing exec bits — `chmod +x upstream/*.sh upstream/scripts/*.sh` (fixed in committed tree, breaks on old clones) |
| `WORKER_HOST: must be set in …/.env.dspark` | build env missing — `upstream/.env.dspark` (committed copy exists) |
| decode ~half speed / acceptance ~26% | Patch 4 missing from the baked image |
| `ValueError: DSpark currently requires uniform effective per-request target context lengths` (vLLM 0.21/3.8) | DSpark regression in newer vLLM with mixed-length batches — this is why aiden stays on vLLM 0.11 (3.75) |
| `No available shared memory broadcast block found in 60 seconds` | benign — other rank compiling; resolves |
| `min_p and logit_bits won't work with speculative decoding` | expected under DSpark |
| `SymmMemCommunicator: Device capability 12.1 not supported` | expected on GB10; falls back to PYNCCL |
| docker container runs as uid 1000, cache root-owned | `sudo chown -R 1000:1000 <hf-cache>` |

## Agents: do / don't

- **Do** read each recipe's README before touching it — they carry the hard-won constraints.
- **Do** keep the reasoning-effort/tool-arg fix files in sync across all recipes that carry
  them, and respect the `/opt/venv` vs `/opt/env` split.
- **Do** keep start-order, port-4000, and offline-serving conventions intact.
- **Do** document the "why" in new commits the way this repo already does (inline comments,
  dated change notes, measurement claims with numbers).
- **Don't** reorder or "tidy" env vars / flags that look duplicated — they are image-specific;
  cross-recipe consistency is not the goal.
- **Don't** edit `upstream/` vendored files; recipe-level files are the customization layer.
- **Don't** change `gpu_memory_utilization`, `max_num_seqs`, KV dtype, or backend names across
  recipes as if they were shared knobs — each was tuned for its image.
