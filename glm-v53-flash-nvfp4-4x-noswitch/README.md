# glm-v53-flash-nvfp4-4x-noswitch — operations README

GLM-5.3-Flash **NVFP4** (the pristine [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4)
checkpoint @ `09b04e5e` — **zero weight modification**) served on the **4-node switchless
ConnectX-7 ring** with vLLM TP4, DFlash2 speculative decoding, 1M-token context, and the
site's existing patched NCCL. Adapted from the
[knapcio recipe](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4)
@ `770d115` (vendored at [upstream/VENDORED-AT.md](upstream/VENDORED-AT.md)).

**Why this lane exists:** the stock nvidia pack keeps its NVFP4 routed experts untouched
and its dense layers BF16; knapcio's lossless8 conversion (dense → 8-bit grids, 0.25%
output error) is deliberately NOT applied. Expected cost vs his switched-fleet numbers:
~10–15% decode (the −7.5 ms/step dense read) and ~9 GiB total KV; the ring transport
costs more (unquantified — bring-up's job to measure). If the result is too slow, the
fallback lane is the lossless8 conversion (see research.md §5).

**Endpoint (after bring-up):** `http://10.69.42.170:8000/v1` — model name
`glm-5.3-flash`, 1,048,576-token context, ~3.25M-token fp8 KV pool (20 GiB/rank pin →
~3.1× full-1M concurrency). No auth/TLS — trusted LAN only. GPU contention rule: the
FP8 E29 stack must be down (`tp4ctl down` in the glm-v53-flash-4x-noswitch recipe)
before this lane goes up.

## Cluster map

| rank | node | mgmt IP | role |
|---|---|---|---|
| 0 | `spark-0f0b` | 10.69.42.170 | leader — API server (`:8000`), `start.sh` runs here |
| 1 | `spark-6d14` | 10.69.42.171 | worker |
| 2 | `spark-6d90` | 10.69.42.172 | worker |
| 3 | `spark-6d24` | 10.69.42.173 | worker |

Ring fabric: 10.10.1.x–10.10.4.x links, MTU 9000, odd links LEFT QSFP (f0), even RIGHT
(f1). SSH from the workstation via `*.shawndo.intra`; rank 0 has a passwordless SSH mesh
to all nodes (same as the FP8 recipe).

## Locations

| path (from repo root) | what |
|---|---|
| `glm-v53-flash-nvfp4-4x-noswitch/upstream/` | vendored knapcio checkout @ `770d115`, **pristine** (zero upstream-file mods; additions listed in VENDORED-AT.md) |
| `glm-v53-flash-nvfp4-4x-noswitch/site/env.switchless` | the site `.env` (hosts, IPs, NCCL pin, paths) |
| `glm-v53-flash-nvfp4-4x-noswitch/site/materialize-weights.sh` | per-node weight materialization (hardlinks; see below) |
| `glm-v53-flash-nvfp4-4x-noswitch/upstream/profiles/stock-nvfp4.env` | site-added zero-mod serving profile (committed inside upstream — an addition, not a mod) |

On the nodes: `~/code/spark-recipes` on branch `glm-v53-flash-nvfp4-4x-noswitch` (all
four, same as the FP8 recipe); `~/.local/nvfp4-tp4/` = deployed overlay (`OVERLAY_REMOTE`)
+ materialized weights; `~/.local/lib/nccl-patched/` = the site NCCL rebuild shared with
the FP8 recipe; HF cache holds both checkpoints (nvidia `09b04e5e`, drafter `bf582e4e`).

## Bring-up (first time)

1. **Weights materialization (every node):**
   `bash glm-v53-flash-nvfp4-4x-noswitch/site/materialize-weights.sh`
   — hardlinks the HF-cache snapshots into `~/.local/nvfp4-tp4/weights/` (zero copy,
   zero modification) and guarantees `chat_template.jinja` exists in the model dir
   (start.sh requires it; falls back to the FP8 stack's zai template if the snapshot
   lacks it).
2. **NCCL staging (the highest-risk item):** the site NCCL build (`~/.local/lib/nccl-patched`,
   SHA `afe5f486…` — the josephdrose/SparkRing patch family the FP8 recipe proved on this
   exact fabric) implements ring-only tree/PAT skipping under `NCCL_SKIP_TREE_CONNECT`, NOT
   the four-PF build's `NCCL_SWITCHLESS_RING_ONLY`. The launcher therefore carries a
   **documented site mod** adding `-e NCCL_SKIP_TREE_CONNECT=1` to the switchless render
   (see `upstream/VENDORED-AT.md` "Site modifications" — re-apply on re-vendor). The staged
   first boot runs the FP8-proven **2-PF-per-node** fabric: the second PCIe views
   (`enP2p1s0f0np0`/`enP2p1s0f1np1`) stay MTU-only (no address) and `IB_HCA` is the two
   addressed PFs (`site/env.switchless`). Filename gotcha: `start.sh` + the artifact check
   + `LD_PRELOAD` all expect `libnccl.so.2.30.7`, but the site dir ships `libnccl.so.2` —
   create the in-dir symlink on every node first:
   `ln -s libnccl.so.2 ~/.local/lib/nccl-patched/libnccl.so.2.30.7`. The full four-PF
   contract (all four PFs per node, othexmr build @ `94c2669` rebuilt against
   `10.10.1–4.0/24`) is the Phase-E follow-up: it needs a one-line subnet-literal change
   in `nccl-2.30.7-dual-pf.patch` + rebuild + validation. Fallback if a boot stalls in
   NCCL init: FP8-recipe transport wiring (proven TP4 on this fabric with this lib).
3. **Image build (every node, from the upstream checkout):**
   `docker build -f Dockerfile.roce -t glm53-roce:v11-b58f34ea .` — minutes, no CUDA
   compile. Digest/base pins per upstream `docs/install.md`.
4. **Deploy + render check (on the head, from `upstream/`):**
   `cp ../site/env.switchless .env` (ignored file — tree stays pristine), then
   `DRY=1 ENV_FILE=.env ./start.sh serve` and read the four rendered docker commands.
5. **Serve:** `./start.sh serve` — launches ranks 3,2,1,0 (worker-first is upstream's
   job), waits out NCCL init + weight load + graph capture (expect a longer boot than
   the switched fleet's 3.6 min — JIT caches are cold on first run).
6. **Gates (within minutes of `/health` 200)** — upstream `bench/`:
   `prefix_scan.py` (mamba-align fix gate — must pass: `GLM_MAMBA_ALIGN_FIX=1` is in
   the inherited profile), the coherent-response + tool-call round-trip checks, and a
   corruption probe (emoji/CJK generation, count U+FFFD — must be 0; the nvidia pack
   scores 0/0/0 upstream). Then `accept_probe` / `conc_bench` for the baseline numbers.
7. **Record** the ring-vs-switched decode delta in `research.md` — that number decides
   whether this lane stays or we revisit lossless8.

## Switching between the FP8 and NVFP4 lanes

GPU contention rule: only one lane serves at a time. The lanes were built to coexist
without renames — switching is teardown + branch checkout:

- **No proxy change either way:** both lanes serve `glm-5.3-flash` on `:8000` behind the
  model-name-proxy (`BACKEND_MODEL=glm-5.3-flash` matches both); `SPOOF_FROM` already
  lists it.
- **No NCCL conflict:** both launchers use `~/.local/lib/nccl-patched` — the FP8 launcher
  preloads `libnccl.so.2` directly, this lane uses the `libnccl.so.2.30.7` symlink; both
  resolve to the same `afe5f486…` artifact. The FP8 stack's netplan (2 addressed PFs) is
  also this lane's staged transport — no fabric change when switching (until Phase E).
- **FP8 → NVFP4:** `tp4ctl down` in the FP8 recipe (`glm-v53-flash-4x-noswitch/upstream`)
  on the head; confirm `glm53_fp8_dflash_tp4` is gone on all 4; on all 4 nodes
  `cd ~/code/spark-recipes && git checkout glm-v53-flash-nvfp4-4x-noswitch && git pull`;
  cache ritual; `./start.sh serve` from this recipe's `upstream/`.
- **NVFP4 → FP8:** `./start.sh stop` here (stops, preserves containers for inspection —
  a relaunch of this lane then needs `docker rm` of the stopped `glm53-nvfp4-r*` first,
  upstream preflight refuses existing names); on all 4 nodes
  `git checkout glm-v53-flash-4x-noswitch && git pull`; then the FP8 stack's normal
  `tp4ctl up`.
- **Branches are per-lane:** the node checkout follows whichever lane is serving. `main`
  is never used on the nodes for either lane.

## Day-2 ops

- Start/stop/status/logs: upstream `./start.sh` (`serve`, stop via
  `scripts/stop_preserving.py` paths, `logs`). Teardown/recreate always cycles **all
  four ranks together** — never one rank.
- KV ladder: 20 GiB/rank is the v1 pin (~3.25M tokens). 24 → ~3.9M and 28 → ~4.5M are
  proven ceilings on identical hardware but MUST be gated per rung (head rank binds;
  "boots, then dies on first concurrent prefill" is the documented failure shape).
- Update procedure: [upstream/VENDORED-AT.md](upstream/VENDORED-AT.md). Any upstream
  `profiles/current.env` change needs a stock-compat review of new switches
  (research.md §3).
- Do not edit files on the nodes directly — recipe changes flow through the repo
  (topic branch → PR → main → pull on nodes), same as the FP8 recipe.

References: upstream [README](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4)
· [docs/switchless.md](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4/blob/main/docs/switchless.md)
· [docs/install.md](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4/blob/main/docs/install.md)
· sibling FP8 recipe `glm-v53-flash-4x-noswitch/` (fabric + NCCL provenance)
