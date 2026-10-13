# glm-v53-flash-nvfp4-tp4-switchless — operations README

GLM-5.3-Flash **NVFP4** (the pristine [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4)
checkpoint @ `da920bb` — **zero weight modification**) served on the **4-node switchless
ConnectX-7 ring** with vLLM TP4, DFlash2 speculative decoding, 1M-token context, and the
site's four-PF switchless NCCL build (`~/.local/lib/nccl-patched-4pf`, SHA `8de68def…`,
research.md "PHASE E COMPLETE"; the FP8 lane keeps its own `nccl-patched` lib — do not
merge the directories). Adapted from the
[knapcio recipe](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4)
@ `770d115` (vendored at [upstream/VENDORED-AT.md](upstream/VENDORED-AT.md)).

**Why this lane exists:** the stock nvidia pack keeps its NVFP4 routed experts untouched
and its dense layers BF16; knapcio's lossless8 conversion (dense → 8-bit grids, 0.25%
output error) is deliberately NOT applied. Expected cost vs his switched-fleet numbers:
~10–15% decode (the −7.5 ms/step dense read) and ~9 GiB total KV; the ring transport
costs more (measured 2026-10-10: c1 latency-bound at ~45–55% of the switched fleet,
recovered at concurrency — research.md changelog, "PHASE E COMPLETE"). If the result is
too slow, the fallback lane is the lossless8 conversion (see research.md §2 decision 5).

**Endpoint (after bring-up):** `http://10.69.42.170:8000/v1` — model name
`glm-5.3-flash`, 1,048,576-token context, ~3.25M-token fp8 KV pool (20 GiB/rank pin →
~3.1× full-1M concurrency). No auth/TLS — trusted LAN only. GPU contention rule: the
FP8 E29 stack must be down (`tp4ctl down` in the glm-v53-flash-fp8-tp4-switchless recipe)
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
| `glm-v53-flash-nvfp4-tp4-switchless/upstream/` | vendored knapcio checkout @ `770d115`, **pristine** (zero upstream-file mods; additions listed in VENDORED-AT.md) |
| `glm-v53-flash-nvfp4-tp4-switchless/site/env.switchless` | the site `.env` (hosts, IPs, NCCL pin, paths) |
| `glm-v53-flash-nvfp4-tp4-switchless/site/materialize-weights.sh` | per-node weight materialization (hardlinks; see below) |
| `glm-v53-flash-nvfp4-tp4-switchless/upstream/profiles/stock-nvfp4.env` | site-added zero-mod serving profile (committed inside upstream — an addition, not a mod) |

On the nodes: `~/code/spark-recipes` on branch `glm-v53-flash-nvfp4-4x-noswitch` (all
four, same as the FP8 recipe); `~/.local/nvfp4-tp4/` = deployed overlay (`OVERLAY_REMOTE`)
+ materialized weights; `~/.local/lib/nccl-patched-4pf/` = this lane's four-PF NCCL build
(SHA `8de68def…`; the FP8 lane's `~/.local/lib/nccl-patched/` is a DIFFERENT artifact —
keep both); HF cache holds both checkpoints (nvidia `da920bb` (+ prior `09b04e5e`),
drafter `bf582e4e`).

## Bring-up (first time)

1. **Weights materialization (every node):**
   `bash glm-v53-flash-nvfp4-tp4-switchless/site/materialize-weights.sh`
   — hardlinks the HF-cache snapshots into `~/.local/nvfp4-tp4/weights/` (zero copy,
   zero modification) and guarantees `chat_template.jinja` exists in the model dir
   (start.sh requires it; falls back to the FP8 stack's zai template if the snapshot
   lacks it).
2. **NCCL + netplan staging (the highest-risk item):** this lane ships its own four-PF
   NCCL build and fabric netplan (Phase E, `site/nccl-4pf/` + `site/netplan/<node>/`):
   stage `~/.local/lib/nccl-patched-4pf/libnccl.so.2.30.7` (real file; SHA pinned in
   `site/env.switchless` — the launcher's preflight enforces it) with an in-dir
   `libnccl.so.2 → libnccl.so.2.30.7` symlink, and install the per-node netplan
   (`sudo cp site/netplan/<node>/40-cx7.yaml /etc/netplan/40-cx7.yaml && sudo netplan
   apply`). The netplan addresses BOTH PCIe views of every cable (f-port octet + 2 on the
   P2 views) — additive over the FP8 plan, so the FP8 lane keeps working, but **never
   re-deploy the FP8 recipe's 2-address netplan here** (it strips the P2 addresses).
   All four PFs must carry QPs per rank at boot (`Connected all rings` + RouteFinal
   lines). Rollback if a boot stalls in NCCL init: point `NCCL_HOST_DIR` back at
   `~/.local/lib/nccl-patched` and re-apply the staged-transport mods documented in the
   `upstream/scripts/transport.sh` comment (FP8-proven 2-PF wiring).
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

> **Runbook: [`../../LANE-SWITCHING.md`](../../LANE-SWITCHING.md)** — the one-doc
> procedure (preflight, both directions, verification, gotchas). The notes below are
> the lane-specific background.

GPU contention rule: only one lane serves at a time. The lanes were built to coexist
without renames — switching is teardown + branch checkout:

- **No proxy change either way:** both lanes serve `glm-5.3-flash` on `:8000` behind the
  model-name-proxy (`BACKEND_MODEL=glm-5.3-flash` matches both); `SPOOF_FROM` already
  lists it.
- **One NCCL build now serves both lanes (FP8 4-PF adoption, 2026-10-10):** this lane
  `~/.local/lib/nccl-patched-4pf/` (`8de68def…`, hardened four-PF) — the FP8 stack's
  `cluster.env` was converted to the same four-PF contract on this branch (its rollback to
  the legacy 2-PF build `~/.local/lib/nccl-patched/` (`afe5f486…`) is documented at
  `NCCL_IB_HCA` in its `cluster.env`, coupled: HCA + NCCL_DIR + EXTRA_DOCKER_ENV append).
  Keep both dirs staged; the fabric netplan (all 4 PCIe views addressed) serves both lanes.
  Preflight before any switch:
  `bash glm-v53-flash-fp8-tp4-switchless/noswitch-prep/scripts/verify-fabric-p2.sh` (read-only,
  safe while serving).
- **FP8 → NVFP4:** `tp4ctl down` in the FP8 recipe (`glm-v53-flash-fp8-tp4-switchless/upstream`)
  on the head; confirm `glm53_fp8_dflash_tp4` is gone on all 4; on all 4 nodes
  `cd ~/code/spark-recipes && git pull` (both lanes share this branch);
  cache ritual; `./start.sh serve` from this recipe's `upstream/`.
- **NVFP4 → FP8:** `./start.sh stop` here (stops, preserves containers for inspection —
  a relaunch of this lane then needs `docker rm` of the stopped `glm53-nvfp4-r*` first,
  upstream preflight refuses existing names); on all 4 nodes
  `git pull`; then the FP8 stack's normal
  `tp4ctl up`.
- **Branches are per-lane:** the node checkout follows whichever lane is serving. `main`
  is never used on the nodes for either lane.

## Day-2 ops

- Start/stop/status/logs: upstream `./start.sh` (`serve`, stop via
  `scripts/stop_preserving.py` paths, `logs`). Teardown/recreate always cycles **all
  four ranks together** — never one rank.
- **Warm reboots:** the JIT/TRITON/tilelang caches live in the per-node overlay root
  (`~/.local/nvfp4-tp4/`). Relaunching against the same `OVERLAY_REMOTE` keeps them
  (warm `/health` ~3–4 min vs ~7 min cold, per tonyd2wild's measurements on the same
  stack). If a change ever forces a FRESH overlay root (new `CTN`/`OVERLAY_REMOTE`),
  copy the old overlay's `cache/` into the new root first — that alone reproduces the
  warm-boot time.
- **After a node power-cut / unclean host reboot, serve fresh — never `docker start`**
  (tonyd2wild's ops rule): the stopped containers from a power-cut predate the boot and
  may reference stale overlay/mount state; recreate them with `./start.sh serve`
  (`docker rm` the stopped `glm53-nvfp4-r*` first — preflight refuses existing names).
  The normal relaunch path already recreates containers, so this only matters when
  reaching for `docker start` by hand.
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
· sibling FP8 recipe `glm-v53-flash-fp8-tp4-switchless/` (fabric + NCCL provenance)
