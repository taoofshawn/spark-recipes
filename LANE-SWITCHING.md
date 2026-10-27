# Lane switching runbook — FP8 ↔ NVFP4 ↔ DeepSeek-V4.1-Flash on the 4-node switchless ring

The one doc a fresh agent needs to switch between the serving lanes without reading all
recipes end-to-end. Deep context lives where noted; this file is the procedure.

**State this was verified against:** 2026-10-10, both GLM lanes boot and switch cleanly
on the four-PF NCCL contract (T1/T2 records:
`glm-v53-flash-fp8-tp4-switchless/research.md` §5 and
`glm-v53-flash-nvfp4-tp4-switchless/research.md` Changelog). The DeepSeek-V4.1-Flash
lane is ADOPTED 2026-10-17 but NOT YET brought up — its switch sections are written
from the adopted recipe, unverified on this fabric until first bring-up (its
research.md changelog records the verification when it happens).

| | FP8 lane | NVFP4 lane | DeepSeek-V4.1-Flash lane |
|---|---|---|---|
| recipe dir | `glm-v53-flash-fp8-tp4-switchless/` | `glm-v53-flash-nvfp4-tp4-switchless/` | `deepseek-v41-flash-tp4-switchless/` |
| weights | zai FP8 checkpoint (lossless) + lossless8-style conversion stack (E36) | stock `nvidia/GLM-5.3-Flash-NVFP4` @ `09b04e5e`, zero weight mod | stock `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e10`, zero weight mod (MIT) |
| context / pool | 262K / ~1.19M-token KV | 1M / ~3.25M-token KV (20 GiB/rank pin) | 1M / free-memory-sized (record actual pool at boot) |
| control | `tp4ctl` (`up`/`down`/`restart`/`health`) on the head | `./start.sh` (`serve`/`stop`/`status`/`logs`) on the head | `./start-cluster.sh` (`check`/`serve`/`stop`/`status`/`logs`) on the head |
| containers | `glm53_fp8_dflash_tp4` | `glm53-nvfp4-r0..r3` | `deepseek-v41-flash-r0..r3` |
| NCCL | **shared**: `~/.local/lib/nccl-patched-4pf` (`8de68def…`, four-PF); legacy 2-PF lib `~/.local/lib/nccl-patched` (`afe5f486…`) is the FP8 rollback artifact — keep both dirs | same shared four-PF lib | same shared four-PF lib (this lane's own rollback = sparkring 2-PF donor lib — see its VENDORED-AT.md) |

Shared facts that make switching simple:

- **One branch serves the active recipes** (the per-lane working branch carries the GLM
  lanes AND the DeepSeek-V4.1-Flash lane). No branch checkout at switch time — `git
  pull` on the nodes is enough.
- **One fabric netplan** (all four PCIe views addressed) and **one NCCL lib** — nothing
  network-level to decide at switch time. All three lanes run the shared four-PF
  contract.
- **Proxy change only when the served NAME changes**: both GLM lanes serve
  `glm-5.3-flash` on `:8000` behind the model-name-proxy (no proxy edit between them);
  the DeepSeek lane serves `deepseek-v41-flash` — switching to/from it is a one-line
  `BACKEND_MODEL` change (+ `SPOOF_FROM` entry) in `model-name-proxy/.env` + proxy
  recreate.
- **One lane at a time** (GPU contention): every recipe uses all reserved GPUs.

## Preflight — run before EVERY switch (cheap, read-only, safe while serving)

```bash
bash glm-v53-flash-fp8-tp4-switchless/noswitch-prep/scripts/verify-fabric-p2.sh
```

Must print `ALL CHECKS PASS`. It gates on the netplan generator, compares live
`/etc/netplan/40-cx7.yaml` vs generated on all 4 nodes, checks all 16 interface
addresses, runs the 16-ping two-view mesh, and SHA-verifies both staged NCCL libs.
Failures: resolve before booting anything (the failure text names the fix).

## Switch NVFP4 → FP8

```bash
# 1. Maintenance window. On the HEAD node, NVFP4 upstream:
cd ~/code/spark-recipes/glm-v53-flash-nvfp4-tp4-switchless/upstream && ./start.sh stop
#    (stops + PRESERVES containers; confirm glm53-nvfp4-r* gone/stopped on all 4 —
#     `docker rm` the stopped ones if you plan to relaunch NVFP4 later, its preflight
#     refuses existing names)

# 2. Preflight (above) — ALL PASS.

# 3. On ALL FOUR nodes:
cd ~/code/spark-recipes && git pull

# 4. ONLY if a repo config change is pending (cluster.env etc.): from the HEAD node
#    (NOT the workstation — its Windows python3 stub breaks deploy):
cd ~/code/spark-recipes/glm-v53-flash-fp8-tp4-switchless/upstream && bash scripts/deploy.sh

# 5. Head node:
cd ~/code/spark-recipes/glm-v53-flash-fp8-tp4-switchless/upstream && tp4ctl up
#    (upstream handles worker-first ordering)
```

## Switch FP8 → NVFP4

```bash
# 1. Head node, FP8 upstream:
cd ~/code/spark-recipes/glm-v53-flash-fp8-tp4-switchless/upstream && tp4ctl down
#    confirm glm53_fp8_dflash_tp4 gone on all 4

# 2. Preflight — ALL PASS.  3. git pull on all 4 nodes (step 3 above).

# 4. Head node, NVFP4 upstream (worker-first 3,2,1,0 is upstream's job):
cd ~/code/spark-recipes/glm-v53-flash-nvfp4-tp4-switchless/upstream && ./start.sh serve
```

## Switch NVFP4 → DeepSeek-V4.1-Flash

```bash
# 1. Head node, NVFP4 upstream:
cd ~/code/spark-recipes/glm-v53-flash-nvfp4-tp4-switchless/upstream && ./start.sh stop
#    (stops + preserves containers; docker rm the stopped glm53-nvfp4-r* if you plan to
#     relaunch NVFP4 later — its preflight refuses existing names)

# 2. Preflight — ALL PASS.  3. git pull on all 4 nodes (step 3 above).

# 4. Proxy (head node): model-name-proxy/.env ->
#    BACKEND_MODEL=deepseek-v41-flash  +  add deepseek-v41-flash to SPOOF_FROM
cd ~/code/spark-recipes/model-name-proxy && docker compose --env-file .env up -d

# 5. Head node (workers 3,2,1 then 0 — upstream launcher ordering, site wrapper drives it):
cd ~/code/spark-recipes/deepseek-v41-flash-tp4-switchless/site && ./start-cluster.sh serve
#    (~8 min to /health from local NVMe; Image must be built + IMAGE_ID pinned first —
#     one-time steps in the recipe README)
```

## Switch DeepSeek-V4.1-Flash → NVFP4

```bash
# 1. Head node (rank 0 FIRST — a worker that starts while an old head still listens on
#    the rendezvous port joins the old head and hangs):
cd ~/code/spark-recipes/deepseek-v41-flash-tp4-switchless/site && ./start-cluster.sh stop
#    confirm deepseek-v41-flash-r* gone on all 4

# 2. Preflight — ALL PASS.  3. git pull on all 4 nodes (step 3 above).

# 4. Proxy: BACKEND_MODEL=glm-5.3-flash (deepseek-v41-flash may STAY in SPOOF_FROM);
#    docker compose --env-file .env up -d

# 5. Head node: cd glm-v53-flash-nvfp4-tp4-switchless/upstream && ./start.sh serve
#    (worker-first 3,2,1,0 is upstream's job)
```

## Verify after any switch

1. `/health` 200 on `http://10.69.43.170:8000` (wireless mgmt `wlP9s9`; the retired
   10.69.42.x wired fabric is gone); `/v1/models` shows the active lane's
   name (`glm-5.3-flash` NVFP4/FP8; `deepseek-v41-flash` for the DeepSeek lane) and
   `max_model_len` (NVFP4/DeepSeek: 1,048,576; FP8: 262,144).
2. Boot log per rank: `Connected all rings` and RouteFinal lines showing **all four
   PFs** (`rocep1s0f0/f1` + `roceP2p1s0f0/f1`). Any NCCL WARN line = stop and capture
   `docker logs` BEFORE any `docker rm -f`.
3. Gates (all lanes): corruption probe (count U+FFFD — must be 0, emoji + CJK intact)
   and a forced tool call. NVFP4 additionally: `bench/prefix_scan.py`
   (`--base` WITHOUT `/v1`, `--model glm-5.3-flash`; run 1 may FAIL near boot — a
   fresh-boot warm-phase transient; the settled warmed re-run is the verdict).
   DeepSeek additionally: `Engram DISK mode` + `BALANCED column assignment` +
   `PACKED single-read shard` boot lines (recipe README gates) and
   `SpecDecoding metrics` mean acceptance length > 1.
   No performance benches (`accept_probe`/`conc_bench` or any throughput/latency run)
   in a switch — benches belong to a config/weight-change session
   (`bench-recipe-update`), never to the switch itself.


## Known gotchas (each one cost a boot)

- **Directory rename (2026-10-10):** the recipes were renamed
  (`glm-v53-flash-4x-noswitch` → `glm-v53-flash-fp8-tp4-switchless`,
  `glm-v53-flash-nvfp4-4x-noswitch` → `glm-v53-flash-nvfp4-tp4-switchless`). The first
  `git pull` on the nodes AFTER that commit lands performs the rename on disk — do it
  with the target lane DOWN (running containers bind-mount paths under the recipe
  dirs; a live rename pulls the mount out from under the engine). Practical order:
  tear down → pull (dirs rename) → preflight → bring up via the NEW paths.

- **Stale mirrors on the nodes**: the FP8 recipe's `upstream/cluster.env` AND
  `upstream/scripts/node/etc/<node>/*.yaml` are gitignored derived files — `git pull`
  does NOT refresh them. After any repo config change: `cp
  glm-v53-flash-fp8-tp4-switchless/noswitch-prep/cluster.env
  glm-v53-flash-fp8-tp4-switchless/upstream/cluster.env` + `render-netplan.sh --write &&
  --check` on the head, then `deploy.sh` (tp4ctl reads the DEPLOYED
  `~/.local/tp4/cluster.env`).
- **Never re-deploy an old 2-address netplan** — it strips the P2 addresses and breaks
  the other lane (the breaking lane still boots; the breakage surfaces at the next
  switch). `FABRIC_P2_VIEWS=1` in `cluster.env` keeps the generator correct.
- **Never swap/merge the NCCL dirs**; the four-PF lib hard-rejects the legacy merged-
  vdev config and vice versa.
- **Teardown always cycles ALL ranks together** — never one rank.
- **DeepSeek stop order: rank 0 FIRST** (`./start-cluster.sh stop` does this) — a
  worker that starts while an old head still listens on the rendezvous port joins the
  old head and hangs (upstream cycle-lane gotcha). Its relaunch needs NO docker rm
  (the site wrapper removes stopped containers) but DOES re-run the upstream launcher
  preflight (MemAvailable ≥ 100 GiB — reboot the rank first if memory is fragmented).
- **DeepSeek Engram packed shards are checkpoint-bound**: after any CHECKPOINT change
  (not config changes), re-run `site/build-engram-packed.sh` per rank — the loader
  verifies geometry, not checkpoint identity.
- **Never bench right after boot** (~30 min warm-up; cold JIT skews ~30%).

## Pointers for depth

- FP8 lane ops/invariants: `glm-v53-flash-fp8-tp4-switchless/README.md`; history + T1 record:
  its `research.md` §5.
- NVFP4 lane ops/invariants: `glm-v53-flash-nvfp4-tp4-switchless/README.md`; history +
  Phase E/T2/T3 records: its `research.md`.
- NCCL four-PF build reproducibility: `glm-v53-flash-nvfp4-tp4-switchless/site/nccl-4pf/`.
- DeepSeek-V4.1-Flash lane ops/invariants: `deepseek-v41-flash-tp4-switchless/README.md`;
  provenance + transferability receipts + watchlist: its `research.md`; upstream lane
  quickstart: its `upstream/profile-deepseek-v41-flash-cycle/README.md`.
- Cluster topology (nodes, ring, addressing): repo `AGENTS.md`.
