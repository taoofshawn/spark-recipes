# Lane switching runbook — GLM-5.3-Flash FP8 ↔ NVFP4 on the 4-node switchless ring

The one doc a fresh agent needs to switch between the two serving lanes without reading
both recipes end-to-end. Deep context lives where noted; this file is the procedure.

**State this was verified against:** 2026-10-10, both lanes boot and switch cleanly on the
four-PF NCCL contract (T1/T2 records: `glm-v53-flash-4x-noswitch/research.md` §5 and
`glm-v53-flash-nvfp4-4x-noswitch/research.md` Changelog).

## The two lanes

| | FP8 lane | NVFP4 lane |
|---|---|---|
| recipe dir | `glm-v53-flash-4x-noswitch/` | `glm-v53-flash-nvfp4-4x-noswitch/` |
| weights | zai FP8 checkpoint (lossless) + lossless8-style conversion stack (E36) | stock `nvidia/GLM-5.3-Flash-NVFP4` @ `09b04e5e`, zero weight mod |
| context / pool | 262K / ~1.19M-token KV | 1M / ~3.25M-token KV (20 GiB/rank pin) |
| control | `tp4ctl` (`up`/`down`/`restart`/`health`) on the head | `./start.sh` (`serve`/`stop`/`status`/`logs`) on the head |
| containers | `glm53_fp8_dflash_tp4` | `glm53-nvfp4-r0..r3` |
| NCCL | **shared**: `~/.local/lib/nccl-patched-4pf` (`8de68def…`, four-PF); legacy 2-PF lib `~/.local/lib/nccl-patched` (`afe5f486…`) is the FP8 rollback artifact — keep both dirs |

Shared facts that make switching simple:

- **One branch serves both lanes** (`glm-v53-flash-nvfp4-4x-noswitch`). No branch
  checkout at switch time — `git pull` on the nodes is enough.
- **One fabric netplan** (all four PCIe views addressed) and **one NCCL lib** — nothing
  network-level to decide at switch time.
- **No proxy change either way**: both serve `glm-5.3-flash` on `:8000` behind the
  model-name-proxy; clients never change.
- **One lane at a time** (GPU contention): every recipe uses all reserved GPUs.

## Preflight — run before EVERY switch (cheap, read-only, safe while serving)

```bash
bash glm-v53-flash-4x-noswitch/noswitch-prep/scripts/verify-fabric-p2.sh
```

Must print `ALL CHECKS PASS`. It gates on the netplan generator, compares live
`/etc/netplan/40-cx7.yaml` vs generated on all 4 nodes, checks all 16 interface
addresses, runs the 16-ping two-view mesh, and SHA-verifies both staged NCCL libs.
Failures: resolve before booting anything (the failure text names the fix).

## Switch NVFP4 → FP8

```bash
# 1. Maintenance window. On the HEAD node, NVFP4 upstream:
cd ~/code/spark-recipes/glm-v53-flash-nvfp4-4x-noswitch/upstream && ./start.sh stop
#    (stops + PRESERVES containers; confirm glm53-nvfp4-r* gone/stopped on all 4 —
#     `docker rm` the stopped ones if you plan to relaunch NVFP4 later, its preflight
#     refuses existing names)

# 2. Preflight (above) — ALL PASS.

# 3. On ALL FOUR nodes:
cd ~/code/spark-recipes && git pull

# 4. ONLY if a repo config change is pending (cluster.env etc.): from the HEAD node
#    (NOT the workstation — its Windows python3 stub breaks deploy):
cd ~/code/spark-recipes/glm-v53-flash-4x-noswitch/upstream && bash scripts/deploy.sh

# 5. Head node:
cd ~/code/spark-recipes/glm-v53-flash-4x-noswitch/upstream && tp4ctl up
#    (upstream handles worker-first ordering)
```

## Switch FP8 → NVFP4

```bash
# 1. Head node, FP8 upstream:
cd ~/code/spark-recipes/glm-v53-flash-4x-noswitch/upstream && tp4ctl down
#    confirm glm53_fp8_dflash_tp4 gone on all 4

# 2. Preflight — ALL PASS.  3. git pull on all 4 nodes (step 3 above).

# 4. Head node, NVFP4 upstream (worker-first 3,2,1,0 is upstream's job):
cd ~/code/spark-recipes/glm-v53-flash-nvfp4-4x-noswitch/upstream && ./start.sh serve
```

## Verify after either switch

1. `/health` 200 on `http://10.69.42.170:8000`; `/v1/models` shows
   `glm-5.3-flash` (NVFP4: `max_model_len` 1,048,576; FP8: 262,144).
2. Boot log per rank: `Connected all rings` and RouteFinal lines showing **all four
   PFs** (`rocep1s0f0/f1` + `roceP2p1s0f0/f1`). Any NCCL WARN line = stop and capture
   `docker logs` BEFORE any `docker rm -f`.
3. Gates (both lanes): corruption probe (count U+FFFD — must be 0, emoji + CJK intact)
   and a forced tool call. NVFP4 additionally: `bench/prefix_scan.py`
   (`--base` WITHOUT `/v1`, `--model glm-5.3-flash`).

## Known gotchas (each one cost a boot)

- **Stale mirrors on the nodes**: the FP8 recipe's `upstream/cluster.env` AND
  `upstream/scripts/node/etc/<node>/*.yaml` are gitignored derived files — `git pull`
  does NOT refresh them. After any repo config change: `cp
  glm-v53-flash-4x-noswitch/noswitch-prep/cluster.env
  glm-v53-flash-4x-noswitch/upstream/cluster.env` + `render-netplan.sh --write &&
  --check` on the head, then `deploy.sh` (tp4ctl reads the DEPLOYED
  `~/.local/tp4/cluster.env`).
- **Never re-deploy an old 2-address netplan** — it strips the P2 addresses and breaks
  the other lane (the breaking lane still boots; the breakage surfaces at the next
  switch). `FABRIC_P2_VIEWS=1` in `cluster.env` keeps the generator correct.
- **Never swap/merge the NCCL dirs**; the four-PF lib hard-rejects the legacy merged-
  vdev config and vice versa.
- **Teardown always cycles ALL ranks together** — never one rank.
- **Never bench right after boot** (~30 min warm-up; cold JIT skews ~30%).

## Pointers for depth

- FP8 lane ops/invariants: `glm-v53-flash-4x-noswitch/README.md`; history + T1 record:
  its `research.md` §5.
- NVFP4 lane ops/invariants: `glm-v53-flash-nvfp4-4x-noswitch/README.md`; history +
  Phase E/T2/T3 records: its `research.md`.
- NCCL four-PF build reproducibility: `glm-v53-flash-nvfp4-4x-noswitch/site/nccl-4pf/`.
- Cluster topology (nodes, ring, addressing): repo `AGENTS.md`.
