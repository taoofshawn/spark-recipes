---
name: bring-up-spark-recipe
description: >-
  Use when bringing a recipe in the spark-recipes repo up on the 4-node DGX Spark
  switchless cluster — tearing down the currently-serving lane, running the fabric
  preflight, launching the FP8 (tp4ctl) or NVFP4 (start.sh) lane worker-first, and
  confirming health. Triggers: "bring up <recipe>", "start <recipe> on the sparks",
  "deploy the glm fp8/nvfp4 recipe", "latest recipe not running", "switch to the
  other lane", "fresh boot after a change".
---

# Bring Up a Spark Recipe (4-node switchless ring)

Deploy one of the two active GLM-5.3-Flash lanes onto the 4-node switchless DGX Spark
(GB10) cluster and leave it serving healthily on port 8000. All 2-node recipes are
archived (`.archived/`) — do not use their flows.

**The standing runbook is [`LANE-SWITCHING.md`](../../../LANE-SWITCHING.md) at the repo
root** — the two-lane switch procedure, preflight, verification, and gotchas. This skill
frames it; that file executes it. Read it first, every time.

## The cluster (fixed, do not re-derive)

Four nodes, switchless QSFP56 ring (rank0→1→2→3→0), SSH via DNS hostnames from the
workstation, never fabric IPs:

| rank | host | mgmt IP (`enP7s7`) | role |
|---|---|---|---|
| 0 | `spark-0f0b.shawndo.intra` | 10.69.42.170 | head — API server :8000 |
| 1 | `spark-6d14.shawndo.intra` | 10.69.42.171 | worker |
| 2 | `spark-6d90.shawndo.intra` | 10.69.42.172 | worker |
| 3 | `spark-6d24.shawndo.intra` | 10.69.42.173 | worker |

Transport: shared four-PF NCCL (`~/.local/lib/nccl-patched-4pf`, SHA `8de68def…`) over
the shared 4-view netplan; legacy 2-PF lib (`nccl-patched`, `afe5f486…`) is the FP8
rollback artifact. Both lanes live on ONE branch (`glm-v53-flash-nvfp4-4x-noswitch`) —
no branch checkout at switch time, `git pull` on the nodes is enough.

## Scope boundary

This skill brings a lane up and keeps it serving. It does NOT research/update recipe
sources (`update-recipe`), benchmark an update (`bench-recipe-update`), or create new
recipes (`adopt-recipe`). It never merges PRs.

## The loop

1. **Assess state** (all 4 nodes): `docker ps` for the other lane's containers
   (`glm53_fp8_dflash_tp4` vs `glm53-nvfp4-r*`), node repo branch/log/status. One lane
   at a time — tear down the other lane first.
2. **Preflight (non-negotiable):** `bash
   glm-v53-flash-fp8-tp4-switchless/noswitch-prep/scripts/verify-fabric-p2.sh` — must print
   `ALL CHECKS PASS` (netplan vs generated, 16 interface addresses, two-view ping mesh,
   both NCCL SHAs). Failure text names the fix; resolve before booting anything.
3. **Stale-mirror check** (the gotcha that costs boots): the FP8 recipe's
   `upstream/cluster.env` and `upstream/scripts/node/etc/<node>/*.yaml` are gitignored
   derived files — `git pull` does NOT refresh them. After any repo config change:
   copy `noswitch-prep/cluster.env` → `upstream/cluster.env` on all nodes, re-run
   `render-netplan.sh --write && --check` on the head, then `scripts/deploy.sh` FROM
   THE HEAD NODE (tp4ctl reads the DEPLOYED `~/.local/tp4/cluster.env`; the
   workstation run hits the Windows python3 stub).
4. **Tear down + bring up** per `LANE-SWITCHING.md` for the target direction:
   - FP8: `tp4ctl up` from `glm-v53-flash-fp8-tp4-switchless/upstream` on the head.
   - NVFP4: `./start.sh serve` from `glm-v53-flash-nvfp4-tp4-switchless/upstream`
     (worker-first 3,2,1,0 is upstream's job). A relaunch after `stop` needs
     `docker rm` of the stopped `glm53-nvfp4-r*` first — preflight refuses existing
     names.
   - Teardown always cycles ALL ranks together — never one rank.
5. **Verify health**: `/health` 200 (not `/v1/models` — it returns 200 with a dead
   engine), `/v1/models` shows `glm-5.3-flash` (NVFP4 max_model_len 1,048,576; FP8
   262,144), boot log per rank shows `Connected all rings` + RouteFinal with ALL FOUR
   PFs (`rocep1s0f0/f1` + `roceP2p1s0f0/f1`). Any NCCL WARN = stop, capture
   `docker logs` BEFORE any `docker rm -f`.
6. **Gates** (per lane, see the recipe README): corruption probe (0 U+FFFD, emoji +
   CJK intact), forced tool call; NVFP4 additionally `bench/prefix_scan.py`
   (`--base` WITHOUT `/v1`, `--model glm-5.3-flash`), `accept_probe`, `conc_bench`.
7. **Record** the bring-up in the lane's `research.md` changelog (dated entry: boot
   markers, gate results, gotchas hit).

## Hard rules

- **PR convention:** recipe changes flow topic-branch → PR → main. Open the PR but do
  NOT merge it unless the user explicitly asked in that session.
- **Never edit files on the nodes over SSH** (no sed/scp/heredocs on repo files). The
  sanctioned node-side steps are exactly the derived-file mirrors named above.
- **Never bench right after boot** — ~30 min warm-up first (cold JIT ~30% slow).
- Invariants are per-image and NOT transferable between lanes: KV pins, max_model_len,
  max_num_seqs, KV dtype, backend flags, scheduler classes. Don't "tidy" env vars.
- Warm-boot tip: relaunching against the same overlay root keeps JIT/TRITON caches
  (~3–4 min to /health vs ~7 cold). A fresh overlay should copy the old overlay's
  `cache/` first.

## Failure modes (4x-specific)

| symptom | cause / fix |
|---|---|
| NCCL init hangs / QP RTR timeout | wrong transport wiring — run the preflight; check the boot env shows `MERGE_NICS=0`, `SWITCHLESS_RING_ONLY=1`, `EXTENDED_IPV4_GIDS=1`, 4-PF HCA list |
| `wrong listener count` at ncclCommInitRank | four-PF lib without `NCCL_IB_EXTENDED_IPV4_GIDS=1` |
| FP8 launcher: "no usable automatic IPv4-mapped RoCE v2 fabric GID" on a P2 HCA | stale deployed cluster.env/gid-checker — see step 3; the allowlist must be `len(hcas)` entries |
| FP8 boots old config after a repo change | tp4ctl reads the DEPLOYED `~/.local/tp4/cluster.env` — run `deploy.sh` from the head |
| container name/overlay-path refused at launch | stopped containers from a previous `stop` not removed |
| `DEVSELECT_BUILD_FAILED` ×1–2 at NVFP4 boot (rank 0) | known watch item — overlay catches it, decode continues on fallback; record it, don't block the bring-up |
| ring decode far below the switched fleet | expected on this ring; c1 is latency-bound — compare against the lane's own bench history, not other operators' numbers |
