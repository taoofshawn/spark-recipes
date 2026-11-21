# AGENTS.md

Operational guide to this repository for agentic/automated contributors. **Scope: three
self-contained 4-node-switchless serving recipes are active.** All 2-node recipes and their
skills/docs knowledge are archived under `.archived/`. Do not resurrect archived flows; this
file describes the live cluster only.

## What this repository is

Three **discrete, independently operable** serving recipes on the **4-node DGX Spark (GB10)
switchless ring**, each TP4 across all four nodes. Each recipe carries its own fabric facts,
netplan generator + committed netplan, host bootstrap, `/etc` set, kernel/driver pins,
four-PF NCCL build tree, preflight, from-zero runbook and `bin/{start,stop}.sh`.

| recipe | weights | context / KV | control | upstream |
|---|---|---|---|---|
| `glm-v53-flash-nvfp4-tp4-switchless` | stock `nvidia/GLM-5.3-Flash-NVFP4` @ `da920bb` (zero weight mod — the lane's premise) | 1M / 20 GiB/rank pin (~3.25M tokens) | `./bin/start.sh` → `upstream/start.sh serve` | vendored knapcio `770d115` |
| `glm-v53-flash-fp8-tp4-switchless` | zai FP8 + the E36 conversion stack | 262K / 14 GiB pool (1,194,033 tokens) | `./bin/start.sh` → `upstream/scripts/tp4ctl up` | pinned jnardiello `ed365a6` (`080fe09` in the README; reconcile) |
| `deepseek-v41-flash-tp4-switchless` | stock `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e10` (zero weight mod, MIT) | 1M / free-memory-sized pool | `./bin/start.sh` → `site/start-cluster.sh serve` | vendored knapcio `540bf07` (v2.6.4) |

The two GLM recipes serve `glm-5.3-flash`; DeepSeek serves `deepseek-v41-flash`. All on
`:8000` behind the model-name proxy, so clients keep one name across recipes.

**There is no lane-switching document, and none is needed.** The one precondition that cannot
be derived — "only one lane at a time" — is executable: each recipe's `bin/start.sh` refuses
while anything else serves (`bin/guard.sh`, fail-closed) and `bin/stop.sh` captures logs then
removes that recipe's own containers. Each recipe's README carries its start/stop/gates.

## Deletion rule (the point of the layout)

**Any recipe directory may be deleted at any time; the others must keep working.** Nothing
under a recipe reads a sibling recipe's files. If you delete one, remove its rows from this
file, `README.md`, the skills, and the proxy's `SPOOF_FROM` in the same commit. The
`model-name-proxy/` is the only deliberate repo-root exception — it exists precisely because
there is more than one recipe.

## Cluster topology (four fixed nodes, switchless ring)

| rank | host | mgmt IP | notes |
|---|---|---|---|
| 0 (head, API server) | `spark-0f0b.shawndo.intra` | 10.69.43.170 | hosts the API process; `tp4ctl`/`start.sh` run here |
| 1 | `spark-6d14.shawndo.intra` | 10.69.43.171 | worker |
| 2 | `spark-6d90.shawndo.intra` | 10.69.43.172 | worker |
| 3 | `spark-6d24.shawndo.intra` | 10.69.43.173 | worker |

**2026-10-19 move:** the cluster is now on the wireless mgmt NIC `wlP9s9`
(10.69.43.0/16) — the wired `enP7s7` is DOWN at the new location. The QSFP56 fabric
(10.10.x) is unaffected. Every recipe pins `MGMT_IPS=10.69.43.x` / `MGMT_IF=wlP9s9` in its
own facts file (`<recipe>/site/fabric/cluster.env` or `noswitch-prep/cluster.env`).

- **Fabric:** 4 QSFP56 cables in a switchless ring rank0→1→2→3→0; each cable presents
  **2 PFs per PCIe root** (`enp1s0f0np0`/`enp1s0f1np1` = the f ports,
  `enP2p1s0f0np0`/`enP2p1s0f1np1` = the second PCIe views). Since the NVFP4 lane's
  Phase E, ALL FOUR views are addressed (netplan `40-cx7.yaml`: P2 octet = f octet +
  2, additive over the old 2-PF plan; `FABRIC_P2_VIEWS=1` in the FP8 recipe's
  `cluster.env` keeps the generator emitting it). MTU 9000. The old
  `192.168.0.x` crossover IPs no longer exist.
- **NCCL (shared):** four-PF switchless build `~/.local/lib/nccl-patched-4pf`
  (SHA `8de68def…`, othexmr stack @ `94c2669` + one-line site GID patch — fully
  reproducible via the NVFP4 recipe's `site/nccl-4pf/build-4pf.sh`). Contract env:
  `NCCL_SWITCHLESS_RING_ONLY=1`, `NCCL_IB_MERGE_NICS=0`,
  `NCCL_IB_EXTENDED_IPV4_GIDS=1`, `NCCL_IB_PRESERVE_PCI_DOMAIN=1`, 4-PF
  `NCCL_IB_HCA`. The legacy 2-PF lib (`~/.local/lib/nccl-patched`, `afe5f486…`) is
  the FP8 lane's rollback artifact — keep both dirs, never swap in place (the four-PF
  lib hard-rejects merged vdevs and vice versa).
- **SSH** to nodes via DNS hostnames from the workstation (`eve`); never fabric IPs.
- **GPU contention rule:** every recipe uses all reserved GPUs and binds `:8000`. One lane at
  a time — enforced by each recipe's `bin/start.sh` (see the deletion rule above); do not
  start a second lane by hand.

## Where the knowledge lives

| doc | contents |
|---|---|
| `<recipe>/README.md` | that recipe's ops doc: start/stop via `bin/`, invariants, gates |
| `<recipe>/research.md` | audit trail: provenance, dated changelog, watchlist, receipts |
| `<recipe>/site/fabric/` (FP8: `noswitch-prep/`) | fabric facts, netplan + committed netplan, vendored host bootstrap, preflight, `REUSE.md` |
| `<recipe>/site/FROM-ZERO.md` (or upstream `docs/install-from-zero.md`) | rebuild that recipe from wiped nodes |
| `<recipe>/upstream/VENDORED-AT.md` | upstream pin + every site modification and its re-apply recipe |
| `<recipe>/site/fabric/check_switchless_nccl.py` (FP8: `noswitch-prep/`) | NCCL artifact identity check |

## Upstream sources (for update passes)

- **NVIDIA forum** (Discourse JSON API, no auth, browser-ish User-Agent): tracked
  threads `381350` (GLM-5.3-Flash main), `382459` (jnardiello switchless), `382939`
  (nvidia-NVFP4 optimization), `385075` (kindling 3x). General sweep every run:
  newest topics in categories `721`/`723` since the last review date + varied searches
  (`glm 5.3 flash`, `dflash`, `nvfp4`, `switchless`). API patterns:
  `/c/.../721.json?order=activity`, `/search.json?q=...`, `/t/<id>.json?page=N`.
- **GitHub:** `knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4` (NVFP4 lane upstream),
  `jnardiello/GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless` (FP8 lane upstream),
  `tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark` + his 2x sibling (serving
  family; his current default is knapcio's stack), kindlingai / eugr / technigma
  stacks for cross-stack ideas.
- **HF checkpoints:** `nvidia/GLM-5.3-Flash-NVFP4` (pin `da920bb` — any update must re-verify
  the zero-mod premise), drafter `incoai/GLM-5.3-Flash-DFlash2` (pin `bf582e4e`,
  **CC BY-NC-ND 4.0** — no derivative re-encodes), `deepseek-ai/DeepSeek-V4.1-Flash-1017`
  (pin `1bae0e10`). Compare `lastModified`/`siblings` at review time.
- **Images:** every lane builds locally (`glm53-roce:v11-b58f34ea` from knapcio's
  `Dockerfile.roce`; the FP8 digest-pinned image per its `noswitch-prep/` records;
  DeepSeek's `dsv41-4x-spark:canary-roce-miaai` from a digest-pinned base) —
  no registry tracking. An image bump cascades into pins, mount paths, and gates.
- **Adoption gate:** every candidate must (a) apply to the pinned image/revision,
  (b) not contradict the lane's validated invariants, (c) carry a measured claim.
  Rejections are recorded with reasons in research.md — they are knowledge too.

## Git / workflow conventions

- The working branch `glm-v53-flash-nvfp4-4x-noswitch` carries BOTH recipes (since the
  2026-10-10 reorg). `main` = stable only; direct pushes to `main` are blocked — all
  changes land via topic branch pushed to origin (no PR opened by agents).
- **Do NOT open PRs.** Push the commits to the topic branch and report the branch name +
  commit range; that is the stopping point. The user opens/merges PRs themselves once the
  remaining open issues are complete (previously PR #82 was opened by an agent — stop doing
  that; existing PRs stay until the user merges).
- **Never edit files on the nodes over SSH** (`sed -i`, `scp`, heredocs). Recipe
  changes flow: repo → topic branch (pushed) → user opens PR → main (or the working
  branch) → `git pull` on the nodes. Node working trees stay pristine.
- **Derived-file mirrors are gitignored on the nodes** (FP8 `upstream/cluster.env`,
  `upstream/scripts/node/etc/<node>/*.yaml`): after a repo config change, copy the
  mirror + re-render on the head + `scripts/deploy.sh` FROM THE HEAD NODE (tp4ctl
  reads the DEPLOYED `~/.local/tp4/cluster.env`; the workstation run hits the Windows
  python3 stub).
- Commit messages: descriptive one-liners in repo history style. Notable tuning work
  gets a dated changelog block in the lane's `research.md` (what, measured
  before/after, gotchas). **README = active-running doc only** — history, TODO, and
  watch items belong in research.md.
- `.gitignore` covers `.research/` (local scratch), `__pycache__`, generated netplans.
  Handoff files live in `.research/` and stay untracked.

## Lane invariants (per-image — NOT transferable between lanes)

- NVFP4: KV_BYTES 20 GiB/rank (24→28 rungs are proven ceilings but MUST be gated per
  rung), MAX_MODEL_LEN 1M, the `stock-nvfp4.env` strip list (zero weight-mod premise —
  the lossless8 fallback is an explicit owner decision, not a drift), start order
  worker-first 3,2,1,0.
- FP8: `GPU_MEM_UTIL=0.85` (0.89+ = NV_ERR_NO_MEMORY on the head), 14 GiB KV pool,
  `MAX_NUM_SEQS=6` + bounded-admission middleware, E35/E36 scheduler stack, pinned
  kernel `6.17.0-1032-nvidia` + driver `580.173.02` (the 7.x kernel breaks ring NCCL —
  no `apt upgrade` on the nodes).
- Shared: port 8000, `glm-5.3-flash`, offline serving, four-PF NCCL contract,
  `FABRIC_P2_VIEWS=1`, all-ranks teardown.

## Teardown rule (repo-wide — every recipe, including future ones)

**Stopping a lane removes its containers.** Every recipe's `bin/stop.sh` must:

1. **capture `docker logs` first** for every rank into a lane-owned file (the container is
   the only place NVFP4 keeps its log — removal destroys it, so this step is not optional);
2. delegate to the recipe's own stop/teardown;
3. **remove** the recipe's containers (`docker rm -f` its own names only) and verify absence;
4. `--keep-containers` is the explicit, single-inspection opt-out.

Never leave stopped containers behind: a preserved container blocks the next `serve` on lanes
whose preflight refuses existing names (NVFP4), and it makes "delete any recipe dir at any time"
false in practice. Warm JIT caches are unaffected — they live in host bind-mounts
(`~/.local/nvfp4-tp4/cache`, `~/.cache/tp4-vllm-cache`), not the container layer.

New recipes inherit this rule; do not copy the upstream "stop but preserve" behaviour.

## Failure modes (4x; each cost a boot)

| symptom | cause / fix |
|---|---|
| NCCL init hangs / QP RTR timeout | transport wiring — run the target recipe's `preflight.sh`; boot env must show `MERGE_NICS=0` + `SWITCHLESS_RING_ONLY=1` + `EXTENDED_IPV4_GIDS=1` + 4-PF HCA |
| `wrong listener count` at ncclCommInitRank | four-PF lib without `NCCL_IB_EXTENDED_IPV4_GIDS=1` |
| FP8 launcher GID preflight rejects a P2 HCA | stale deployed config/gid-checker allowlist (site fix #7) — redeploy from the head |
| FP8 boots old config after a repo change | tp4ctl reads the DEPLOYED `~/.local/tp4/cluster.env` — `deploy.sh` from the head |
| container name/overlay path refused | a container from a previous run still exists — `bin/stop.sh` captures logs and **removes** this recipe's containers, so re-run it, then start |
| two lanes "running" at once | impossible by construction — each recipe's `bin/start.sh` refuses while anything else serves (`bin/guard.sh`, fail-closed). If you see it, someone bypassed the guard; stop both and use `bin/start.sh` |
| DeepSeek stop leaves a hung cluster | stop order matters: **rank 0 first** (`start-cluster.sh stop` does this). A worker that starts while an old head still listens on the rendezvous port joins the old head and hangs |
| DeepSeek refuses to start; `doctor` reports `MemAvailable` low | memory fragmentation on that rank after long uptime — **reboot that rank**, not the lane |
| DeepSeek loses Engram shards after a checkpoint change | packed shards are checkpoint-bound — re-pack per rank (`start-cluster.sh pack`); the loader verifies geometry, not identity |
| a live recipe directory rename pulls mounts out from under a running engine | do renames with the lane **DOWN**: tear down → pull → preflight → bring up |
| re-deploying an old 2-address netplan | strips the P2 addresses and breaks the four-PF listener contract; your recipe's committed netplan always has both views |
| `DEVSELECT_BUILD_FAILED` ×1 at NVFP4 boot (EVERY rank, ~20:22:30 boot-moment) | RESOLVED 2026-10-11: upstream platform limitation — stream-captured conditional-node CUDA-graph parents reject NCCL-collective child graphs at `capture_end` (`cudaErrorInvalidValue`; reproducible with stock NCCL, mode/registration-independent). Device-side select NEVER arms on this driver stack; host fallback is the validated baseline. Zero action — expected boot line. Re-verify only if the kernel/driver pin changes (nvfp4 research.md 2026-10-11 entry). |
| `tool_choice:"required"` + thinking-on killed the engine (grammar-bitmask assert) | FIXED 2026-10-10: the E35 runner-k worker-side draft trim desynced the grammar bitmask (8 rows) from the trimmed logit mapping (4) — grammar steps no longer trim + the worker re-slices the bitmask; thinking-on + required is now a supported combination (fp8 research.md 2026-10-10 entry) |
| ring decode far below switched fleets | expected — c1 is latency-bound; compare against the lane's own bench history only |
| bench right after boot reads ~30% slow | JIT/TRITON warm-up — wait ~30 min, never bench cold |

## Agents: do / don't

- **Do** read the target recipe's README (start/stop, gates) + its latest research.md
  entries before touching it. There is no switch runbook — `bin/start.sh` enforces the
  one-lane-at-a-time precondition.
- **Do** run the target recipe's `preflight.sh` before every switch; record anomalies (failed/retried
  boots, NCCL WARNs, gate failures, config changes, doc fixes) in research.md — a clean
  switch with no surprises updates nothing.
- **Do** keep site mods documented in VENDORED-AT.md with re-apply recipes; keep
  generated files generated (never hand-edit).
- **Don't** resurrect archived 2-node flows (compose pairs, `.env.nodeN`, crossover
  IPs, sparkrun) — they live in `.archived/` for reference only.
- **Don't** tune shared knobs across lanes as if they were common (KV pins, mem util,
  seq caps, backends are per-image).
- **Don't** restart a lane to "fix" slowness without captured evidence; diagnosis
  first (see the troubleshoot-slowness skill).
