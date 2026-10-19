# research.md — glm-v53-flash-nvfp4-tp4-switchless

Audit trail, provenance, and the decision record for the zero-weight-modification
NVFP4 lane on the 4-node switchless ring. `README.md` stays the running ops doc only.

## 1. Provenance

| what | value |
|---|---|
| upstream source | `github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4` |
| vendored pin | `770d115` (2026-09-29 tip; `3edfbc9..770d115` diffed pre-vendor: docs + the switchless transport only — no serving-kernel changes) |
| vendored on | 2026-09-29, `.git` stripped, detached |
| checkpoint | [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) @ `da920bb0b9f4a06727223a349e55468e38352348` — **stock, zero modification** (re-pinned from `09b04e5e` 2026-10-12, owner-approved; layer 45 leaves the NVFP4 ignore list → served BF16; all 33 weight shards byte-identical to `09b04e5e`. See the 2026-10-12 re-pin changelog entry) |
| drafter | [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) @ `bf582e4e` — stock bf16 (no `drafter_fp8.py` re-encode; CC BY-NC-ND 4.0) |
| image | `glm53-roce:v11-b58f34ea`, built per-node from upstream `Dockerfile.roce` (RoCEnante present but inert under `TRANSPORT=switchless`) |
| site files | `site/env.switchless` (the `.env`), `site/materialize-weights.sh`, `upstream/profiles/stock-nvfp4.env`, `upstream/VENDORED-AT.md` |

Why knapcio (not alexellis / jnardiello / tonyd2wild): only stack that (a) anchors on the
exact nvidia checkpoint with its NVFP4 experts untouched, (b) ships a switchless-ring
transport mode, (c) is the measured fastest (switched fleet, one boot, 2026-09-29: c1
decode prose 90.3 / code 124.0 / structured 168.8 tok/s; c16 aggregate 331–879; cold
prefill 2.8–3.5K tok/s to 128K; 3.6-min boot; GSM8K 246/250, HumanEval 95.7%).

## 2. Decision record (2026-09-29, owner-confirmed)

1. **Name** `glm-v53-flash-nvfp4-4x-noswitch` (parallel to the FP8 recipe).
2. **NCCL: inspect-first.** Hardware facts (owner, 2026-09-29): each inter-node link is a
   200G QSFP56 cable presenting **2×100G virtual PFs** (one per PCIe root — the 2-node
   recipes' `IB_PORTS=rocep1s0f0,roceP2p1s0f0` pair is one cable's two functions; the
   `.f1` names are the right-hand port). The four-PF contract therefore asks for devices
   that demonstrably exist; the gap is **configuration, not hardware**: the FP8 ring
   addresses only one PF per port (2 of 4 per node — its "two addressed fabric interfaces
   per node" invariant). Bring-up step 1: (a) netplan-address the second PF of each port
   on all four nodes (fabric plan extends 4 → 8 logical links; site range widened to
   `10.10.0.0/21` in site/env.switchless), (b) run `scripts/check_switchless_nccl.py` on
   the existing site rebuild (`2.30.7-1`, SHA `afe5f486…`) — configuring PFs alone is not
   sufficient: "a two-device-only release is not sufficient merely because its filename is
   the same" (upstream docs/switchless.md), so the build must also support four devices.
   If it fails, build othexmr's four-PF NCCL (`feat/four-pf-source-integration` @
   `94c2669`) against the site addressing. **Known risk:** upstream's four-PF build was
   validated on `10.100.224.0/22`; our ring is `10.10.0.0/21` — a site rebuild+validation
   is likely. Fallback lane: the FP8 recipe's proven fabric/NCCL wiring against this image
   (jnardiello's fidelity arm booted the tonyd2wild image family TP4 on exactly this
   fabric; do NOT set `NCCL_SWITCHLESS_RING_ONLY`/`NCCL_IB_MERGE_NICS=0` — failed RDMA QP
   connect there).
3. **Weights: already on all four nodes** (owner, default HF cache). No fan-out needed;
   materialization (`site/materialize-weights.sh`) hardlinks the snapshots into
   real-file dirs because `start.sh` bind-mounts `MODEL_DIR` directly (`/model`) and HF
   snapshots use relative symlinks into `blobs/` that break inside a container mount.
   Hardlinks = zero copy, zero disk, provably unmodified bytes.
4. **v1 profile: 1M ctx on a conservative pool.** `MAX_MODEL_LEN=1048576`,
   `KV_BYTES=21474836480` (20 GiB/rank → ~3.25M fp8-KV tokens at the measured ~40.6K
   tokens/GiB → 3.1× full-1M concurrency). Owner explicitly chose the 3.1× conservative
   ratio; 24→28 GiB/rank rungs are proven ceilings on identical hardware (tonyliu312
   4,545,221 tokens @ 28 GiB/rank, head rank binding) but gate each rung.
5. **Zero weight modification is the product, not an accident** — if the lane is too
   slow/unpleasant the fallback is knapcio's lossless8 target (`scripts/build_lossless8.sh`
   + `drafter_fp8.py`, 0.25% output error, ~10–15% decode gain + ~9 GiB KV) — revisit
   decision explicitly, don't drift into it.

## 3. The stock-compat delta (what `profiles/stock-nvfp4.env` strips and why)

`current.env` @ `770d115` assumes the lossless8 target and the fp8blk drafter for:

| stripped switch | assumes |
|---|---|
| `QMIX_FP8_BLOCK=1`, `VLLM_TEST_FORCE_FP8_MARLIN=1`, `QMIX_DEBUG_LAYERS=1` | lossless8 overlay loader filter + sentinel (no overlay exists on the stock pack) |
| `GLM_DENSE_FAST=1`, `GLM_DENSE_FAST_PREFILL=1` | dense layers on 8-bit grids (stock pack's dense layers are BF16) |
| `GLM_DS_DRAFT_HEAD_FP8=1`, `GLM_LV_DRAFT_FP8_KV=1` | the fp8blk-converted drafter (this lane serves stock bf16) |

Everything else is inherited live (source-order: `stock-nvfp4.env` → `current.env`), so
upstream kernel/scheduler wins keep arriving. **Watch item:** any upstream
`profiles/current.env` change needs a stock-compat review of new switches — a new
QMIX/DENSE/drafter-FP8 switch inherited silently is the failure mode. Expected cost of
the strip list: ~10–15% decode (−7.5 ms/step dense read) + ~1.1 ms drafter graph;
measured at bring-up, not assumed.

## 4. Receipts carried into this recipe

| claim | receipt |
|---|---|
| nvidia pack = corruption-free (0/0/0 U+FFFD; attention kept BF16, 132-entry ignore list) | tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark README, issue #23 |
| stock nvidia pack boots unmodified on the tonyd2wild image family (TP2) | same repo, DFlash2 launcher default since 2026-09-24 |
| stock nvidia pack TP4 = unmeasured publicly; the "does not come up" failure applies only to the built attn-quantized checkpoint | tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark README (`NVFP4_PATCH` section) |
| native MTP impossible on the nvidia pack (BF16 MTP head; no sm121 backend serves NVFP4 target + unquantized draft MoE) → DFlash2 is the only drafter | tonyd2wild issue #23 |
| ~40.6K fp8-KV tokens per GiB (block 2304, DFlash2 slot-share) | tonyliu312: 4,545,221 @ 28 GiB/rank; tonyd2wild 1M lane: 3,834,498 @ 24 GiB/rank |
| knapcio switched-fleet performance + quality gates | upstream README @ `770d115` (sparkDash 1.8.8, RigMark screen, qeval/GSM8K/HumanEval) |
| NVFP4-experts-vs-FP8-checkpoint on the same engine: code +40%, JSON +50% | upstream README (Marlin W4A16 vs official FP8 checkpoint, same quality gate) |
| precision ladder: W4A4 experts 16–21% MoE error (rejected); 8-bit dense 0.25%; NVFP4 experts = pack quant | upstream README rejected-experiments + weights docs |
| switchless transport = community-contributed, CPU-tested only upstream; ring decode penalty unquantified | upstream `docs/switchless.md` (othexmr PR #1) |
| mamba-align correctness fix must be active | upstream issue #2 + `GLM_MAMBA_ALIGN_FIX=1` (inherited) + `bench/prefix_scan.py` gate |

## 5. Watchlist / open items

Refreshed 2026-10-11 against the changelog (superseded rows closed with pointers; no
open item dropped). The 2026-10-11 T3a root-cause CLOSED the `DEVSELECT_BUILD_FAILED`
watch item raised in the Phase E entry (upstream platform limitation, every rank, never
armed on this driver stack — see that entry; re-verify only if the kernel/driver pin
changes). §5 never carried it separately, so no row here.

- **CLOSED 2026-10-10 — Second-PF netplan plan** — superseded by Phase E: the full
  four-PF netplan was rendered and deployed 4/4 (P2-view octet = f-port octet + 2,
  additive over the FP8 plan), 16/16 pings verified across both PCIe views per cable.
  Per-link map in the Phase E changelog entry; deployed mirror at
  `site/netplan/<node>/40-cx7.yaml`. (The original "8 new /24s" reading was corrected
  pre-bring-up: the contract is 2 cable subnets/node × 2 PCI roots, one address each.)
- **CLOSED 2026-10-10 — NCCL four-device verdict** — superseded by Phase E: the four-PF
  lib (othexmr stack @ `94c2669` + one-line site GID-range patch, SHA `8de68def…`) is
  built, identity-check PASS ×4, fully reproducible via `site/nccl-4pf/build-4pf.sh`;
  RouteFinal shows all four PFs carrying QPs on every rank, reproduced at T2 the same
  day. The `afe5f486…` 2-PF rebuild question is moot — that lib stays staged only as
  the FP8 lane's rollback artifact.
- **CLOSED 2026-10-12 — true 1M-token context gate + deep-context quality check
  (FIX-3)** — the ~1M request ran end-to-end TWICE (thinking OFF + ON, independent
  prefills — the 2nd missed cache, `prefix_cache_queries` delta 998,405 / hits 0) and
  the deep-context quality ladder PASSED at every depth in both modes. Receipts in the
  2026-10-12 FIX-3 changelog entry below.
- **OPEN — ~1M prefix-cache retention across concurrent sessions (DeepSeek issue #43
  failure shape REPRODUCED on this lane 2026-10-12).** Store at depth works (an
  immediate byte-identical repeat of a ~997K prompt hits 99.8–99.9%), but a SECOND
  ~1M session's prefill silently evicts the first session's resident prefix at ≤78%
  pool occupancy (headroom ≥22%) — the first session's next turn pays a full ~437 s
  re-prefill. The ~3.25M-token pool claim holds for CAPACITY, not for multi-session
  retention at ~1M depth (effective retained sessions ≈ 1). See the 2026-10-12-late
  changelog entry for the S1–S7 receipt. First suspects (mirroring the upstream
  report): a structural reservation in the hybrid-model KV/prefix-tail sizing rather
  than LRU-under-pressure — root cause NOT yet identified on this stack. Agent-facing
  mitigation until root-caused: deep (~1M) sessions should not assume cache survival
  across concurrent deep sessions; ~5–7 min cold re-prefill per revisit is the cost.
  UPSTREAM REPORT FILED 2026-10-13: knapcio issue #10 (S1–S7 methodology + receipts;
  asks for independent repro and whether it is expected vLLM block-reservation
  semantics or a vLLM-level bug).
- **CLOSED 2026-10-10 — Ring decode penalty** — measured and compared at first bring-up
  and Phase E: c1 is latency-bound at ~45–55% of knapcio's switched fleet (44.7/75.8/
  96.3 tok/s at Phase E); the four-PF transport pays off at concurrency (json c8 +26%,
  c16 +15%) and prefill TTFT (json c8/c16 ~6.5 s → 0.48/0.58 s); T2 reproduced within
  −2.3%…+3.8%. Lane stays zero-mod; lossless8 fallback NOT triggered — that remains an
  explicit owner decision per §2 decision 5, not a watch item. External corroboration
  (knapcio issue #9, 2026-10-12 pass): a third party hardware-qualified
  `TRANSPORT=switchless` on a physical 4-node ring with 4 RDMA functions/node all
  addressed — same shape as this ring.
- **OPEN — GLM_ROCE_ALLREDUCE / GATHER_ROUTE hardware qualification** — still never
  hardware-qualified (no changelog entry covers them). Under `TRANSPORT=switchless` the
  launcher forces `GLM_ROCE_ALLREDUCE=0` and `GATHER_ROUTE` is inert (upstream
  `docs/switchless.md`), so every collective runs on the patched NCCL ring. Stays on
  the list as the first suspects if the ring ever misbehaves.
- **CLOSED 2026-10-10 — glm47 `tool_choice:required` parity** — forced tool calls are
  clean at every boot on this lane (first bring-up, Phase E, T2: clean
  `get_weather({"city": "Tokyo"})`); NO structural-tag override (TC-45) is needed —
  the upstream glm47 parser handles it. Parity with the FP8 lane confirmed.
- **OPEN — Upstream churn** — knapcio's `main` moves fast. Latest full pass 2026-10-12
  (see the dated changelog entry): knapcio tip still `770d115`, nothing adoptable as
  config; nvidia checkpoint tip moved to `da920bb` (PARKED re-pin candidate, owner
  decision). Prior full pass 2026-10-10 (tonyd2wild review + forum sweep): knapcio tip
  still `770d115`, nothing adoptable landed after 2026-09-29. Next action: re-run the
  VENDORED-AT update procedure periodically and on any adopted change — especially any
  upstream `profiles/current.env` change, which needs the §3 stock-compat review.
- **PARKED (owner decisions, not adopted — 2026-10-12 pass):**
  - **~~Checkpoint re-pin candidate~~ ADOPTED 2026-10-12 (owner instruction; bench
    receipts in the changelog entry below) and KEEP finalized 2026-10-13:** the FIX-3
    quality ladder re-run on `da920bb` (the re-pin's quality motivation) shows NO
    quality regression — 16/16 probes PASS, identical to the `09b04e5e` baseline run
    (2026-10-13 changelog entry); owner decision: the lane stays on `da920bb`.
  - **KV prefix-cache store-vs-retain bench methodology** (knapcio issue #8, via
    MiaAI-Lab DeepSeek issue #43): ADOPTED 2026-10-12-late — run the same session; it
    reproduced the upstream retention failure on this lane (new §5 OPEN row).
  - **tonyd2wild "power-cut relaunch: serve fresh, don't `docker start`"** ops rule —
    ADOPTED 2026-10-12 as a README Day-2 note (owner instruction).

## Changelog

- **2026-10-19 — Site move: mgmt network changed to wireless (wlP9s9, 10.69.43.0/16;
  wired enP7s7 DOWN). First post-move boot FAILED on the old subnet; site config
  updated; boot retried.**
  - Move validation: DNS re-pointed to the wireless IPs, SSH from workstation OK to
    all 4; `verify-fabric-p2.sh` ALL CHECKS PASS (QSFP56 fabric 10.10.x unaffected —
    16 interface addresses, two-view ping mesh, both NCCL SHAs). Node-to-node
    wireless mesh pinged OK in both directions; node-side DNS resolves the new IPs.
  - First `./start.sh serve` boot: every rank's EngineCore died at vLLM
    `shm_broadcast` zmq bind — `ZMQError: Cannot assign requested address
    (addr='tcp://10.69.42.170:0')` — the launcher passes `VLLM_HOST_IP` from the
    pinned `IPS` map, which still held the old wired subnet. `FABRIC_IFACE=enP7s7`
    was also stale (interface DOWN).
  - Fix: `site/env.switchless` `IPS` → `10.69.43.170–173`, `FABRIC_IFACE` →
    `wlP9s9`; README endpoint + cluster map updated. Node `.env` mirror re-cut from
    the branch before the retry boot. No fabric/NCCL change: `IB_HCA`, netplan,
    four-PF lib all untouched (ring transport never used the mgmt NIC).
  - Result of the retry boot: recorded in the entry below if it succeeded.

- **2026-10-13 (later) — FP8 → NVFP4 switch-back per LANE-SWITCHING.md (final state:
  NVFP4 UP and healthy on `da920bb`); prefix_scan run-1 FAIL → run-2 PASS — the
  fresh-boot transient pattern REPRODUCED on a second boot.**
  - Switch: `tp4ctl down` (FP8, after its check-f0 verification — see the FP8
    research.md 2026-10-13 entries), `verify-fabric-p2.sh` ALL PASS, `git pull` all
    4 nodes (docs-only), `./start.sh serve` worker-first. Boot fast (~3 min to
    `/health` 200 — warm page-cache reload after the 5-h prior boot; cold-cache
    boot was ~8 min). `/v1/models` → `glm-5.3-flash` max_model_len 1,048,576;
    **0 NCCL WARN on every rank**; model-name-proxy healthy throughout.
  - Gates: corruption 0 U+FFFD (emoji + CJK intact); forced tool call clean
    (`get_weather({"city": "Tokyo"})`).
  - `prefix_scan`: run 1 **FAIL** ("warm-only scan anomalies", drift itself sub-limit
    0.0238/0.0377 vs 0.06, 0 warm-without-hit, solo 4/4) at ~15 min boot age; run 2
    **PASS** (0.0443/0.0638 vs limit 0.0664, 0 warm-without-hit) at ~31 min boot age.
    Same run-1-FAIL-then-settle shape as the 2026-10-13 re-pin boot — now seen on TWO
    consecutive fresh boots, always in the warm phase nearest boot traffic. Updated
    reading of the re-pin entry's "transient": this looks like a RECURRING fresh-boot
    lifecycle behavior of the warm-phase anomaly check (JIT/warm-up-adjacent), not a
    checkpoint- or config-level regression. Watch: if it ever FAILS a warmed re-run,
    that is a real signal — treat the settled PASS as the verdict and record the
    failed run. Receipts: `.research/scratch/prefix_scan-switchback-20261013*.json`.

- **2026-10-13 — FIX-3 ladder RE-RUN on `da920bb` (the re-pin's quality motivation,
  untested by the throughput benches): 16/16 probes PASS — NO quality regression;
  D1 FINALIZED: owner decision KEEP `da920bb`; D2 DECIDED: client-side per-request
  `spec_k` (no lane change). Lane untouched otherwise (HTTP-only, idle-gated; lane
  up ~5 h). Raw: `.research/scratch/fix3-repin-20261013/` (untracked).**
  - Method: identical harness to the 2026-10-12 FIX-3 run (goshi/quant-depth-quality-
    bench @ `21647f5`, same SEED 1000, same probes/order; 64K/256K/512K/998K ×
    thinking OFF/ON × quality/corruption). Same-prompt design keeps content parity
    with the `09b04e5e` baseline run for a direct A/B; the shallow thinking-ON rows
    hit prefix cache again (same caveat as before — quality is content, unaffected).
  - Results: 16/16 PASS (code task exec-checked correct + needle reproduced at every
    depth, 0 U+FFFD everywhere, `finish=stop`, no loop signatures) in BOTH thinking
    modes — byte-for-byte the same verdict as the `09b04e5e` run. The forum
    deep-context failure thread still does not reproduce, now also on `da920bb`.
  - Timing (thinking-OFF clean prefills, `09b04e5e` → `da920bb`): 998K quality
    437.0→456.8 s (+4.5%), 998K corruption 437.5→452.0 s; 512K 189→194.8 s (+3.1%);
    256K 86.1→92.6 s (+7.6%); 64K 20.8→20.7 s (flat). All four ~998K rows were full
    cache misses (consistent with the retention OPEN row — interleaved ~1M prefills
    evict each other), so these are clean prefill comparisons: a small consistent
    prefill cost on `da920bb`, directionally in line with the conc_bench finding and
    near the −2.3%…+3.8% boot-to-boot band. Not a quality or stability regression.
  - **D1 (owner, finalized 2026-10-13): KEEP `da920bb`.** Rationale: the re-pin's
    entire motivation was deep-context quality via the BF16 layer-45 — the ladder
    re-run confirms no quality regression and the throughput cost is small, c1 is
    flat, and gates are clean. Revert path (re-materialize `09b04e5e` + one restart,
    zero repo changes) stays documented in the 2026-10-13 re-pin entry if ever
    needed. §5 parked-improvements row updated.
  - **D2 (owner, decided 2026-10-13): mixed-batch json_schema cliff mitigation =
    client-side per-request `vllm_xargs {spec_k: 3}` for structured-output-heavy
    multi-stream sessions** — zero lane change, no invariant touched, reversible
    per-request (the boot log confirms per-request spec_k is honoured). A global
    `default_k: 3` is rejected (c1 json −27%); the batch-conditional k overlay stays
    PARKED unless real agent traffic shows persistent mixed-batch pain after the
    client-side mitigation (would need its own bench-recipe-update session).
- **2026-10-13 — Upstream-churn pass (both lanes; doc-only): no adoptable changes
  anywhere; one new watch item (thermal).**
  - knapcio tip still `770d115` (no new commits since 09-29). jnardiello tip moved
    `ed365a6` → `53359a2` (2026-10-01, docs-only: E37/E38 diagnostic results archived
    — no code/config; no re-vendor needed). tonyd2wild still `87aa9c0`.
    HF checkpoints: `nvidia/GLM-5.3-Flash-NVFP4` tip == our pin `da920bb`; drafter
    `bf582e4e` unchanged.
  - Forum sweep (categories 721/723 + dflash/nvfp4/switchless searches; tracked
    threads 381350/382459/382939/385075 — no new posts on the latter three since the
    2026-10-12 pass; 381350's new posts are stack-comparison chatter): one NEW
    relevant topic — **385388 "Hard thermal shutdown during prefill on long context"
    (2026-10-13)**: 2-node GB10 units hard-shutdown under sustained long-prefill GPU
    load; fan curves never hit 100%; one unit RMA'd (hardware). **Watch item opened
    (thermal, sustained long prefill):** our FIX-3 ladders are exactly this load
    class. Read-only check DURING the 2026-10-13 ladder re-run: all four nodes
    thermal zones 44.6–48.8 °C, GPUs 43–46 °C at ~2.07–2.09 GHz — cool, no
    mitigation needed. Community mitigation if ever needed: `dgx-spark-fan-control`
    systemd service / custom fan curve. Re-check only if a prefill-heavy session
    ever shows a thermal anomaly or a hard shutdown.
  - Also from 381350 chatter: rodman80's `glm-5.3-flash-w4a16-2x-DGX-Sparks` (W4A16
    INT4 + DFlash2, 2× Spark, claims zero deep-context quality loss at ~2× speed
    cost) — NOT adoptable (weight-mod premise + 2-node family); recorded as an
    alternative-quant watch row only.
  - knapcio issue #10 filed by us today (prefix retention, see the §5 row).

- **2026-10-13 — CHECKPOINT RE-PIN DEPLOYED: `da920bb` is live on all four ranks;
  BEFORE/AFTER bench verdict — quality gates clean, c1 flat, a small consistent
  concurrency-side cost (−4…−11% aggregate at c4–c16) — owner decides keep vs revert
  (revert = re-materialize `09b04e5e` + one restart; zero repo changes needed).**
  - **Deploy (no node repo change needed):** `site/materialize-weights.sh` takes the
    snapshot as an explicit arg, so the pin flip is a runtime materialization — lane
    torn down first (the weights dir is bind-mounted; the rm-rf-rehardlink must never
    run under a live container), script run ×4 pointing at the
    `da920bb0b9f4a06727223a349e55468e38352348` snapshot (present in all 4 nodes' HF
    caches, owner-downloaded; 42 files, config `41db2811…`, hf_quant_config
    `1bc0d896…`), preflight ALL PASS, `./start.sh serve` worker-first. Boot: /health
    200 (~7 min), 0 NCCL WARN, RouteFinal four PFs all ranks, KV pool 3,207,651 /
    3.06× (identical to BEFORE — expected, shards byte-identical); container
    `/model/config.json` verified `41db2811…` = da920bb (rule-5 proof the new pin is
    mounted). Zero-mod premise re-verified live pre-deploy: 33 shards byte-identical
    to `09b04e5e` (blob-level), only the layer-45 ignore additions 132→134 in both
    JSONs.
  - **Gates:** corruption 0 U+FFFD + emoji/CJK intact; forced tool call clean.
    `prefix_scan` ran TWICE: run 1 **FAIL** (cold-warm drift 0.2341 vs limit 0.06,
    cold-cold 0.0318 fine; no errors/degenerates/warm-misses), run 2 **PASS**
    (0.0593/0.0249 vs limit 0.089 — cold-warm identical to BEFORE's settled 0.0246).
    Verdict: the run-1 spike did not reproduce and is recorded as a boot-lifecycle
    transient (its warm phase ran closest to the gate traffic); the settled value
    matches BEFORE exactly.
  - **accept_probe c1 (n=6×512 tok each, deterministic):** prose 2.237→2.243 accept /
    51.3→51.3 ms / 43.9→43.8 tok/s (flat); code 4.481→4.311 accept (−3.8%) / 59.1→59.1
    ms / 75.8→73.2 tok/s (−3.4%); json 5.741→5.775 / 60.3→60.8 / 95.1→94.7 (flat).
    The code-acceptance drop is checkpoint-level (drafter unchanged, target layer-45
    numerics changed) — mechanistically consistent with the re-pin.
  - **conc_bench wall-aggregate tok/s (the discriminator), BEFORE → AFTER:**
    | cell | prose | code | json |
    |---|---|---|---|
    | c1 | 44.1→42.2 (−4.3%) | 87.2→88.0 (+0.9%) | 91.5→90.5 (−1.1%) |
    | c2 | 63.9→64.1 | 113.4→105.1 (−7.3%) | 130.9→126.6 (−3.3%) |
    | c4 | 98.6→94.9 (−3.8%) | 163.0→150.0 (−8.0%) | 200.1→187.5 (−6.3%) |
    | c8 | 141.5→133.6 (−5.6%) | 227.6→202.9 (−10.9%) | 266.5→247.5 (−7.1%) |
    | c16 | 199.4→185.4 (−7.0%) | 289.8→278.6 (−3.9%) | 332.2→314.9 (−5.2%) |
    U+FFFD 0 everywhere; json TTFT still 0.42/0.58 s at c8/c16.
  - **Verdict reading:** same-config boot-to-boot variance on this lane is
    −2.3%…+3.8% (T2 record); 12/15 cells negative and 5–6 cells outside that band —
    a REAL but modest concurrency-side cost, consistent with one more BF16
    (un-quantized) layer in the per-step path. c1 (latency-bound) is flat. Quality
    motivation for the re-pin (deep-context/layer-45) is untested by these benches —
    the FIX-3 ladder would need a re-run to probe it. **KEEP vs REVERT is the
    owner's call; the lane is left UP and healthy on `da920bb`.** Raw:
    `.research/scratch/owner-20261012/` (`repin-before*`, `repin-after*`,
    `uniformk-*`, `retain-*`).
- **2026-10-12 (late) — owner-decisions session: uniform-spec_k A/B (FIX-2 follow-up),
  KV store-vs-retain bench (knapcio #8 methodology), and the checkpoint re-pin
  PREP. Fresh BEFORE boot for matched boot states (bench-recipe-update rule 6); the
  14 h-warm prior boot was torn down. Owner pre-authorized all steps + restarts
  ("sparks are FREE", backend on remote provider). Receipts:
  `.research/scratch/owner-20261012/` (untracked).**
  - **Boot**: all-ranks teardown + `docker rm` + `./start.sh serve` worker-first;
    preflight ALL CHECKS PASS; `/health` 200 (cold ~8 min); `GPU KV cache size:
    3,207,651 tokens, Maximum concurrency for 1,048,576: 3.06x`; 0 NCCL WARN;
    RouteFinal all four PFs all ranks; `spec-probe: active; per-request
    vllm_xargs.spec_k honoured` + LeversScheduler batch-uniform confirmed in the
    boot log (rule-5 config verification). Gates: corruption 0 U+FFFD + emoji/CJK
    intact; forced tool call clean.
  - **FIX-2 uniform-spec_k A/B (owner decision #2)** — arms adaptive (control) vs
    every-request `vllm_xargs {spec_k: 3}` vs `{spec_k: 2}`, each in c4 prose-only AND
    c5 mixed (4 prose + 1 `json_schema` minItems 250) shape, 4 rounds per arm,
    rounds 2–4 analyzed (round 0 excluded), boot age 32–55 min:
    | arm | prose ms/step | mixed ms/step | mixed accept | c1 json tok/s | c1 code | c1 prose |
    |---|---|---|---|---|---|---|
    | adaptive | 26.1 (25.4–27.4) | 36.4 (35.8–37.2) — **reproduces the +37% cliff** | 3.49–3.87 | 95.1 | 75.8 | 43.9 |
    | uniform k=3 | 23.9 (23.3–24.5) | **27.0 (26.4–27.3) — cliff GONE** | 2.79–2.91 | 69.3 (−27%) | 62.9 (−17%) | 43.5 (−1%) |
    | uniform k=2 | 22.1 (21.8–22.5) | 24.0 (23.8–24.2) | 2.45–2.52 | 56.8 (−40%) | 53.3 (−30%) | 41.7 (−5%) |
    - **Bonus finding:** at batch ≥3 the adaptive policy's high state (k→7) runs the
      PIECEWISE graph (full-graph families exist only per the launch table: k7@bs1,
      k5@bs2, k3@bs3–27) — so uniform k=3 is FASTER than adaptive even pure-prose at
      c4 (23.9 vs 26.1 ms/step). The cliff is adaptive-k7 + grammar-bitmask stacking.
    - **The trade:** uniform k=3 kills the mixed-batch cliff (36.4 → 27.0 ms/step,
      −26%) and improves c4 decode, but caps single-stream acceptance — c1 json
      95 → 69 tok/s (−27%), code −17% (measured via `accept_probe --k 3` on the same
      boot). A global `default_k: 3` is therefore NOT recommended as the lane
      default (c1 json is a headline number); the data supports (a) per-request
      `spec_k: 3` for structured-output-heavy multi-stream sessions (client-side,
      zero lane change), or (b) a batch-size-conditional policy (k_hi only at bs=1)
      as a FUTURE overlay change needing its own bench session. **No config
      changed — owner decision on the trade.**
  - **KV prefix-cache store-vs-retain bench (owner decision #4)** — knapcio #8 /
    MiaAI-Lab #43 methodology, unique-content sessions, thinking off, temp 0, direct
    :8000, gauges between steps; prompts /tokenize-calibrated (A512 511,758 tok,
    A998 996,916, B998 997,949):
    | step | wall | prefix_cache hits delta | verdict |
    |---|---|---|---|
    | S1 A512 cold | 189.5 s | 0 | cold (expected) |
    | S2 A512 repeat | 1.2 s | 511,488 (99.9%) | **HIT** — store fine at 512K |
    | S3 A998 cold | 437.6 s | 0 | cold (expected) |
    | S4 A998 repeat (STORE) | 2.95 s | 995,328 (99.8%) | **HIT — ~1M prefixes store fine** |
    | S5 B998 cold | 438.6 s | 0 | cold (expected) |
    | S6 B998 repeat (STORE) | 2.25 s | 997,632 (99.9%) | **HIT** |
    | S7 A998 repeat (RETAIN) | **436.2 s** | **0** | **FULL MISS — A's prefix evicted by B's prefill** |
    - **The DeepSeek issue #43 failure shape REPRODUCES on this lane:** eviction at
      ≤78% pool occupancy (2.5M of 3.2M tokens resident after S5; headroom ≥22%),
      NOT an LRU-under-capacity event and NOT a store failure. The FIX-3 ladder's
      998K full-misses are explained by the same mechanism (interleaved ~1M
      corruption probes evicting the code prefix at that depth; 64–512K repeats hit
      because 2–3 resident mid-depth prefixes fit). New §5 OPEN row opened;
      multi-session ~1M retention ≈ 1 session until root-caused. This also RE-SCOPES
      §2 decision 4's "3.1× concurrency": true for capacity, not for concurrent
      deep-session retention.
    - Harness note: `usage.prompt_tokens_details.cached_tokens` is null on this
      build — hit/miss read from `prefix_cache_hits_total` deltas + wall time.
  - **Standard benches on the BEFORE side (re-pin A/B baseline, checkpoint
    `09b04e5e`):** `prefix_scan.py` **PASS** (drift 0.0451/0.0246 vs limit 0.0677,
    0 warm-without-hit, solo lookups 4/4); `accept_probe` c1 prose/code/json =
    2.24/4.48/5.74 accept, 51.3/59.1/60.3 ms-step, 43.9/75.8/95.1 tok/s;
    `conc_bench` wall-aggregate c1 44.1/87.2/91.5 · c2 63.9/113.4/130.9 · c4
    98.6/163.0/200.1 · c8 141.5/227.6/266.5 · c16 199.4/289.8/332.2, U+FFFD 0
    everywhere, json TTFT 0.43/0.56 s at c8/c16. Consistent with the T2/Phase-E
    band. The AFTER side (checkpoint `da920bb`) benches the same set on a fresh
    boot after the re-pin deploy.
  - The re-pin itself (materialize-weights.sh default → `da920bb`, verified
    snapshot-complete on all 4 nodes + zero-mod premise re-verified live: 33 shards
    byte-identical, only config.json/hf_quant_config.json layer-45 ignore additions
    132→134) deploys after owner merge — separate dated entry follows.

- **2026-10-12 — NVFP4 lane brought UP (handoff 06 FIX-2/FIX-3 boot; owner-authorized
  teardown/switch — "sparks are FREE", backend on remote provider). FP8 → NVFP4 switch
  per LANE-SWITCHING.md, all gates green.**
  - Switch: `tp4ctl down` on the head (FP8 `glm53_fp8_dflash_tp4` gone on all 4 ranks;
    it had been serving 13 h — the owner confirmed free use) →
    `verify-fabric-p2.sh` ALL CHECKS PASS → `git pull` on all 4 nodes (main @ `1544545`;
    the head clone was still parked on the merged-out topic branch, switched to main) →
    `./start.sh serve` worker-first from
    `glm-v53-flash-nvfp4-tp4-switchless/upstream`.
  - Derived-mirror gotcha hit (expected from the 2026-10-10 rename): `upstream/.env` is
    gitignored and lives only in the old pre-rename dir — copied
    `glm-v53-flash-nvfp4-4x-noswitch/upstream/.env` → the new upstream per the mirror
    convention (its own header documents this flow). Profile check confirms the lane
    invariants: `stock-nvfp4.env` MAX_MODEL_LEN 1048576, KV_BYTES 20 GiB/rank,
    4-PF NCCL contract (`nccl-patched-4pf` `8de68def…`), FABRIC 4-view addressing.
  - Boot: preflights PASS, containers `glm53-nvfp4-r0..3` launched ~07:55 EDT,
    `/health` 200 at ~08:03 EDT (8 min, cold-cache). `/v1/models` → `glm-5.3-flash`
    max_model_len 1,048,576 (NVFP4 signature). Boot logs: `Connected all rings` on all
    4 ranks with RouteFinal showing ALL FOUR PFs (rocep1s0f0/f1 + roceP2p1s0f0/f1);
    **0 NCCL WARN lines on any rank**; the known-boot DEVSELECT line was not observed
    this boot.
  - Gates (pre-warm-up, functional only): corruption probe 0 U+FFFD, emoji + CJK
    intact; forced tool call (`get_weather`) OK; `bench/prefix_scan.py` (97 K prefix,
    3×3 drift) **verdict PASS** (drift cold-cold 0.0622 / cold-warm 0.0456 vs limit
    0.0933, zero warm-without-hit, solo lookups 4/4 correct both phases). Receipts:
    `.research/scratch/fix2fix3-20261012/` (untracked).
  - Warm-up rule: no performance numbers before ~08:33 EDT; FIX-2 (mixed-batch
    draft-truncation cliff) and FIX-3 (1M-token gate + deep-context ladder) run on
    THIS boot after warm-up.
- **2026-10-12 — FIX-3 (handoff 06): true ~1M-token context gate + deep-context
  quality ladder — §5 row 1 CLOSED; ALL probes PASS at every depth, both thinking
  modes; the forum deep-context failure thread did NOT reproduce on this stack.**
  - Method: goshi/quant-depth-quality-bench generators @ commit `21647f5` (fetched
    fresh; stdlib), custom ladder — per (depth ∈ {64K, 256K, 512K, ~1M} × thinking
    {OFF, ON}): one exec-checked code probe (balanced-parens, RU, seed 1000, executed
    with its assertions) + one corruption probe (needle Cyrillic code token + emoji/CJK
    block embedded, reproduce + U+FFFD count). Sequential HTTP only, idle gate before
    every request, direct `:8000`. Prompt sizing calibrated against the server
    tokenizer (`/tokenize`): filler is 2.76 chars/token (the repo's 2.1 estimate is
    off for this tokenizer) — depth_k 1000 = 998,405 tokens < 1,048,576 − budget.
    Receipts: `.research/scratch/fix2fix3-20261012/` (`fix3-ladder.json`,
    `fix3-stdout.txt`).
  - **§5 row 1 gate: the ~1M-token request COMPLETED prefill+decode end-to-end twice**
    (thinking OFF 437.0 s, thinking ON 437.5 s — independent prefills: the ON run
    MISSED prefix cache, `prefix_cache_queries` +998,405 with 0 hits;
    `request_prefill_kv_computed_tokens_sum` +998,405 — nothing truncated). Wall
    prefill-dominated rate ≈ 2.3K tok/s at 998K (512K: 189 s ≈ 2.7K tok/s; 256K: 86 s
    ≈ 3.0K tok/s; 64K: 21 s). Transient KV occupancy ≈ 31% of the 20 GiB/rank pool
    (1M of ~3.25M tokens); before/after gauge samples show idle 0.0 (sampled between
    requests).
  - Quality ladder: code task CORRECT at 64/256/512/998K in BOTH modes (16/16 rows
    pass); corruption probes: needle correct everywhere, emoji + CJK reproduced,
    **0 U+FFFD at every depth/mode**, no loop signatures (max repeat ≤1, threshold
    ≥8), no special-token contamination, `finish=stop` on every row.
  - Caveat on the shallow thinking-ON rows: the ON runs immediately follow the OFF
    runs on the SAME seeded prompt — 64K–512K prefill hit the prefix cache (wall 1.5–2.8 s
    is cache-hit decode, NOT clean prefill); the two 998K rows are both clean (ON
    missed cache at that depth). The PASS/PASS results are unaffected (quality is
    content, not timing).
  - Forum t381350 non-thinking deep-context failures (≥~300K across quants incl. stock
    NVFP4) did NOT reproduce here: non-thinking was CORRECT at 256K/512K/998K on the
    exec-checked task and needle. No new watch item opened; note the thread also used
    different task sets — the parked checkpoint re-pin `da920bb` candidate (owner
    decision) remains the next lever if a future probe disagrees.

- **2026-10-12 — FIX-2 (handoff 06): mixed-batch draft-truncation cliff check
  (knapcio issue #7 tip 1) — CLIFF PRESENT via `response_format: json_schema` (NOT via
  `tool_choice: "required"`, which fast-paths on this stack). No config change; the
  workaround goes to a bench-recipe-update PLAN (owner decision).**
  - Method: A/B at c4, temp 0, `ignore_eos`, ~3.5K-token prose in / 500 out, idle gate
    (`num_requests_running/waiting == 0`) before every round, 3 rounds each; step math
    from `/metrics` deltas (`spec_decode_num_(drafts|accepted_tokens)_total`), direct
    `:8000` (no proxy), boot age ~35 min at first round. Receipts:
    `.research/scratch/fix2fix3-20261012/` (`fix2-rounds.json`, `fix2-verdict.json`,
    `fix2-control-c5.json`, stdout captures).
  - Run A (4 prose): ms/step 26.9/27.6/26.2 (mean **26.5**), accept/step 3.23–3.57,
    prose streams 33–40 tok/s.
  - Run B (4 prose + 1 grammar-constrained): **36.0/36.8/36.4 ms/step (mean 36.4,
    +37.2% vs A)**, prose streams drop to 27–33 tok/s (≈ −20%).
  - Run C control (5 concurrent UNCONSTRAINED prose — batch-size confound check):
    ms/step 22.9/23.4/22.7 (mean **23.0**), accept 3.25 — batch size 5 does NOT slow
    steps; the inflation is the constrained stream itself. One grammar-constrained
    (json_schema, `minItems: 250` so it decodes a full 800-token window) request
    inflates EVERY stream's step time ~+37% (vs c5: +58%).
  - Mechanism note (differs from knapcio's "drafts truncated for every stream"): the
    global accept/step does NOT collapse in B (3.57–3.75, propped up by the highly
    predictable constrained stream) — on THIS stack the cost is a slower verify step
    while a grammar-constrained stream is decoding (grammar-bitmask application per
    step across the batch), not lost drafting. `tool_choice: "required"` does not
    trigger it here: the request fast-paths (one chunk, 0.4 s wall, single
    tool-call delta, never joins the decode batch) — knapcio's trigger does not
    transfer as-is; `response_format: json_schema` is the live agent-serving trigger
    (structured outputs), and constrained JSON with `minItems`/long arrays buffers
    delivery (chunks at end) while decoding the full window.
  - Bench-recipe-update PLAN (NOT adopted; owner decision): A/B the uniform-draft-length
    workaround on this stack — same FIX-2 harness, all streams forced to one `spec_k`
    via the lane's `vllm_xargs: {spec_k}` override (e.g. k=2, k=3) vs the default
    adaptive policy, mixed constrained batch, acceptance + ms/step + tok/s receipt.
    Touches nothing at rest; runs as a bench candidate only.
  - Related serving guidance (no change): agents mixing thinking prose with forced
    structured outputs share the batch, so one constrained request taxes every
    concurrent stream ~20% while it decodes — bounded by the constrained stream's
    length.

- **2026-10-12 — FIX-1 (handoff 06): host CPU floor-clock check (knapcio issue #7) —
  ALL FOUR NODES HEALTHY. Question closed; no owner decision needed.**
  - Method: knapcio's CPPC feedback-counter probe (`/sys/devices/system/cpu/cpuN/
    acpi_cppc/feedback_ctrs`, busy loop pinned to the core, counters diffed across 2 s;
    reference counter assumed 1 GHz). Read-only SSH, zero lane interaction.
  - Core layout on these nodes (cpuinfo_max_freq per cpu): cpu0–4 + cpu10–14 =
    Cortex-A725 (max 2,808,000 kHz), cpu5–9 + cpu15–19 = Cortex-X925 (max 3,900,000
    kHz) — 10+10, same X925 numbering window as knapcio's units (cpu5–9/15–19 big).
  - Delivered frequencies under load (one X925 + one A725 core per node):

    | node | kernel | X925 delivered | A725 delivered |
    |---|---|---|---|
    | spark-0f0b | `6.17.0-1032-nvidia` | 3893 MHz | 2805 MHz |
    | spark-6d14 | `6.17.0-1032-nvidia` | 3896 MHz | 2805 MHz |
    | spark-6d90 | `6.17.0-1032-nvidia` | 3896 MHz | 2803 MHz |
    | spark-6d24 | `6.17.0-1032-nvidia` | 3894 MHz | 2804 MHz |

  - Reading: healthy ≈ 3900 / 2800 — every node delivers ~99.8% of max on both
    clusters. The floor-clock signature (~1378 / ~1013 MHz on affected 6.17 units) is
    ABSENT on this ring, even though our kernel `6.17.0-1032-nvidia` is in knapcio's
    affected 6.17 family (their units: DGX OS 7.5 / `6.17.0-1031-nvidia`). The kernel-7.x
    fix remains banned (ring-NCCL invariant) and is now moot for clock reasons —
    this cluster does not exhibit the symptom it fixes. No runtime change attempted
    (none needed); future benches inherit this receipt and do not need to re-run the
    check unless the kernel/driver pin changes.

- **2026-10-12 — Upstream pass (update-recipe sweep, doc-only — no config/pin changed):
  checkpoint tip moved past our pin, a deep-context quality thread, and three knapcio
  issues; nothing adoptable as config.** Full facts verified live 2026-10-12; raw
  artifacts in `.research/scratch/` (`config-09b04e5e*.json` / `config-main.json`,
  `hf_quant_config-*.json`, `pin_tree.json` / `tip_tree.json`, `knapcio-issue7.md`).
  - **nvidia/GLM-5.3-Flash-NVFP4 main moved past our pin:** pinned `09b04e5e`
    (2026-09-11) → tip `da920bb` (2026-09-29 21:30). Tree-level diff: ONLY `config.json`
    + `hf_quant_config.json` changed — both add `"model.language_model.layers.45*",
    "model.layers.45*"` to the ignore list (layer 45 no longer NVFP4-quantized, i.e.
    served BF16). All 33 weight shards byte-identical (oid + size compared, `.research/
    scratch/pin_tree.json` vs `tip_tree.json`); the zero-mod premise is intact at the
    new rev. **PARKED as an improvement candidate, NOT adopted** — owner decision:
    re-download + bench-recipe-update before any re-pin; possibly connects to the
    deep-context quality findings below (a BF16 layer-45 may be a targeted quant fix).
  - **Forum t381350 deep-context quality thread (posts 2026-10-11 → 10-12):**
    ajvazan's `github.com/goshi/quant-depth-quality-bench` and technigma's screening
    show quality collapse at deep context (≥~300K tokens) in NON-thinking mode across
    quants INCLUDING stock vLLM NVFP4, with failures disappearing in reasoning mode.
    Not adoptable as config — it motivates the deep-context quality extension of the
    1M-token gate (FIX-3, this session).
  - **knapcio issue #7 (2026-10-02, host CPU floor clocks):** on ASUS GX10 / DGX OS 7.5
    / kernel `6.17.0-1031-nvidia`, Grace CPU cores can deliver their FLOOR frequency
    (~1.38 GHz X925 / ~1.0 GHz A725 vs 3.9/2.8 max) while `scaling_cur_freq` reports
    full speed; no runtime fix works on 6.17 — the fix is kernel 7.0 (+8–10%
    single-stream decode, GPU busy 82–90% → 94–95%). **Our nodes run
    `6.17.0-1032-nvidia` — the affected family** (verified live 2026-10-12, all four
    `uname -r`). The kernel fix is BANNED on this cluster (7.x breaks ring NCCL — FP8
    lane invariant), so the per-node delivered-MHz check (FIX-1, this session) is
    diagnosis-only; a positive finding is an owner decision, not a runtime change.
    Full issue text archived at `.research/scratch/knapcio-issue7.md`.
  - **knapcio issue #7 tip 1 (mixed-batch draft-truncation cliff):** in the TP4
    community recipe (vLLM + draft-model spec decoding with per-step draft truncation),
    ONE request with `response_format: json_schema` or `tool_choice: "required"`
    switched off draft-length truncation for EVERY stream in the running batch (4
    streams 65 → 103 ms/step, 31.5 → 22 tok/s each); `tool_choice: "auto"` did not
    trigger it. Our lanes serve agents that mix thinking prose with forced tool calls —
    if the cliff exists here it is a live deficit. Note the lane's all-json c1 bench
    (96.3 tok/s) does NOT test this; only a MIXED batch does → FIX-2 A/B check (this
    session).
  - **knapcio issue #8 (KV prefix-cache store-vs-retain bench methodology, via MiaAI-Lab
    DeepSeek issue #43):** PARKED as a bench-method improvement candidate for the
    ~3.25M-token concurrency claim — owner decision, not adopted this session.
  - **knapcio issue #9 (supporting citation only):** a third party hardware-qualified
    `TRANSPORT=switchless` on a physical 4-node ring, 4 RDMA functions/node, all
    addressed — same shape as this lane's ring. Cited alongside the ring-decode row
    below (§5); no action.
  - **tonyd2wild tip moved `96658cf0` → `87aa9c0` (2026-10-10):** adds DERISKED
    uncensored weights on knapcio's stack — **REJECT for this lane** (weight-mod
    premise, §2 decision 5). His "power-cut relaunch: serve fresh, don't `docker
    start`" ops rule parked as a README Day-2 note candidate (owner decision; low
    priority — the lane's normal relaunch path already recreates containers).
  - **Standing watch items unchanged:** GLM_ROCE_ALLREDUCE / GATHER_ROUTE hardware
    qualification (§5); upstream-churn periodic re-run; owner merge-gate on the working
    branch. knapcio tip STILL `770d115` — no re-vendor needed.

- **2026-10-11 — T3a ROOT-CAUSED: `DEVSELECT_BUILD_FAILED` is an upstream platform
  limitation, not a lane/overlay bug. Watch item CLOSED (understood; no fix reachable on
  this stack).** Diagnosis-only session (handoff
  `.research/handoffs/01-nvfp4-devselect-rank0.md`); FP8 was serving → log capture +
  standalone ring probes, no lane restart, no overlay/weight touch. Probe scripts +
  raw logs archived under this recipe's `.research/` (untracked); node scratch cleaned.
  - **Correction of the T3a / Phase-E record: the failure is NOT rank-0-only.** The
    preserved 2026-10-10 20:22 boot logs (T2 boot; all four exited containers still
    inspectable) show `DEVSELECT_BUILD_FAILED` exactly **once on EVERY rank**, all within
    40 ms (r0 20:22:30.297Z, r2 .320, r3 .332, r1 .336), byte-identical tracebacks. The
    "ranks 1–3 clean" observation was a worker-log gap. It therefore predates Phase E
    trivially: the device-select feature has NEVER armed on this driver stack.
  - **Failure point (exact, rank 0):**
    ```
    glm-devselect: target graph M=8 node types {'Kernel: 0>': 1594, 'WaitEvent: 6>': 2, 'Host: 3>': 101, 'EventRecord: 7>': 202, 'Memcpy: 1>': 17}; int-ws extents {2: 856064, ...}; GDN qsl buffers 4; indexer builders 1; slot groups 7; ctx groups 1
    glm-devselect: DEVSELECT_BUILD_FAILED AcceleratorError: CUDA error: invalid argument
      File "/overlay/overlay/glm_devselect.py", line 1458, in capture_model   -> build(self)
      File "/overlay/overlay/glm_devselect.py", line 596, in build            -> with torch.cuda.graph(parent, pool=mgr.pool):
      File "torch/cuda/graphs.py", line 477, in __exit__                       -> self.cuda_graph.capture_end()
    torch.AcceleratorError: CUDA error: invalid argument
    ```
    The whole body (select Triton kernel + 7× begin_capture_to_if_node / fixups /
    child-graph embed / end_capture) ran without a Python exception; the failure is
    strictly at `capture_end`. Consequence per rank: `_D.failed` set → `eligible()`
    False forever → the parent never launches (`glm-draft-trunc: v0 steps N: … dev 0`
    at every status line; devselect TIMELINE shows only `v0H` = host-chosen graph,
    20k steps, zero parent launches). Decode continues on the proven host path;
    benches unaffected — the fallback IS the lane's validated baseline.
  - **Root cause (isolated on the live 4-node ring with the lane DOWN, 4-rank torchrun
    in the lane image with the exact NCCL-4pf contract + `LD_PRELOAD` of
    `libnccl.so.2.30.7`; tiny tensors):** a stream-captured CUDA-graph **parent holding
    conditional IF nodes rejects a child-graph node whose graph contains NCCL
    collectives** — `cudaStreamEndCapture` returns `cudaErrorInvalidValue` (= the boot
    signature). Full matrix (all 4 ranks agree in every cell):
    | case | result |
    |---|---|
    | plain parent + NCCL-child (embedded instantiated graph w/ all_reduce) | PASS |
    | IF-parent, no child embed | PASS |
    | IF-parent + NCCL-child — instantiated OR uninstantiated | **FAIL invalid argument** |
    | IF-parent + NCCL-child captured direct in the body (no child node) | **FAIL** |
    | IF-parent + D2D-only child (no collective) | PASS |
    | IF-parent + NCCL-child under `capture_error_mode` global / thread_local / relaxed | FAIL in all three |
    | same, with a concurrent allocating background thread | FAIL (noise-independent) |
    | IF-parent + NCCL-child with NCCL graph registration env off (`NCCL_GRAPH_REGISTER=0`, `NCCL_CUDA_GRAPH_REGISTER=0`) | FAIL |
    | **stock** torch-bundled NCCL 2.30.7, NET=Socket TCP, no 4pf lib: IF + NCCL-child | **FAIL identically** |
    The stock-NCCL cell rules out our 4pf patch stack: the rejection is upstream —
    CUDA 13.0 driver / NCCL 2.30.7 on GB10 — not config/env/transport. Note the earlier
    probe-campaign OOM incident (owner record in the FP8 recipe's research.md,
    2026-10-11) happened during the offline phase of this same campaign; probe45-style
    VRAM exhaustion was the cause; every probe since ran within a few-MB budget.
  - **Manual (non-capture) parent construction: dead end on CUDA 13.0.** CUDA 13 moved
    conditional nodes to the handle API (`cudaGraphConditionalHandleCreate` +
    `cudaConditionalNodeParams{handle, type, size}`); cuda-python 13.3.1 exposes **no
    device-condition binding** (no `conditionData`/device-pointer field, no
    `cudaGraphConditionalAddNode`), so `glm_devselect`'s premise — the IF condition is a
    device-computed L written by the captured select kernel — cannot be expressed
    without stream capture. Capture-time conditionals (torch `begin_capture_to_if_node`)
    are the only binding path, and that path is what fails with NCCL children.
  - **Decision: no fix lands.** No env/config knob exists (mode/registration/noise all
    ruled out); a `glm_devselect.py` rewrite cannot express the semantics under CUDA
    13.0; driver/CUDA are pinned by the NCCL contract. The overlay's catch-and-fallback
    is working as designed and the fallback is the validated baseline — the ×1-per-rank
    `DEVSELECT_BUILD_FAILED` line at boot is now EXPECTED and understood, zero action.
  - **Re-verification protocol (when the kernel/driver pin ever changes):** boot the
    lane normally; `docker logs glm53-nvfp4-r0 | grep DEVSELECT_BUILD_FAILED` — if the
    line is GONE, the upstream limitation is fixed: confirm arming via
    `glm-draft-trunc: v0 steps` status lines showing `dev` > 0 and devselect TIMELINE
    rows `v0P` (parent ran), then bench. If still present, re-run the archived probes
    (`.research/nccl_embed_probe*.py`) to re-classify.

- **2026-10-10 — RECIPE RENAMED:** `glm-v53-flash-nvfp4-4x-noswitch` →
  `glm-v53-flash-nvfp4-tp4-switchless` (owner reorg; the FP8 sibling renamed to
  `glm-v53-flash-fp8-tp4-switchless` in the same commit). The working BRANCH keeps its
  `glm-v53-flash-nvfp4-4x-noswitch` name. Docs updated to the new paths; historical
  entries below keep their period names. Node-side: apply the rename at a lane switch
  (containers bind-mount paths under the recipe dir — see LANE-SWITCHING.md gotchas).

- **2026-10-10 — CROSS-LANE VALIDATION COMPLETE: switch FP8 → this lane PASS, T3 watch
  items closed.** (T1 — the FP8 lane's own four-PF boot — is recorded in the FP8
  recipe's research.md; this lane was torn down for it and re-adopted after.)
  - **T2 switch-back (FP8 → this lane): PASS, parity with the Phase-E record.**
    Maintenance window: `tp4ctl down` + container rm (FP8), `verify-fabric-p2.sh` ALL
    PASS, `./start.sh serve` worker-first 3,2,1,0. Warm boot (`/health` 200 ~6 min).
    Boot markers identical to Phase E: `Connected all rings` ×2/rank, RouteFinal with
    all four PFs on every rank (f0/f1/P2f0/P2f1). Gates: corruption probe 0 U+FFFD;
    forced `tool_choice:required` clean; `prefix_scan.py` **PASS** (cold-cold drift
    0.0517, cold-warm 0.042 vs limit 0.0776 — well away from the limit, vs Phase E's
    0.1367/0.0392 vs 0.205; misquotes 0/1; scan-incorrect 5/6+6/6 — the known watch
    item, slightly better than Phase E's 6/6+6/6 and the staged boot's 7/7; warm hit
    96,768 tokens). accept_probe c1: prose 45.0 (+0.7%), code 77.3 (+2.0%), json 95.2
    (−1.1%) vs Phase E. conc_bench full ladder within −2.3%…+3.8% of Phase E (c16
    190.6/282.4/316.8 vs 192.6/283.2/322.3), U+FFFD 0 everywhere, json TTFT still
    0.47/0.58 s at c8/c16. **The lane switch is clean and the Phase-E transport numbers
    reproduce.** Raw: `bench-results/20261010-t2-switchback/`.
  - **T3a — `DEVSELECT_BUILD_FAILED` RECURS: watch item stays OPEN.** (ROOT-CAUSED
    2026-10-11, Changelog entry below: upstream platform limitation, all four ranks,
    feature never armed on this driver stack — this entry's "ranks 1–3 clean" was a
    worker-log observation gap.) Fresh boot:
    1× on rank 0 (ranks 1–3 clean) — same signature as the Phase-E boot (which showed
    ×2 on r0): `glm_devselect.py:596 build` → `torch.cuda.graph capture_end` →
    `AcceleratorError: CUDA error: invalid argument`, caught by the overlay, decode
    continues on the dev-0 fallback; benches unaffected. Unknown whether it predates
    Phase E (the staged-boot containers were removed before log comparison); it is
    reproducible at boot on rank 0 — investigate the devselect pool/capture order
    separately from transport work.
  - **T3b — prefix_scan cold-cold drift: CLOSED (no flag).** Fresh 0.0517 vs Phase E
    0.1367 (limits differ per run: 0.0776 vs 0.205 — the tool computes its own limit
    from the run's spread); trending AWAY from the limit, treat the Phase-E 0.1367 as
    first-boot-of-the-lib variance, not a trend.
  - Branch does NOT merge yet — explicit owner gate (T1 PASS + T2 PASS + T3 recorded
    are met; the merge itself is the owner's call).
- **2026-10-10 — PHASE E COMPLETE: serving on the full four-PF NCCL contract.** The
  staged 2-PF transport (previous entry) is replaced by the real four-PF build; both
  staged-transport site mods to `upstream/scripts/transport.sh` are REVERTED — the
  switchless render's functional lines now match upstream `770d115` byte-for-byte
  (`NCCL_SWITCHLESS_RING_ONLY=1`, `NCCL_IB_MERGE_NICS=0`, no `SKIP_TREE_CONNECT`).
  - **NCCL build (site/nccl-4pf/, fully reproducible via `build-4pf.sh`):**
    othexmr/switchless-nccl @ `94c2669` patch stack on NCCL `v2.30.7-1` @ `73cf1122…`
    (base tree `3e7de6f9…`), patch order skip-tree-pat → advertise-all-listener-gids →
    hardened-switchless → dual-pf.site, final patched tree `cf677895…`. The ONLY
    site-modified patch is `nccl-2.30.7-dual-pf.site.patch` (SHA `334664f9…`): a one-line
    `tp4ValidateListener` GID-range change `10.100.224–227` → `10.10.1–4` so the validator
    accepts this ring's addressing (contract otherwise unchanged: 2 cable subnets/node ×
    2 PCI roots, full cross product, `gids[15] ∉ {0,255}`). Built on spark-6d14 in docker
    (`nvidia/cuda:13.0.2-devel-ubuntu24.04`, sm_121): **lib SHA256 `8de68def…a41a87`,
    61,426,528 bytes, 165 symbols**. Identity check (`check_switchless_nccl.py`) PASS ×4
    after staging to `~/.local/lib/nccl-patched-4pf/` on all 4 nodes (real file
    `libnccl.so.2.30.7` + `libnccl.so.2` symlink).
  - **Separate staging directory (`nccl-patched-4pf`, NOT the FP8 recipe's
    `nccl-patched`):** the four-PF lib's hardened switchless path triggers on
    `NCCL_SWITCHLESS_RING_ONLY=1` alone and hard-rejects `NCCL_IB_MERGE_NICS != 0`; the
    FP8 launcher sets only `SKIP_TREE_CONNECT=1` with default merged vdevs. Swapping in
    place would break the FP8 lane's next boot. The FP8 dir keeps `afe5f486…` untouched;
    rollback for this lane = one-line `NCCL_HOST_DIR` revert + re-apply the staged
    transport mods (documented in the transport.sh comment).
  - **Netplan (host-side, mirrored as site/netplan/<node>/40-cx7.yaml):** every cable's
    /24 gains one address per node on the second PCIe view, last octet = f-port octet + 2
    — additive over the FP8 plan (f-port addresses unchanged), so the FP8 lane's 2-PF
    addressing keeps working. Map: L1 rank0 f=.1/P2=.3, rank1 f=.2/P2=.4; L2 rank1
    f=.2/P2=.4, rank2 f=.3/P2=.5; L3 rank2 f=.3/P2=.5, rank3 f=.4/P2=.6; L4 rank3
    f=.4/P2=.6, rank0 f=.1/P2=.3. Deployed 4/4 (backup `/etc/netplan/40-cx7.yaml.bak-phaseE`
    per node), verified 16/16 pings across both PCIe views per cable. **FP8-lane hazard:
    never re-deploy the FP8 recipe's 2-address netplan — it strips the P2 addresses and
    breaks this lane (noted in the FP8 recipe's research.md).**
  - **Boot verdict:** preflight identity PASS ×4 → `Connected all rings, use ring PXN 0
    GDR 0` on every rank → RouteFinal diagnostics show **all four PFs carried QPs on
    every rank** (`rocep1s0f0/f1` + `roceP2p1s0f0/f1`), P2-view GIDs (10.10.x.3/.4) in
    actual QP routing, subnet-aware routing overriding with PCI-root preservation
    (`pci0000:00` / `pci0002:00`). Engine init 51.7 s warm (JIT caches hot). `/health`
    200, `glm-5.3-flash`, max_model_len 1,048,576.
  - **Gates:** corruption probe 0 U+FFFD (emoji + 海洋 intact); forced
    `tool_choice:required` → clean `get_weather({"city": "Tokyo"})`; `prefix_scan.py`
    **PASS** (tool's own score verdict, exit 0: drift 0.1367 cold-cold / 0.0392 cold-warm
    vs 0.205 limit, misquotes 1 cold / 0 warm, scan-incorrect 6/6+6/6 — the known watch
    item, baseline was 7/7; degenerate 0, errors 0, warm hit 96,768 tokens, sizing
    P≈97.4K mod 2304 in-window).
  - **accept_probe c1 (k=7 family):** prose 2.22 / 50.0 ms-step / **44.7 tok/s** (+2.1%);
    code 4.36 / 57.6 / **75.8** (−0.7%); json 5.76 / 59.6 / **96.3** (−0.8%) — flat, as
    expected: c1 is latency-bound and the extra PFs add aggregate bandwidth, not
    single-stream RTT.
  - **conc_bench full ladder (wall-aggregate tok/s, U+FFFD 0 everywhere):** c1
    41.9/86.9/92.5 · c2 62.8/108.3/130.9 · c4 97.9/157.4/190.3 · c8 139.1/206.4/
    **252.0** · c16 192.6/283.2/**322.3**. vs the 2-PF baseline: json c8 **+26%**,
    json c16 **+15%**, everything else −1% to +6%. **The baseline's json TTFT watch item
    (~6.5 s at c8/c16) is RESOLVED: 0.48/0.58 s** (prefill-phase collectives spread over
    the four PFs). Raw artifacts: `bench-results/20261010-phaseE-4pf/`.
  - **Verdict:** the lane-defining c1 gap vs the switched fleet is UNCHANGED (latency-
    bound); the 4-PF transport pays off at concurrency and prefill TTFT. The lane stays
    on the zero-mod NVFP4 stack; the lossless8 fallback is not triggered. Numbers taken
    ~50 min after boot, after prefix_scan + accept traffic (fully warm).
  - **Watch item (new):** `glm-devselect: DEVSELECT_BUILD_FAILED` ×2 at boot (cuda graph
    `capture_end` invalid argument, caught by the overlay; decode continues with the
    dev-0 fallback; benches unaffected). (ROOT-CAUSED 2026-10-11, Changelog below:
    upstream platform limitation on every rank; the ×2-on-r0 reading was the same
    worker-log gap.) Unknown whether it predates Phase E (staged-boot
    containers were removed before logs could be compared) — check on the next reboot.
  - Raw build/deploy trace: site patch GID line =
    `gids[i][12]!=10 || gids[i][13]!=10 || gids[i][14]<1 || gids[i][14]>4`;
    build log `/home/sdrew/nccl-4pf-build/build.log` on spark-6d14 (scratch dir, not repo).
- **2026-10-10 — update-recipe pass vs tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark.**
  Full review of his repo (README @ `96658cf0` 2026-09-30 tip, runs/2026-09-29-knapcio-stack +
  -effort-low, docs, issues #1–#12) plus a forum sweep (t382459 jnardiello switchless,
  t382939 2x-optimization, t381350 main GLM thread, t385075 kindling TP=3; posts after
  2026-09-29). Headline findings:
  - **His throughput headline is knapcio's lossless8 stack, not a config we lack.** His
    2026-09-29 default runs knapcio's stack unmodified at pin `770d115` — our exact vendored
    pin — but through the lossless8 weight conversion (8-bit dense grids + fp8blk drafter),
    measured against HIS OWN previous lane (nvidia-glm53-attn, NVFP4-quantized attention).
    The +36–57% prose decode is mostly that weight conversion; our lane deliberately serves
    the pristine pack (§2 decision 5), so the delta is already priced into §3's strip list
    (~10–15%). No engine-level change exists in his repo to adopt: knapcio's tip is still
    `770d115`, and our `stock-nvfp4.env` inherits `current.env` live, so every kernel/scheduler
    win from that profile already reaches this lane.
  - **ADOPTED then REVERTED same day — `DEFAULT_EFFORT=low`** (in `stock-nvfp4.env` site
    section). His 2026-09-29
    A/B on the same stack (runs/2026-09-29-knapcio-effort-low): 69-scenario tool-eval quality
    93.5 low vs 92.8 high; output tokens 631 vs 978, thinking text 227 vs 602 chars; median
    answer 1.3 s vs 1.8 s; decode tok/s unchanged (69.5 vs 70.1). This lane serves agents —
    fewer burned thinking tokens is time-to-answer, the metric agents feel. Current.env ships
    `high`; our bring-up benches ran at high, so accept_probe/latency baselines shift
    slightly on the next bench. Frame honestly: tok/s does NOT rise; tokens-wasted falls.
    **Owner reverted to the inherited `high` the same day** — deeper default thinking is
    wanted; per-request `chat_template_kwargs {"reasoning_effort": "low"}` remains available
    for short answers. A comment in `stock-nvfp4.env` records the finding and the one-line
    revert path so a future pass doesn't re-research it.
  - **ADOPTED (docs) — warm-reboot cache note** in README Day-2 ops: fresh-overlay relaunch
    should copy the old overlay's `cache/` first (his measurement: /health ~200 s warm vs
    ~542 s cold on the same stack). Our normal relaunches reuse `~/.local/nvfp4-tp4/`, so this
    only bites when a fresh `OVERLAY_REMOTE` is forced.
  - **RECORDED (dead ends, don't spend a boot):** his "tried and did not keep" table — a
    deeper draft table keeping all depths 1–7 (±3% every cell) and the `batch-max` scheduler
    mode (±2%) are both measured no-ops on the same stack. Also recorded: `TRUNC_COST=r12` is
    untried by anyone (reject — no measured claim); uncapped GPU clocks gave him +4–8% prefill
    but he keeps the 2200 MHz cap for hard-power-off mitigation (watch item, not adopted);
    keep the SPEC_TABLE's every-depth-1–7 shape — his runbook gotcha: replacing it with a
    compact table broke DEVSELECT graph capture and halved single-stream prose.
  - **REJECTED — weight-mod lanes:** his `nvidia-glm53-attn` (NVFP4 attention on the 403
    BF16 tensors, needs `NVFP4_PATCH=1` glm5next mounts) and knapcio's lossless8 both
    modify weights — contradicts this lane's premise (§2 decision 5). External validation
    of the fallback lane's expected size: vedcsolution's TP2 eugr PR #421 (forum t382939
    #187) reports +23–27% decode from online-NVFP4 of the BF16 dense layers — consistent
    with §3's ~10–15% strip-list estimate direction (different lane/topology, not a receipt
    for this stack).
  - **REJECTED — lane parameters:** his 500K window / 24 GiB KV are fleet choices for the
    lossless8 pack (43.76 GiB/rank weights vs our ~47.6); 1M ctx + 20 GiB pin here are owner
    decisions (§4). His `MNBT=8192`/`max-num-seqs 64` findings are from the older fp8/marlin
    lane and he explicitly flags them as not re-measured on the current one.
  - Forum sweep: nothing adoptable landed after 2026-09-29 in the GLM threads (t382459,
    t382939, t381350, t385075 are kindling/TensorFold/eugr/technigma stacks on other
    topologies; the tool-loop thinking-loss bug discussed there is a different stack's chat
    template — verify our lane still passes the tool-call gate at the next bring-up).

- **2026-10-10 — FIRST BRING-UP COMPLETE: serving on the 4-node ring.** Staged transport
  (2 PFs/node + site SparkRing lib). Boot → `/health` 200 in ~7 min cold (JIT caches
  populate during first traffic; no model conversion steps). Record:
  - **NCCL boot failure #1 (fixed as site mod #2, `upstream/VENDORED-AT.md`):** knapcio's
    switchless render forces `NCCL_IB_MERGE_NICS=0` → two separate virtual devices → NCCL
    pairs channel c → vdev (c mod 2) with no subnet knowledge → the rank3→rank0 channel
    used 10.10.1.x at rank 0 against 10.10.4.x at rank 3 → `ibv_modify_qp` 110 timeout →
    `Connected` never reached, engine init failed. Fix: `MERGE_NICS=1` (FP8 launcher's
    proven value) — one merged vdev + the site lib's subnet-aware routing picks the PF per
    connection. After fix: `Connected all rings, use ring PXN 0 GDR 0` on all ranks;
    merged vdev `rocep1s0f0+rocep1s0f1` speed=400000 ndevs=2.
  - **First-boot gotchas fixed on the way:** `start.sh` missing +x (index fix, `3ac8622`);
    `stock-nvfp4.env` strip leak (EXTRA_ENV line-start switches `QMIX_FP8_BLOCK=1` and
    `GLM_DENSE_FAST_PREFILL=1` reached the containers — caught by the DRY render, fixed
    by flattening EXTRA_ENV before the strip, `7afa37b`); `bench/prefix_scan.py` wants
    `--base` WITHOUT `/v1` and `--model glm-5.3-flash` (its default is upstream's
    `GLM-5.3-Flash-FP8`).
  - **Gates:** corruption probe 0 U+FFFD (emoji + 海洋 intact); forced `tool_choice:required`
    → clean `get_weather({"city": "Tokyo"})` (TC-45 watch item: no structural-tag override
    needed here — the upstream glm47 parser handles it); `prefix_scan.py` **PASS**
    (0 errors, 0 degenerate, 0 false-anomalies; drift 0.0338 cold-cold / 0.0347 cold-warm
    vs 0.06 limit; 96,768-token warm prefix hits; sizing P≈97.4K, mod 2304 in-window).
  - **accept_probe (c1, k=7 family):** prose accept 2.20 / 50.3 ms-step / **43.8 tok/s**;
    code 4.40 / 57.9 / **76.3**; json 5.79 / 59.5 / **97.1**. pos-0 acceptance
    0.638 / 0.882 / 0.97 — healthy, gentle per-position decay.
  - **conc_bench (full ladder, wall-aggregate tok/s prose/code/json, U+FFFD 0 everywhere):**
    c1 43.8/87.4/91.6 · c2 62.1/102.8/128.6 · c4 92.4/155.1/189.6 · c8 133.6/204.6/199.8 ·
    c16 192.2/282.6/280.7.
  - **The lane-defining number:** ring decode is ~45–55% of knapcio's switched fleet
    (c1 90.3/124.0/168.8 → 43.8/76.3/97.1; c16 331–879 → 192–342). Two stacked causes:
    (a) ring transport (NCCL ring latency + opposite-node transit; a 200G/cable four-PF
    upgrade is Phase E's lever), and (b) this lane's zero-mod strip list (~10–15% per
    §3). If Phase E doesn't close the gap, the lossless8 fallback decision comes here.
  - **Watch items:** prefix_scan misquotes 2/1 + incorrect 7/7 (non-strict PASS; consider a
    `--strict` rerun); json TTFT ~6.5 s at c8/c16 (structured prefill chunking under the
    6919-token budget — monitor); numbers taken ~30 min after first traffic (warm-up
    penalty mostly paid but not fully).
  - Endpoint verified end-to-end through the proxy: `spark-llm` → real backend
    `glm-5.3-flash` (mode A).

- **2026-10-10 — pre-bring-up review + staged transport decision.** Full review against
  the live 4-node cluster before first activation. Findings and fixes:
  - **`FABRIC_IFACE` was wrong**: `site/env.switchless` had `enp1s0f0np0` (a fabric PF,
    10.10.1.1 — reaches only rank 1). `transport.sh` renders it into
    `NCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME`/`MN_IF_NAME`/`TP_SOCKET_IFNAME`, which must
    reach all ranks → fixed to `enP7s7` (the mgmt NIC, same value the FP8 launcher uses;
    verified live: 10.69.42.170/16).
  - **NCCL artifact gap pinned down.** Strings analysis of the site lib (`afe5f486…`,
    josephdrose/SparkRing patch, FP8-proven): implements `NCCL_SKIP_TREE_CONNECT` +
    `NCCL_IB_SUBNET_AWARE_ROUTING`/`..._PREFIX_LEN`; does NOT implement the four-PF
    contract's `NCCL_SWITCHLESS_RING_ONLY`, `NCCL_IB_EXTENDED_IPV4_GIDS`,
    `NCCL_IB_PRESERVE_PCI_DOMAIN`. With this lib, the launcher's `NCCL_SWITCHLESS_RING_ONLY=1`
    is a no-op and NCCL init would attempt ring-impossible tree connects (the FP8 launcher
    always sets `NCCL_SKIP_TREE_CONNECT=1`). `transport.sh` rejects `NCCL_*` keys in
    `EXTRA_ENV`, so the only non-image-level path was the **documented site mod** to
    `upstream/scripts/transport.sh` (one line, switchless branch; recorded in
    `upstream/VENDORED-AT.md` "Site modifications" with re-apply instructions).
  - **Staged transport (owner-approved)**: first boot runs the FP8-proven 2-PF-per-node
    fabric (`IB_HCA=rocep1s0f0,rocep1s0f1`, second PCIe views stay unaddressed);
    the four-PF upgrade is Phase E. Plan for Phase E: the othexmr four-PF build
    (`feat/four-pf-source-integration` @ `94c2669`) hard-codes the listener IPv4-GID bytes
    `10.100.224–227` in `patches/nccl-2.30.7-dual-pf.patch` (`tp4ValidateListener`) — our
    `10.10.1–4.0/24` needs that one line changed + rebuild + validation; netplan then adds
    the second PCIe-root address per node per cable **in the existing 4 cable subnets**
    (the contract is 2 cable subnets/node × 2 PCI roots, NOT 8 new /24s — the validator
    caps at 2 subnets per node and requires one address per PCI root within each).
  - **Filename gotcha fixed at bring-up**: `start.sh`'s check + `LD_PRELOAD` expect
    `libnccl.so.2.30.7`; the site dir ships `libnccl.so.2` → in-dir symlink on all 4 nodes.
  - **Lane coexistence (owner requirement: switch back and forth with FP8)**: no renames
    needed — distinct container/CTN names, distinct `OVERLAY_REMOTE` roots, NCCL dir
    shared without conflict (`libnccl.so.2` vs symlink), proxy `BACKEND_MODEL`
    (`glm-5.3-flash`) matches both lanes. Switch procedure documented in README.
  - Weights verified present on all 4 nodes (nvidia `09b04e5e` + drafter `bf582e4e`,
    identical inventory); nvidia snapshot ships `chat_template.jinja` → fallback path
    never triggers.
- **2026-09-29** — Recipe created: vendored knapcio `770d115`, site env + zero-mod
  profile + weight materialization script; decisions 1–5 recorded above. Uncommitted,
  awaiting owner review.
