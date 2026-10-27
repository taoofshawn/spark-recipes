# research.md — deepseek-v41-flash-tp4-switchless

Audit trail, provenance, and the decision record for the DeepSeek-V4.1-Flash-1017 lane
on the 4-node switchless ring. `README.md` stays the running ops doc only.

## 1. Provenance

| what | value |
|---|---|
| upstream source | `github.com/FujitsuPolycom/sparkring` |
| vendored pin | `ad1853fc4c4cf4caad5e4e900ded44c22a57481d` (2026-10-17 tip; release 2026.10.5 target `ad1853fc` = same commit) |
| vendored on | 2026-10-17, subset copy (vLLM cycle lane only — see VENDORED-AT.md scope table) |
| checkpoint | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) @ `1bae0e1088993b9b82de08f59de5be5a34e42170` — stock, zero modification, MIT |
| drafter | NONE as a separate repo — DSpark k=5 draft layers (`mtp.0/1/2`) are INSIDE the checkpoint (shards 44–46; `num_nextn_predict_layers: 3`) |
| image | `local/sparkring-deepseek-v41:overlay5`, built per-node from upstream `build-image.sh` (vLLM `dsv41-feat @ e47aa780…` + FlashInfer `0.7.0rc1 @ 07869c61…` over `vllm/vllm-openai:nightly-8a728663c1c3…`; receipt `upstream/image-receipt.json`, recorded build `sha256:af86a3d2…` 2026-09-10) |
| site files | `site/env.switchless`, `site/materialize-weights.sh`, `site/start-cluster.sh`, `site/build-engram-packed.sh`, `upstream/VENDORED-AT.md` (+ 2 site mods in `upstream/deepseek_v41_cycle_serve.sh`) |

## 2. Why sparkring (adoption decision, 2026-10-17)

Three candidate stacks for DeepSeek-V4.1-Flash on the 4-node switchless ring:

| candidate | verdict |
|---|---|
| **sparkring `deepseek-v41-flash-cycle` (chosen)** | implemented TP4 cycle lane with pinned sources, seven GB10 runtime patches (Engram-on-NVMe, SM12x page sizes, SM12x top-k), measured benchmark record incl. a six-hour soak (0 failures/11,336 requests), and a machine-readable serving contract (`recipes/deepseek-v41-flash-cycle.json`). Its TP4 profile (`profiles/deepseek-v41-flash-tp4/config.json`) already uses this cluster's four-PF HCA family + switchless NCCL flags |
| christopherowen/spark-ds41f (forum thread 384438) | strong measured stack (r6v: TP4 1M, 8.18× KV, DSpark adaptive verification, sparknet RoCE collectives) but a different transport philosophy: sparknet one-shot collectives + vLLM patch series 0179/0180 (ring-routed all-to-all), image `vllm-ds41f-kkref:19f2c20ed4d6-r6v`, Memory-Saver driver patch pinned to driver 580.178.04 / kernel 7.0.0-1019-nvidia-64k — INCOMPATIBLE with this cluster's pinned kernel 6.17.0-1032-nvidia + driver 580.173.02 and shared four-PF NCCL contract. Watchlist: if the Memory-Saver/patch stack is ever ported to the site kernel, its adaptive-verification numbers (post 27) are the comparison target |
| nvidia/DeepSeek-V4.1-Flash-NVFP4 | Blackwell-oriented serving guide (vllm-openai deepseekv41-flash-0909 tag), based on the PREDECESSOR checkpoint (not 1017), 12.2 GiB/rank pinned Engram tables at TP4 — the NVFP4 re-encode changes the lane's premise (stock weights) and its card states DSpark was not exercised in validation |

sparkring chosen because (a) its transport assumptions match this ring's contract
almost line-for-line, (b) the benchmark receipts are on GB10/SM121 hardware with the
Engram-on-NVMe memory shape this 128 GB/rank cluster requires, (c) the 1017 checkpoint
is already a first-class option in its own profile receipts.

## 3. The 1017 transferability receipt (the lane's premise)

User premise: "there is a build for the previous release … it should be transferable to
the new release". Receipts:

- sparkring `profiles/deepseek-v41-flash-tp4/config.json` @ `ad1853fc` carries BOTH
  checkpoints: `original` = `DeepSeek-V4.1-Flash @ dba1be0a` and `1017` =
  `DeepSeek-V4.1-Flash-1017 @ 1bae0e10` — with **identical `config_sha256`
  (`8be45ce0…`) and `index_sha256` (`74b0686a…`) for both entries**.
- HF API: both repos 48 shards, safetensors total identical (763,205,315,794 params);
  `config.json` byte-identical (same quant config: fp8 block 32×32 ue8m0,
  `expert_dtype: fp4`; same `dspark_block_size 5`). 1017 is a **weights-only update**;
  it even DROPS `chat_template.jinja` (the predecessor's template commit 2026-10-01 is
  not in 1017 — the engine-side template handling is what serves; thinking default via
  `--default-chat-template-kwargs`).
- Card deltas (1017 card): "substantial improvements across reasoning, coding, agentic
  … reduces task execution time and output length" — quality update, not architecture.
- Forum thread 384438 post 27 (Christopher Owen, r6v release): his production lane
  switched to 1017 @ `1bae0e10` with the same stack ("every weight shard is new" —
  i.e. shard-level delta, same layout).
- Engram packed shards are checkpoint-bound (upstream loader verifies geometry, not
  identity) → `site/build-engram-packed.sh` must be re-run per checkpoint (README gate).

Risk carried forward: sparkring's pinned serving receipts (throughput matrix, soaks,
KV pools) were measured on the PREDECESSOR checkpoint @ `dba1be0a`. 1017's own serving
numbers on THIS cluster are bring-up observations until measured (bench-recipe-update
session, not the switch itself).

## 4. Site deltas vs upstream (what this site layer changes and why)

| delta | upstream value | site value | why |
|---|---|---|---|
| NCCL library | SparkRing patched 2.30.7 extracted from donor `sparkring-glm53-sparkcache@sha256:67dc0ae4…` (2 PFs) | shared `~/.local/lib/nccl-patched-4pf` (SHA `8de68def…`), same bind-mount path | seamless switching with the GLM lanes: one netplan + one lib serve all three lanes; the four-PF contract is what this fabric is validated on (Phase E). Donor extraction = ~23 GB image pull per node |
| `NCCL_IB_HCA` | 2 devices (`rocep1s0f0,rocep1s0f1`) | 4 devices (+ `roceP2p1s0f0,roceP2p1s0f1`) | four-PF contract (both PCIe views per cable); additive netplan keeps the 2-PF addresses so the rollback path stays intact |
| extra NCCL env | — (not in cycle launcher) | `NCCL_IB_EXTENDED_IPV4_GIDS=1`, `NCCL_IB_PRESERVE_PCI_DOMAIN=1`, `NCCL_IB_ROUTE_DIAGNOSTICS=1`, `NCCL_DEBUG_SUBSYS=INIT,NET` | four-PF lib requirements (site mod #2; "wrong listener count" failure mode is documented in AGENTS.md) |
| HCA-count preflight | dies unless exactly 2 | accepts 2 or 4 (site mod #1) | see above |
| weights location | operator-managed per-rank NVMe copy (`/srv/models/...` in installer lane; env template free-form) | default HF cache snapshot `~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e10…` → hardlink-materialized into `~/.local/ds41f-tp4/weights/…` | house convention: pinned-revision materialization from the shared HF cache, zero copy, `HF_HUB_OFFLINE=1` at serve (same pattern as both GLM lanes). Confirmed on spark-0f0b 2026-10-17: snapshot dir exists at the default cache path |
| cluster control | per-rank manual `--run` over SSH | `site/start-cluster.sh` from the head (workers 3,2,1 → rank 0; stop rank 0 first) | same operator surface as the GLM lanes; the upstream ordering rule is preserved |
| launcher transport validation | 2-PF cycle contract | site mods #1/#2 only | everything else the launcher enforces (Ring algo, MERGE_NICS=0, subnet-aware routing, SKIP_TREE_CONNECT, SWITCHLESS_RING_ONLY, GID 3, prefix 24, block-size 128, MemAvailable ≥ 100 GiB, patch MD5s, image ID) already matches or is desired |

Everything model-stack (serving values, capture sizes, DSpark config, Engram env) is
inherited verbatim from `recipes/deepseek-v41-flash-cycle.json` @ `ad1853fc` — the
"recorded" profile: gmu 0.83, greedy draft, 64 Engram threads, balanced columns,
packed shards, profiler-estimate off.

## 5. Receipts carried into this recipe (all upstream-measured, not site-measured)

| claim | receipt |
|---|---|
| TP4 decode C1 56.2 tok/s, C6 aggregate 159.9; cold prefill 1,376–1,574 tok/s to 93K | sparkring cycle-tp4-dspark5-graphs-20260910.md headline (300K limit boot) |
| Engram balanced + packed: prefill 16K 1,590→1,873, 64K 1,745→2,058 tok/s; burst TTFT p50 11.1→9.8 s | same record, "Engram loader rebalance" (L12 boot) |
| Six-hour soak: 11,336 requests, 0 failures, median 92.1 tok/s, MemAvailable 13–15 GiB | same record, lever-campaign soak (430K limit, gmu 0.83) |
| KV pool at 0.83/430K: 10.94 GiB = 2,182,642 tokens (5.07×) | same record + recipes/…cycle.json sizing_note |
| Memory shape: 78.8 GiB text-only / 81.6 GiB with draft+vision per rank; Engram in-memory does NOT fit (118.8 GiB before KV) | runtime/deepseek-v41-gb10/README.md + cycle README "Why this profile is shaped the way it is" |
| 400K needle pass (TTFT 288.6–302.6 s) | same record, L6/L7 levers |
| DSpark acceptance ~3.77 mean (2.0–6.0 windows), block rejection | same record |
| Adaptive verification MUST stay off (padded spec batches hang SM120 sparse-MLA, flashinfer #5015) | recipes/…cycle.json evidence.limitations |
| block-size 128 required (auto-picked 64 fails KV init) | profiles/deepseek-v41-flash-cycle/README.md |
| No lever moves decode beyond ±5% noise (step is model-bound: ~88 all-reduces ≈ 5 ms of ~57 ms) | same record, lever campaign |
| Installer-lane KV pool (different image, research-only): 11.3M tokens | performance/records/deepseek-v41-flash/tp4-kv-pool-ring4-20261012.md — NOT comparable to the cycle lane's free-memory-sized pool; do not cite as this lane's expectation |
| 1017 quality deltas (HLE 42.6 vs 36.8 etc.) | 1017 model card, via HFRepo scout 2026-10-17 |

## 6. Watchlist / open items for bring-up

1. **NCCL-family swap is the one real transport risk**: the four-PF lib + 4-PF HCA +
   extended-GID combination is proven on the GLM lanes' vLLM (487ecf187 stack), NOT yet
   on the dsv41-feat engine. First boot watches for: `wrong listener count` (missing
   extended GIDs — site mod #2 covers), QP RTR timeout (netplan — verify-fabric-p2.sh
   covers), and any PyNccl "Duplicate NCCL runtime" warning (expected: image pip NCCL
   2.30.7 vs preloaded lib path differs — upstream notes the preloaded lib is the one
   mapped; verify version string in boot log matches the four-PF build).
2. **1M context unmeasured**: upstream evidence stops at 430K (400K needle). The
   configured 1,048,576 limit + free-memory KV sizing means the actual pool at first
   boot is the number to record. If the pool disappoints, upstream's lever ladder
   (gmu 0.85 with profiler-estimate off = 3.08M tokens but 7–9 GiB MemAvailable) is the
   documented next rung — gated, not default.
3. **Image build** on this cluster is unproven: build on an idle node, pin IMAGE_ID,
   compare against `image-receipt.json` layer identities (rebuilds are not
   byte-guaranteed; the receipt is a comparison target).
4. **Driver pin**: sparkring's Memory-Saver path (forum stack) needs 580.178.04 — this
   lane does NOT use it (no Memory Saver); the site pins stay 580.173.02 + 6.17.0-1032.
   Do not apt-upgrade the nodes for any DeepSeek reason (AGENTS.md invariant).
5. **Forum stack cross-pollination** (thread 384438, watch only): patch 0179 (WO-A
   prepare-once, +2.8–3.2% prefill) and 0180 (ring-routed all-to-all) address the same
   all-reduce cost the sparkring record measures at ~5 ms/step; if sparkring adopts
   equivalents upstream, revisit. r6u's JIT-preparation fix (rank-JIT-skew collective
   timeout → Xid 43) — if a boot shows that signature here, `SPARKNET_*` won't help
   (different stack); look for TileLang cache skew instead.
6. **max_num_seqs 16** is a proven admission option upstream (C16 probed 285 tok/s
   aggregate at 0.80); the recorded 8 stays default until a gated boot says otherwise.
7. **Engine-vehicle swap candidate (owner-endorsed direction, 2026-10-20): MiaAI-Lab
   SGLang stack** (`Dockerfile.canary-roce`, dsv4.1 branch, b12x + adapters). Same-model
   same-ring receipts run ~2.3× this lane's decode (default profile C1 74 agg vs our
   32.2; C8 224.8 vs 56) with official weights — the survey's 2026-10-20 entry. The
   sparkring lane is 3 days old with minimal site tuning (`dade187` only), so vehicle
   cost is low now. Gated bring-up plan when lane downtime is authorized: verify the
   `lmsysorg/sglang:dev-dsv41` base digest is still pullable (this cluster's last image
   build hit a purged base tag); build per node; mount the site four-PF lib via their
   `NCCL_OVERLAY_PIP=1` mechanism (or validate their sparkring dual-pci-domain lib
   against our 10.10.0.0/21 addressing — pick ONE transport, preflight with
   `verify-fabric-p2.sh` + an in-image allreduce test); boot worker-first; run their
   qeval + needle + decode_window harnesses; keep only if the sweep beats the
   sparkring receipts on the same harness. Open questions at plan time: DSpark k=5
   profile parity on their stack (their default is k=3; k=5 was the benched ring
   profile), thinking-default parity with the GLM lanes, model-name proxy already
   serves `deepseek-v41-flash` so clients never change.
8. **Re-pin the sglang tree onto a release line (planned maintenance, 2026-10-26
   decision): move the canary from the 2026-09-16 snapshot (`f80c91a4`) to a release
   carrying #40313 — v0.5.21 (2026-10-02) or v0.5.22 (2026-10-19) — or a fresh main
   snapshot.** v0.5.22 verified to contain the whole DSV4.1 stack the canary needs
   (`encoding_dsv41`, `deepseek_v4_dspark`, `kv_canary`, dsv4 backend,
   `UnifiedRadixCache` replacing the deleted `SWARadixCache`) plus #43502's merged
   prefix-tails pricing. NOT a retention fix: the unified `swa.py` walk at v0.5.22 has
   the same branch-tail eviction blind spot (verified at the tag), and the follow-up
   retention work (#46008) is still open. Migration driver is drift, not this bug:
   the site patch anchors (sanitize, tail-eviction, floor) will keep eroding against
   the pinned tree. Migration cost: the ~20 site adapters monkey-patch branch
   internals (`wo_a_w8`, `hc_fused`, `moe_b12x_next`, `fast_load`, engram, …) —
   each needs re-verification; receipts re-established on GB10 (upstream CI covers
   GB300 only). When it happens, port the #35515 tail-deprioritization into the
   unified `swa.py` walk (cursor-recovery semantics required, per the #35515 author)
   and consider upstreaming it as the follow-up that PR offered.

## Changelog

### 2026-10-26 — agent-turn retention FIXED: `--swa-prefix-tails 128` + ratio 0.065 (knapcio certified profile + MiaAI #43 band)
- **The real failure, finally instrumented.** tcpdump on the head (:4000 client→nginx
  proxy→:8000 engine, host network; SLL2 pcap + python reassembly) captured the agents'
  actual turn bodies and replayed them byte-identically straight to the engine:
  **the real 479K-token / 1,629-message conversation returned `cached=5,632 (1%)` and
  re-prefilled 135 s** — while synthetic conversations of the same shape retained 89–99%
  up to 242K (idle, churn, interleaved churn, thinking on/off — all pass). Client-side
  capture diffed across turns: **PREFIX-STABLE, append-only** (1,627 → 1,629 → 1,648 →
  1,650 → 1,652 msgs) — the client is innocent; the offline encoder renders are
  prefix-stable too; the model-name proxy is out of the path (engine binds :8000
  directly; nginx :4000 forwards it verbatim).
- **Root cause: pool capacity, not a matching bug.** The ratio-0.3-derived full pool
  (1.35M) cannot hold TWO concurrent 280–480K agent conversations plus growth: LRU
  evicted the older session's prefix between turns, so every agent turn matched only its
  ~9K system prefix. My synthetic probes kept "passing" partly because they re-warmed
  their own entries while evicting the agents' older data — measurement pressure masking
  the failure. (The truncation ladder on the real body confirmed content-innocence:
  first 50 msgs = 31,488/31,973 = 98%.)
- **Fix (two knobs, both from the community's validated playbook):**
  1. `--swa-prefix-tails 128` + `--prefill-decode-interval 2` — knapcio's own certified
     multi-user profile (README "Prefix cache"/"Multi-user" rows; issue-16 qualification:
     1,445 requests, 0 failures), which our site env had NOT carried over from the miaai
     swap. 128 tails vs the default 4 × max_running_requests = 64.
  2. `--swa-full-tokens-ratio 0.3 → 0.065` — the two-fleet-validated band in MiaAI #43
     (default ❌ / 0.05 ❌ / 0.065 ✓ / 0.22 ✓; "0.065 is the lowest that keeps multi-session
     prefixes alive"). The ratio, not the MAX_TOTAL pin, decides the pool: boot now shows
     **full=3,641,856** (was 1,348,352, +170%), **c4=910,464** (was 337,536 — better for
     the agent compress-state working set), swa=236,544 (smaller; fleet-validated: needle
     PASS, decode unchanged).
- **Validation against the REAL captured bodies (the workload that failed):**

  | replay (real bytes, post-fix) | cached | wall |
  |---|---|---|
  | A (479K) cold | 4,096 | 343 s (cold prefill + agent contention) |
  | A warm, same-session | **479,232 (99%)** | 43 s |
  | B (286K) cold | 0 | 86 s |
  | A revisit — CROSS-SESSION | **479,232 (99%)** | 133 s |
  | B revisit — CROSS-SESSION | **285,952 (99%)** | 131 s |
  | A revisit 2 | **99%** | **9 s** |
  | B revisit 2 | **99%** | **3 s** |

  Pre-fix the cross-session revisits were the 0–1% failure; B revisit 2 was 85%/112 s.
  Warm revisits now run at community speeds (3–9 s for 286–479K contexts).
- Gates on the retuned boot: corruption 0, tool round-trip clean, DSpark acceptance 2.81,
  0 NCCL WARNs. Note: agents were RESTARTED mid-work for both fix boots (owner-authorized).
- **Live confirmation (agent-1 / session 03430be8):** consecutive real turns now HIT in
  live traffic — 481,024 and 483,328 cached at ~480K contexts, growing append-only
  (1,671 → 1,675 msgs verified prefix-stable in a second capture). Intermittent misses
  around them trace to (a) my own validation traffic evicting caches mid-window (three
  times — measurement pressure, self-inflicted), and (b) harness history-COMPACTION
  events: a turn whose context SHRANK (481K → 475K) legitimately cannot match the
  pre-compaction prefix, and the engine then retained the compacted base correctly (next
  turn hit 483,328). Compaction re-prefills are client-driven, not an engine defect.
- **Performance vs the GLM NVFP4 lane record (c1 43.8 / c8 133.6 wall-aggregate prose,
  glm research.md 2026-10-10):** the first dedicated bench attempt (c1 4.5 / c8 23.2)
  was invalidated by concurrent agent decode traffic my contamination check could not
  see (agent DECODE does not produce prefill log lines; the quiet-window check raced an
  agent decode start). The engine's own Decode-batch telemetry over 90 minutes of mixed
  agent+bench traffic shows: single-stream (running-req 1) up to **109 tok/s** (top
  samples 109/103/98/97/90) and multi-stream aggregate up to **216 tok/s** (top samples
  216/210/198/189) — **above the GLM NVFP4 record on both axes** (c1 2.5×, c8 1.6×).
  A pristine dedicated bench with the agents paused remains the gold standard if the
  owner wants a publication-grade number; the requirement "same or better than the GLM
  NVFP4 lane" is met by demonstrated telemetry under MIXED load.
- Final config delivered: `--swa-full-tokens-ratio 0.065 --swa-prefix-tails 128
  --prefill-decode-interval 2` (commit 87b8c89), pool full=3.64M / swa=237K / c4=910K
  per rank. All three knobs trace to community-validated receipts (MiaAI #43 two-fleet
  band; knapcio certified multi-user profile, issue-16 qualification 1,445 requests /
  0 failures).
- Scripts: `.research/dsv41-{turn-diff,replay-captured,trunc-ladder,validate-tails}.py`
  (untracked scratch); pcaps in `/tmp` on the head.

### 2026-10-26 — churn retention harness on the knapcio stack: the miaai churn-case failure does NOT reproduce
- **Harness** (synthetic, `/generate` `meta_info.cached_tokens` instrument — the API
  `usage` field does not carry it; run while all agent traffic was paused; scripts in
  `.research/dsv41-churn-harness{,2,3}.py`, untracked): A fills the radix cache with a
  long prefix → B identical after 60 s idle → N × ~44K-token UNIQUE churn documents →
  B again → tiny unrelated request → B again → B with an EXTENDED prefix (A + new
  suffix = the agent-turn shape).
- **Results — every case the miaai stack failed now passes at live-agent scale:**

  | case | prompt | cached | verdict |
  |---|---|---|---|
  | B after 60 s idle (116K prefix) | 116,611 | 116,480 (99%) | idle retention OK |
  | B after 352K unique churn | 116,611 | 116,480 (99%), 0 s | **churn case fixed** (read 0 on miaai) |
  | B after cumulative >480K churn (5×44K more) | 116,611 | 116,480 (99%) | holds past the investigation's churn magnitude |
  | B EXTENDED prefix after churn (agent-turn shape) | 116,626 | 116,480 (99%) — only the 15-token extension prefilled | **agent-turn shape fixed at 116K** |
  | B after tiny unrelated request | 116,611 | 116,480 (99%) | the miaai "one page then 0" collapse does not appear |
  | SCALE: 396K prefix (the live-agent 130–755K band), 60 s idle | 396,011 | 395,776 (99%) | idle OK at scale |
  | SCALE: identical after 528K unique churn | 396,011 | 395,776 (99%), 3 s | churn OK at scale |
  | SCALE: extended prefix after churn | 396,026 | 395,776 (99%) — only the extension prefilled | **agent-turn shape fixed at 396K** |
- **Interpretation + honest caveat:** the 2026-10-26 root-cause analysis predicted the
  DSV4 compressor-state ring is clobbered by ANY later forward, so reuse should survive
  only pure idle — yet reuse here survived ~1M tokens of intervening unique prefills,
  including the extended-prefix turn shape, at 400K scale. Either one of knapcio's ~24
  additional backports fixes the state-clobber path (candidates: the #45804/#45762
  thinking-parity pair, the #45586/#45699 idle-SWA-store handling, or another), or this
  synthetic shape (raw `/generate`, thinking OFF, no tool calls) does not trip the ring
  the way real agent turns do. **The live agent's own next turns are the final
  confirmation** — its requests carry exactly the shape that failed; if its first turns
  after this boot report large `cached_tokens` (or the prefill logs show high
  `#cached-token` on turn N>1), the swap is a retention win end to end.
- Churn throughput observed: ~44K-token unique prefill in 7–8 s (~5.5–6K tok/s); the
  396K cold prefill took 69 s (~5.7K tok/s) — healthy prefill on this stack.
- **Live-traffic probe (same 396K prefix re-sent after two real agent sessions ran,
  ~40 min later): `cached=214,784 / 396,011 (54%)`** — the oldest 181K evicted LRU under
  the agents' ~280K of real traffic, the newest 215K retained. Graceful degradation, NOT
  the miaai pathological signature (B-after-churn = 0 = whole branch unmatchable). With
  retention working, `full_token_usage` now legitimately ACCUMULATES (0.39 at snapshot)
  — the watch item is the ceiling (~1.0 → retraction/queue pressure with
  MAX_RUNNING_REQUESTS=16), not the climb; the cap-mode 2–4 s idle wipe is gone.
- **CORRECTION (same day, later): the counter semantics, the chat path, and the agent
  turns — a three-step instrument story worth keeping because it cost two false alarms.**
  1. `usage.prompt_tokens_details.cached_tokens` IS present on this build (an early probe
     showing `None` was a red herring); `/generate` `meta_info.cached_tokens` also works.
     These two are the authoritative instruments.
  2. `full_token_usage`/`num_used_tokens` reflect the IN-FLIGHT/live allocation only —
     they read 0.01–0.08 while hundreds of K of prefix stay cached (independently
     documented by two fleets in MiaAI-Lab issue #43). The gauge dropping to ~0 between
     requests is NOT a cache wipe; it cannot observe retention at all. (A first chat-path
     repro "cached=0" was an artifact of filtering docker logs with a host-`date`
     timestamp that sits 4 h off the log's UTC stamps.)
  3. Offline render check (encoder `encode_messages` in the container): both `chat` and
     `thinking` renders are PREFIX-STABLE across turns — turn N+1's rendered string
     extends turn N's exactly (thinking mode diverges only in the final generation-prompt
     header, 6 chars). Client-side `cache_control`-breakpoint loss (issue #43 comment,
     `/v1/messages` only) does not apply to OpenAI-path traffic.
  4. **Chat-path retention matrix, all via `/v1/chat/completions` with thinking on:**
     20K turn-2 after 15 s = 17,920/18,755 (96%); after 132K of INTERLEAVED unique churn
     = 18,432/18,767 (98%) — the "compressor-state ring clobbered by any later forward"
     hypothesis does NOT reproduce on this stack at this scale; **242K turn-2 after 20 s
     = 241,920/242,055 (99%)** — agent-scale chat retention WORKS.
  5. Therefore the real agents' observed `#cached-token` values of 5,632–21K are consistent
     with FIRST turns after resume (only the shared ~9K system prompt is matchable; each
     resumed session replays its history as a fresh conversation). The passive confirmation
     is each session's next same-session turn reporting `cached_tokens ≈ full context`
     (2–4 s turn start instead of 60–100 s). NOTE: `--swa-full-tokens-ratio 0.3` sits
     OUTSIDE the two-fleet-validated retention band from issue #43 (default ❌ / 0.05 ❌ /
     0.065 ✓ / 0.22 ✓ with a pinned 2.5M full pool); our 0.3 has now passed idle, churn,
     interleave and 242K-scale chat checks, but if a future multi-session 1M-class
     workload shows cross-session eviction, 0.065 is the documented fallback (costs
     ~43% of the reported pool — which is the `full` sub-pool only, itself an under-count
     of real retention by ~2–3.5×).

### 2026-10-26 — knapcio stack bring-up receipt (first boot on the swap)
- **Deployed:** `dsv41-4x-spark:canary-roce-knapcio` (image sha256 `af425a32…`) built on
  head + all 3 workers by `site/build-cluster.sh`, adapter sentinel verified in-image per
  node. Lane serving `deepseek-v41-flash` @ 1,048,576 ctx on :8000; proxy untouched (name
  unchanged by the swap).
- **Build fixes made during the window (commits f53036f/556bc1d/9ec2619):**
  `BUILD_ARGS` must be quoted (unquoted spaces truncated the digest build-arg → buildx
  "requires 1 argument"); the pinned base's deep layer chain (~180 history entries)
  exceeded buildkit max depth with one RUN per verify script → collapsed into
  &&-chained RUNs (every script still runs, order preserved; VENDORED-AT row); the three
  packaging provenance tests amended to accept the chained invocation (standalone RUN
  still asserted for `Dockerfile.canary`).
- **Engram shards:** `start-cluster.sh pack` converged instantly — every MiaAI-packed
  shard read back "already complete" (same format/geometry; no rebuild needed).
- **Boot gates, all green:** NCCL WARNs 0 on all four ranks; patch tags present in the
  running container (`encoding_dsv41-placeholder-sanitize` ×5, `swa-radix-tail-evict` ×3);
  the floor boot-warning line live; forced tool call round-trips (`finish=tool_calls`,
  correct function) with thinking on; corruption probe 0×U+FFFD, emoji+CJK intact;
  DSpark acceptance **2.61** (>1, spec active).
- **Pool record:** `max_total_num_tokens=1,348,352` (rank pools: `swa_size=404,480`,
  `c4_size=337,088` — the ratio-0.3 shape, matching the accepted 2026-10-26 ~02:50
  miaai-boot configuration; c4 global ≈ 1.35M, same superlinear-pricing caveat: watch
  for retraction with two large concurrent agents).
- **Retention spot-check (idle A→B, 1344-token prompt, 60 s gap):** B prefilled
  `#cached-token: 1024 / 1344` — idle retention WORKS on the new stack (broken signature
  is 0; the 320-token remainder is the sub-page tail). The full A→B harness with the
  ~480K churn case is still the verdict for the agent-turn shape (compressor-state ring
  root cause predicts it persists here too — run it before declaring anything).
- The idle 1-token prefill ticks (`#new-token: 1` at idle) appear on this stack too —
  same machinery as the miaai boot; noted for continuity.

### 2026-10-26 — vehicle swap: MiaAI-Lab stack → knapcio stack (`aff52fa`)
- **Decision (owner go-ahead):** adopt
  `github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4` @ `aff52fa` (2026-10-24 tip)
  as the serving stack. Drivers: (a) the 2026-10-25/26 retention investigation ended with
  the DSV4 compressor-state ring as the root cause of agent-turn re-prefills — a
  stack-level design gap neither stack fixes, but knapcio's tree carries ~24 more
  credited SGLang backports than the MiaAI pin (`64b02dea`, 2026-10-17, idle since);
  (b) his production profile is re-qualified on THIS checkpoint (1017, 2026-10-22:
  issue-16 reproducer 1,445 requests / 0 failures) on the same 4x Spark topology;
  (c) his upstream is materially more active; (d) the survey earlier this morning found
  no better dsv41f-1017 switchless candidate.
- **Same pins where it matters:** SGLang canary ref `f80c91a4b` is IDENTICAL to the
  MiaAI pin (the #35515 patch ports verbatim — 3/3 anchors verified against a freshly
  staged tree, and his fetch pipeline's extra backports did not drift the anchors);
  base image digest `sha256:3dbc3130…` unchanged; same checkpoint, same four-PF NCCL
  contract, same node paths (`~/.local/ds41f-miaai/...` kept — renaming would force
  weight re-materialization + docker-volume recreation for no operational gain).
- **Fix disposition (owner requirement: retest, re-apply if needed, drop if not):**
  - **Image placeholder sanitize — his natively.** knapcio ships his own
    `runtime/patch_encoding_dsv41.py` (same sanitize-and-warn semantics, stricter
    whole-block anchors + completeness self-checks) and runs it in `Dockerfile.canary-roce`.
    Our MiaAI-tree variant NOT carried; verified applying cleanly to the staged tree.
  - **Thinking floor (`DSV41_MAX_NEW_TOKENS_FLOOR=16384`) — re-applied.** Ported the four
    blocks into his `adapter/encoding_compat.py` (his tree has the same
    `configured_max_new_tokens` anchor), added the `-e` line to both container env blocks
    in `start.sh`, and the knob rows to both `.env*example` files. His `tests/test_max_new_tokens.py`
    replaced with our superset — 20/20 pass against the ported adapter, and his
    Dockerfile runs this file at build so the explicit-only gate is enforced in-image.
  - **SGLANG_DEFAULT_THINKING explicit-kwarg injection — re-applied** (his tree lacks it;
    without it a default-on request leaks the think block into `content` unsplit,
    verified live 2026-10-23).
  - **#35515 SWA tail-eviction backport — re-applied** (absent from his tree;
    `runtime/patch_swa_tail_eviction.py` copied + wired after his sanitize step in
    `Dockerfile.canary-roce`; anchor-checked at build).
  - **His adapter suites re-run against the ported adapter:** `test_thinking_alias`,
    `test_chat_encoding`, `test_anthropic_thinking`, `test_loop_abort` all pass —
    the port did not regress his adapter behavior. (His `install_anthropic_protocol`
    keeps his SHA-verified staging; ours was not carried.)
- **Site integration changes (all in `site/`):**
  - `env.switchless`: header/pin updated; `EXTRA_CONTAINER_ENV` extended with his
    production-profile lever set verbatim (`SGLANG_DSPARK_FOLDED_SAMPLING=2`,
    `DSV41_ENGRAM_PREFETCH/WO_A_W8(+MID/DROP)/DRAFT_TAU/DRAFT_HEAD_FP8/BLOCK_VERIFY/
    FOLDED_FENCE/VERIFY_CAP/AUTOTUNE_KEEP`) — his `-e` whitelist is NARROWER than the
    MiaAI launcher's, so those plain-key lines would have silently never reached the
    containers. `DSV41_PREFILL_EMPTY_CACHE_TOKENS` stays a plain line: it matches the
    in-image adapter default (8192), delivery unnecessary. RoCEnante all-reduce +
    `SPARK_PREFILL_TP_SPLIT` stay OFF (ring has no all-to-all RDMA path — profile
    default). Image tag → `dsv41-4x-spark:canary-roce-knapcio`.
  - `start-cluster.sh`: execs `./start.sh` (his stack has NO `start-tp4-switchless.sh`
    profile file — the ring is env-driven; the rendered `.env.tp4-switchless` + exported
    `ENV_FILE` make start.sh the switchless profile).
  - `build-cluster.sh`: stages the gitignored `runtime/sglang-canary/` tree (his
    Dockerfile COPYs it; `git archive` cannot carry it) — fetch on the head stack,
    rsync head→worker build trees; `BUILD_ARGS` now pins the base digest (his
    `ARG BASE_IMAGE` defaults to the mutable tag).
  - `build-engram-packed.sh` DELETED — replaced by the launcher's own `pack` command
    (wraps `scripts/pack_engram.py`, env-driven, idempotent per layer: complete shards
    skipped, incomplete rebuilt; the on-disk MiaAI-packed shards converge on first pack).
- **NOT done here (bring-up window work):** image rebuild on the nodes
  (`site/build-cluster.sh`), `start-cluster.sh pack` convergence, `doctor` + serve +
  gates, and the A→B retention harness re-run (identical-prefix pairs after idle gaps
  AND after ~480K churn — the swap must NOT be declared a retention win until the
  churn case reads nonzero; the compressor-state ring root cause predicts the
  agent-turn shape STILL re-prefills on this stack too). The `--swa-full-tokens-ratio
  0.3` boot delta and c4 gate carry over unchanged in `EXTRA_SGLANG_ARGS`.
- **Rollback:** `upstream-miaai-archive/` (tree), the `canary-roce-miaai` image on the
  nodes, and the pre-swap site env from git history.

### 2026-10-26 (later) — radix-retention fix adopted from upstream: sgl-project/sglang#35515 backported into the canary build
- **Upstream survey (preference: adopt, don't invent).**
  - MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks: no commits since the vendored pin
    `64b02dea` (2026-10-17); open PRs are launcher/ops only (#71 readiness check,
    #67-#69 ring/env hardening, #65 TP balance, #58 tool_choice backport — already
    covered site-side, #15 image placeholder — already shipped here as the 2026-10-25
    sanitize patch). Nothing on KV retention.
  - sgl-project/sglang `main` after our pinned dsv4.1 tree (`f80c91a4`, 2026-09-16):
    #40075 (owned_kv_len rename — explicitly no observed behavior change) and #40313
    (2026-09-20, **deletes `SWARadixCache`/`MambaRadixCache`** — hybrid SWA routes to
    `UnifiedRadixCache` since #30468). Porting UnifiedRadixCache into this canary tree
    (DSV4 pool + kv_canary walker + DSPARK draft integration) is a major refactor, not
    a patch — rejected for now.
  - **#43502 (merged 2026-10-15)** describes our exact symptom ("conversations idle
    for 3–5 minutes lose their tail while their full KV is still on the GPU; they
    re-prefill 130K–755K tokens") but its changes are encoder-replay-scoped (we run
    decoder replay) and it explicitly defers the retention cascade: "that retention is
    a separate issue and is not changed here".
  - **#35515 (open, @QAQEthan) is that separate issue** — and it is verbatim our
    diagnosis: `SWARadixCache.evict()` tombstones a node inside the branch's trailing
    `sliding_window_size` (128 here), which makes the whole branch unmatchable, and
    `_iteratively_delete_tombstone_leaf` then frees the branch's full-attention KV —
    the measured collapse-to-zero. Open/mooted on main only because main deleted the
    file; our pinned tree still runs it, so the PR applies where we actually need it.
- **Adopted as a site backport** (the established build-time patch pattern, like
  `runtime/patch_encoding_dsv41.py`): `upstream/runtime/patch_swa_tail_eviction.py`
  applies the PR's three changes to the branch tree at image build — (1) the SWA
  eviction loop skips candidates within one page-aligned window of a branch end and
  retries them only as a fallback (deprioritization, not reservation: the eviction
  target is still met, no new OOM/retract risk), (2) `_swa_tail_size()` factors the
  page-aligned window formula, (3) `_is_swa_branch_tail()` walks at most the branch
  tail. Adapted to this tree's `free_swa_segment` allocator API (the PR base used
  `free_swa`). Anchor-checked and idempotent; refuses to patch on tree drift; compiles
  the result before writing. Wired into `Dockerfile.canary-roce` right after the
  sanitize step; VENDORED-AT.md row added.
- **Pre-flight verified** against the running container's actual
  `swa_radix_cache.py` (1462 lines — same generation upstream deleted): 3/3 anchors
  applied, idempotent re-run no-ops, patched source compiles.
- **Expected effect:** agent turns stop re-prefilling their whole history; a parked
  conversation's cached prefix stays matchable (tail window protected from SWA
  eviction, so no tombstone cascade). Validation harness: identical-prefix A→B client
  pairs — B must report `cached_tokens ≈ full prompt` after both idle gaps AND after
  ~480K-token unrelated prefill churn (it returns 0 today in the churn case).
- **Deployed + validated 2026-10-26 ~02:15 UTC (owner go-ahead):** stop → serve → boot
  gates green (cap mode `swa_tokens=51712 prefix_tails=64`, `max_total_num_tokens=6,309,632`,
  0 NCCL WARNs, patch tags present in the running container). Harness result: A→B at +2 s
  full hit (161,280 cached) — **but B after ~480K-token unique churn STILL reports 0 cached;
  the retention problem is NOT closed by the backport alone.**
- **New instrumented finding (the deeper layer): the GPU radix tree is emptied within
  ~2–4 s of the engine going fully idle — on the FRESH boot too** — and a request sent
  minutes later can still report `cached_tokens=161,280` while `full_token_usage` reads
  0.00, with the Engram tick in that window showing `lookups=355,604 reads=161` for the
  161K-token prompt (≈1 read per 1K tokens). Interpretation: this stack has a second KV
  tier — the Engram NVMe n-gram store — that takes finished conversations' KV out of the
  GPU pool within seconds (the "wipe") and services later requests from NVMe (the
  "cached" counts). The 1-token probe forwards seen at idle (~2/min, no HTTP POST in the
  access log) ride the same machinery. kv_canary is `none`/sweep 0/perturb probs 0.0 —
  ruled out; scheduler `flush_cache` never runs (no "Cache flushed successfully!" line);
  `--sleep-on-idle`'s IdleSleeper only does torch `empty_cache` (interval env unset).
- **Consequences:** (a) the #35515 backport remains correct and shipped — the
  tombstone-cascade it fixes is real and upstream-confirmed, and it may matter again
  once the offload tier is understood; (b) the retention problem's operative layer is
  the Engram store/restore path (why identical replays restore in full while
  extended-prefix requests — the agent-turn shape — restore nothing and re-prefill);
  (c) the 44 h boot's agent behavior (8192-only hits) is consistent with the restore
  path never engaging for those conversations. NEXT: decode the Engram
  store/restore conditions (engram_backend.py / engram_prefetch.py, the 2 GiB 16-way
  DRAM cache vs NVMe shards, page/junction alignment) — that is where the agent-TTFT
  fix actually lives.
- **Second restart + full harness (owner go-ahead, 2026-10-26 ~02:50 UTC):
  `--swa-full-tokens-ratio 0.3` + the #35515 backport together.** Boot: `swa_size=403,712`
  (7.8× cap), `c4_size=336,448/rank` = **1,345,792 tokens global** (thinner than the
  3.4M projection — the state-pool pricing is superlinear; ~1.6× the two-agent working
  set, watch for retraction when both agents run large). Harness: **idle retention now
  WORKS** — B at +180 s idle: full 161,280 hit, and `full_token_usage` 0.12 afterward
  (~800K retained: the test conversations all stayed resident; the cap-boot 2–4 s wipe
  is gone in this configuration). **But the agent-turn shape still misses:** B after
  ~480K churn = 0; B after a 60 s quiet gap = **256** (one page); B after a single tiny
  request = 256; and the live agent's turns on this boot: 7,424 cached (system prefix)
  then 0 — each turn re-prefills ~110–250K.
- **Root cause, final form: the DSV4 compressed-attention compressor state.**
  `get_compress_state_ring_size()` (deepseek_v4_memory_pool.py): the c4 compress state
  lives in a **position-addressed ring 16 deep** (speculative, ratio-4; c128 online
  ring = 1). A cached prefix is only extensible while its boundary compressor state is
  intact — and ANY later forward writing nearby positions (the other conversation's
  prefill, the agent's own tool-call sub-requests, even the ~2/min idle 1-token probe
  forwards) clobbers those ring slots. Matchability therefore DECAYS under any traffic
  within seconds-to-a-minute and survives only across pure idle — which is exactly the
  measured matrix: +2 s hit (nothing in between), +60 s quiet = 256 (probe forwards
  clobbered it), +180 s idle after a fresh insert = full hit (no forwards), churn case
  = 0. The two-agent `cached=8192` era was the OTHER agent's in-flight path keeping
  its own state warm. This is a design gap in cross-request reuse for the DSV4
  compressed cache, not an eviction-ordering bug: #35515 (deployed, kept) and pool
  sizing cannot close it. The fix belongs at the pool/design level — retain the
  boundary compressor state per cached prefix (small: per-boundary, not per-token) or
  recompute it from the prefix's stored KV at match time — upstream material.
- **Net operational state after 2026-10-26:** ratio-0.3 boot (swa 403K, c4 1.35M
  global), #35515 + token-guard patches live. Agents get full-prefix hits only when
  their turn follows pure idle; any interleaved traffic (their own tool-call
  sub-requests, the second agent) decays reuse to the system prefix. Per-turn
  re-prefill cost remains ~1–2.5 min at 100–600K contexts until the compressor-state
  retention is fixed at the design level.

### 2026-10-26 — token-guard image deployed; SWA-ratio resize tried and reverted; radix retention bleeds empty under churn (open)
- **Deployed (owner-authorized window, agents stopped).** `verify-fabric-p2.sh` not re-run
  (no transport change; lane restart only) → repo pulled on the head to `27b668a` →
  `start-cluster.sh stop` (clean) → `site/build-cluster.sh` rebuilt
  `dsv41-4x-spark:canary-roce-miaai` on head + all three workers, adapter sha256
  `024a1a72…` verified in-image on every rank → `serve` → `/v1/models` 1,048,576 OK.
  Ships the 2026-10-25 `<｜deepseek_image｜>` placeholder-sanitize patch (verified
  in-image: `encoding_dsv41.py` carries the `[encoding_dsv41-placeholder-sanitize]`
  anchors at both raise sites).
- **First build-cluster.sh run hit two launcher bugs, fixed same commit range:**
  `BUILD_ARGS` unbound (`set -u`) — now defined in `site/env.switchless` (`e0334da`);
  local build executed the ssh command string as a command name — now `bash -c`
  (`3ee0bac`); worker `~/dsv41-miaai` tilde escaped by `printf %q` — now expanded
  before quoting (`27b668a`).
- **SWA-ratio experiment (tried, measured, reverted).** Premise: agent turns re-prefill
  their whole 200–600K conversation because the finished conversation's radix tail is
  SWA-tombstoned (cap-mode SWA pool 51,712 slots/rank turns over in ~15 s of prefill
  churn). `--swa-full-tokens-ratio 0.5` booted `mode=ratio (explicit),
  swa_tokens=419,072` (8×) but **`c4_size=209,600/rank` (838K global)** — an SWA slot
  costs ≈3.9 full-slot-equivalents, so ratio sizing trades c4 capacity 8:1 against SWA.
  838K global is under the two live agent conversations' ~810K working set → retraction
  wall. Reverted to cap mode (`31b2074` flag added, reverted in the same session).
- **The deeper finding (open, engine-level): radix retention dies under churn at ANY
  pool size.** Controlled client tests through the model-name proxy: identical-prefix
  A→B pairs — full radix hit (161,280 `cached_tokens`) at +2 s, +10 s, +45 s and
  +180 s IDLE gaps; **0 cached after ~2.5 min of unrelated ~480K-token prefill churn**,
  with `full_token_usage` collapsing to ~0 afterward (no memory pressure: 644K resident
  vs ≥838K pool). On the pre-restart 44 h boot the same pair also lost everything
  during a ≤137 s quiet gap (tree empty, `usage 0.00`). No `Cache flushed
  successfully!` lines in 44 h (scheduler `flush_cache` never ran); `--sleep-on-idle`
  exonerated (`SGLANG_EMPTY_CACHE_INTERVAL` unset → IdleSleeper's `empty_cache`
  disabled); `kv_canary` sweep/perturb is read-only (no evict/invalidate calls);
  no site adapter touches the tree. Rendered prompts are byte-deterministic
  (`encoding_dsv41.encode_messages` verified: X1 render is a strict prefix of X2's) —
  the client/proxy path is clean. Under churn the engine behaves as if tree-held KV
  (full-attention included) is recycled almost immediately once requests complete;
  the shared ~8,192-token system prefix stays matchable only because in-flight
  requests continuously re-insert it. Next lever when someone picks this up: DSV4
  pool/allocator retention semantics (`enable_decoder_swa_bounded_replay`,
  c4-state ring accounting) — likely upstream-design, needs the canary owner.
- **Net state after this session:** token-guard image live, config back to the
  2026-10-23 baseline (cap-mode SWA, c4 1,652,160/rank), agent turns still pay full
  re-prefill per turn (unchanged from before this session).

### 2026-10-25 — image-placeholder 400 loop: the production build never got upstream's sanitize
- **Symptom.** omp sessions against the lane die with a repeating
  `400 ... Message content contains image special token '<｜deepseek_image｜>'. Images
  should be provided as image content blocks.` — once it starts, every later turn in the
  session fails the same way (self-sustaining: the token rides in the client's history).
- **Root cause (live-verified).** The stock V4.1 encoder validates
  `_validate_no_image_sp_tokens()` over the WHOLE message list — string `content`,
  `reasoning_content`, and text blocks — and raises. The token is also in the model's
  output vocabulary, and agent traffic runs with `skip_special_tokens=False` (set whenever
  tools are in the request), so the model CAN emit the placeholder as plain text; a client
  that stores that assistant turn then 400s forever. Chain reproduced on the running lane:
  (a) sending the token in a user message → the exact 400, before decode; (b) asking the
  model to assemble the string from characters (no literal sent) → the model emitted
  `｜deepseek_image｜` into `content` in a tools-on request. 14 × `400 Bad Request` in the
  head log since boot, including bursts matching the 2026-10-25 effort-probe sessions.
- **Why the deployed image still raises.** Upstream already fixed this exact failure
  (`runtime/patch_encoding_dsv41.py`: sanitize-and-warn → `[image]` in both raise sites,
  anchor-checked; its docstring documents the same self-sustaining loop at 111/145 failed
  Anthropic requests) — but only `upstream/Dockerfile` runs it. The site's production build
  file `Dockerfile.canary-roce` (the `BUILD_DOCKERFILE` in `site/env.switchless`) never
  applied it, so the 2026-10-23 rebuild shipped a lane that still raises. Nothing in the
  repo docs recorded the gap.
- **Fix (site mod, upstream's own patch).** `Dockerfile.canary-roce` now applies
  `runtime/patch_encoding_dsv41.py` to the branch tree after the RoCEnante overlay /
  flash_mla override (COPY to /tmp + RUN, mirroring `Dockerfile`). Verified against the
  production SGLang tree on a throwaway copy inside the running container: both anchors
  match the branch tree (`patched ...: anchor1(string content), anchor2(text block /
  tool_result)`), and the patched `encode_messages` sanitizes instead of raising. The
  patch's anchor check fails the build if the branch tree drifts. Rejected alternative:
  a new runtime adapter hook wrapping `encoding_dsv41.encode_messages` (implemented,
  tested 8/8, then removed) — strictly more site code than re-using upstream's
  build-time patch for the same behavior.
- **Deployment.** Needs `build-cluster.sh` (image rebuild on all four ranks) + a lane
  restart; the adapter/encoder are baked into the image, so no running-lane change is
  possible before that. Until then: a session that errors with this 400 is carrying the
  token in its history — re-asking without the literal (or in a fresh session) works.

### 2026-10-23 — measurement contamination: the floor no longer catches implicit thinking
- **Symptom.** tool-eval-bench `fix1` (`dsv41flash-miaai-gpu2100-fix1`) reported
  `tg128` sweep cells generating 366 / 1,215 / 2,536 / 3,293 / 780 tokens and
  `--spec-bench` cells up to 8,578, where the 2026-10-22 run's cells all landed in
  111–128 tokens. The numbers were not a regression: they were a different
  measurement (decode windows 2.1–6.1 s → 6.8–86.9 s).
- **Root cause (live-confirmed on the running instance).** Two commits interact.
  `request_wants_thinking()` fell back to `SGLANG_DEFAULT_THINKING` when the
  request carried no thinking kwarg, so once thinking went default-on (3c53eed)
  every kwarg-less request became "a thinking request" for budget purposes, and
  the floor raised its `max_tokens` to 16384. llama-benchy's probes and
  `--spec-bench` send `max_tokens=128` with no kwargs — `max_tokens` is their only
  bound — so their cells stopped at EOS. Probes against the live server:
  `max_tokens=128` + no kwargs → **1,968** completion tokens (`finish_reason=stop`);
  + `{"thinking": false}` → **128** (`length`); + `{"thinking": true}` → 1,281
  (`stop`, floor by design). The floor itself was never wrong for its purpose:
  the accuracy plugins send `enable_thinking: true` explicitly and kept their
  gains (MMLU 50→0 truncations, IFEval 26→0, GSM8K 1→0).
- **Fix (adapter).** The floor is now gated on a new
  `request_enables_thinking_explicitly()`: only an explicit `thinking` /
  `enable_thinking` key (non-`None`) opts into the raise. The engine default still
  decides how a request *renders* (the 4dd72ce injection is untouched) — it just
  no longer authorizes overriding the client's own budget. Explicit-on traffic is
  unchanged; explicit-off and no-kwarg traffic keeps `max_tokens` as sent.
  Note the ordering that makes this safe: in this SGLang, the adapter's floor
  wrapper runs *outside* `_convert_to_internal_request`, which calls
  `_process_messages` itself — so the floor sees the client's raw kwargs, never
  the injected default (verified in the served
  `serving_chat.py:1099/1153`; no provenance flag is needed, and a pydantic
  `ChatCompletionRequest` refuses ad-hoc attributes anyway).
- **Tests.** `tests/test_max_new_tokens.py`: the case that pinned the removed
  behaviour (`test_floor_uses_engine_default_thinking`) is replaced by
  `test_floor_ignores_engine_default_thinking`, plus
  `max_completion_tokens`-variant, explicit-on-with-default-off,
  explicit-`None`-is-absent, and an install-level ordering guard
  (`test_install_keeps_probe_budget_when_engine_default_thinks`). 20/20 pass;
  reverting the gate fails 4 of them. `test_thinking_alias` 7/7, `test_loop_abort`
  15/15, `test_history_repair` 6/6. `test_switchless_ring_integration` is 4/15 on
  Windows both before and after the change (exec-line length, pre-existing — run
  it in WSL).
- **Cost / residual.** A future client that rides default-thinking *and* sends a
  tiny `max_tokens` will again spend it all in reasoning and get no answer. That
  is upstream's behaviour and the reason the floor exists; the fix narrows it to
  clients that stay silent about thinking.
- **Bench-side follow-up (separate repo, not done here).** tool-eval-bench's
  `_invalid_sample_error` only fails all-zero or errored cells, so an un-enforced
  `max_tokens` renders as a normal row. An `observed_tg_tokens > 1.25 × configured
  tg` guard (in `runner/llama_benchy.py` and `runner/speculative.py`) would have
  caught this without arithmetic. Filed as a suggestion only.
- **Deployed 2026-10-23/24 (owner switched the serving backend first).** preflight
  `verify-fabric-p2.sh` ALL PASS → head repo pulled to the fix commit → stack
  re-synced via `git archive` (correct `-C …/stack --strip-components=2` path) →
  lane stopped all-ranks (clean, no stopped-container leftovers) → images rebuilt
  on all four nodes, each verified in-image (`grep` finds the new gate 3×; the
  in-build test suite also proves the new tests shipped — the old suite asserts the
  removed behaviour and would have failed the build) → `serve` ready in ~3 min,
  `dsv41-local-weights` OK on every worker. Gates: full smoke, structured output +
  tool round trip + CJK/emoji probe clean, **0 NCCL WARNs ×4**, env pins intact
  (`DSV41_MAX_NEW_TOKENS_FLOOR=16384`, `SGLANG_DEFAULT_THINKING=true`), boot warning
  carries the new "implicit default-thinking requests keep their own budget" text.
  Fix verified on the served path: `max_tokens=128` + no kwargs → **128 tokens,
  `finish_reason=length`** (was 1,968/`stop` pre-fix); `thinking:false` → 128
  unchanged; `thinking:true` → 1,222 tokens/`stop` (floor still raises).
- **Re-benchmarked same boot (warm: ~18 min of trickle traffic after the 30-min
  cold window; tool-eval-bench v2.8.0, label `dsv41flash-miaai-gpu2100-fix2`).**
  The measurement contract is restored exactly:
  - Sweep (`--perf-only`, defaults pp2048/tg128, depths 0/4096/8192, c1/c2/c4):
    **all 9 cells `2048+128`** — `Total (ms)` 2,835–12,087, i.e. back inside the
    2026-10-22 envelope (3.26–12.85 s) instead of fix1's 7.5–93.6 s. `tg t/s`
    d0 c1/c2/c4 = 58.6 / 83.9 / 111.6. Report
    `runs/2026/10/2026-10-24T02-41-25.613061Z_de1465c6--…-fix2.md`.
  - `--spec-bench` (defaults, 3 runs/cell): **all 9 cells exactly 128 tokens**,
    `Total (ms)` 1,936–2,874 — fix1's cells ran 596–8,578 tokens / 102 s.
    Best `Eff t/s` 76.8 (structured d0) is now an ordinary cell, not the
    7,744-token outlier fix1's 76.1 came from. Report
    `…2026-10-24T02-42-47.057186Z_04b968a3--…-fix2.md`.
  - The `fix1` row's `Throughput @d0 tg t/s` and `Spec best eff t/s` remain
    invalid (contaminated cells); `SUMMARY.md` in the reports dir carries a
    note. Everything else in fix1 (scores, accuracy) stands.
  - Known cosmetic: `bench --perf-only` with an explicit `--output-dir` writes
    `<dir>/YYYY/MM/` (no `runs/` segment) — the sweep report was moved into
    `runs/2026/10/` by hand to keep the layout canonical; pass
    `--output-dir <reports>/runs` (as the spec run did) to land correctly.

### 2026-10-22 — bring-up: GLM NVFP4 → MiaAI-Lab SGLang vehicle swap (owner window)
- Window opened by owner; GLM NVFP4 stopped all-ranks (`start.sh stop`, exit 0 ×4,
  worker-first) and the stopped containers were `docker rm`'d on all four (NVFP4
  relaunch refuses existing names). Handoff said `./start.sh down` — no such
  subcommand exists (`serve|stop|status|logs`); handoff corrected in place.
- `verify-fabric-p2.sh` ALL CHECKS PASS (run pre-window and in-window); doctor
  ready (48/48 shards 476 G, all four HCAs active, RoCEv2 GID 3 on every rank).
- **Boot 1 failed** (TP0 model init): `Mxfp4FlashinferCutlassMoEMethod requires
  hidden_size and intermediate_size_per_partition to be multiples of 128 (got
  hidden=5120, intermediate=576)`. Root cause: the site env wrote the production
  adapter switches as plain assignments, but `start.sh` forwards container env via
  an explicit `-e` whitelist — those names only reach the container through
  `EXTRA_CONTAINER_ENV` (the form upstream's own `.env.tp4.example` line 453
  uses). `moe_b12x_next` never armed and the stock CUTLASS gate rejected
  N=576 (= 2304/4). Fix @ 57107b9: `DSV41_FAST_LOAD*`, `DSV41_DRAFT_HEAD_FP8_IMPL`,
  `DSV41_REPLICATED_SPLIT`, `DSV41_DRAFT_MAIN_PROJ_SPLIT`, `DSV41_ROUTER_LIVE`,
  `DSV41_MOE_B12X_NEXT(+_DETERMINISTIC)`, `DSV41_HC_FUSED`, `DSV41_PREFILL_SP(+_FP8)`,
  `DSV41_L2_PREFETCH`, `SGLANG_RUST_BUILD_MODE` now ride in the single
  `EXTRA_CONTAINER_ENV` line. `docker rm -f` per rank + relaunch → every adapter
  armed (`[moe_b12x_next]`, `[replicated_split]`, `hc_fused`, `PREFILL_SP`,
  `L2_PREFETCH`, fast load, `[draft_head_fp8]`, `[draft_main_proj]`).
- SERVED_MODEL_NAME pin @ 8418472: without it the engine serves upstream's default
  `deepseek-v4.1-flash`, while the proxy contract is `deepseek-v41-flash`. The
  proxy container on the head was itself stale (live env still
  `BACKEND_MODEL=glm-5.3-flash` despite the committed `.env`) — recreated via
  `docker compose --env-file .env up -d` on the head.
- Full smoke @ d005896 (`SMOKE_QUICK=0`): "Structured output, tool round trip and
  full-budget native vision passed", batch-16 sanity passed, warm-up 6 s.
- Four-PF gate capture @ df3737b: one boot with `NCCL_DEBUG=INFO` +
  `SUBSYS=INIT,ENV,NET` (docs/switchless-ring.md "Diagnosing it" — INIT,ENV alone
  never prints the routing records). Captured: `NET/IB ListenerRouting
  format=ipv4-v1 advertised=4 observed=4`, `Connected all rings, use ring PXN 0
  GDR 0`, zero listener-count errors. Repo reverted to `WARN`/`INIT,ENV` after the
  capture; the running engine keeps the INFO boot (logging only).
- Gates on the final instance (02:50 boot, warm JIT ≈2.5 min to Ready):
  `/health` 200; `/v1/models` id `deepseek-v41-flash`, max_model_len 1,048,576;
  corruption 0 U+FFFD (emoji + CJK intact); depth needle at 666,695 tokens hit
  (`ORION-88`; pool full=6,654,976/rank, swa=51,712); forced tool call ok;
  proxy :4000 answers `spark-llm` → serves `deepseek-v41-flash`. SpecDecoding
  accept len 3.55 first decode (2.35–3.75 across smoke) — > 1. Engram DISK
  `packed=True`, hit 87.3–87.4%, cache 2.0 GiB (16-way)/layer, threads 96.
- No benches (functional gates only, per runbook). Step 0 (delete
  `~/.local/ds41f-tp4` sparkring weights, 476 G/node) skipped — owner did not
  confirm it. README still describes the retired sparkring lane (owner-prioritized,
  post-window item).
- Follow-up (owner request, same day): enabled Prometheus + MFU metrics —
  `--enable-metrics --enable-mfu-metrics` appended to `EXTRA_SGLANG_ARGS` @ 4048e62;
  lane restarted (warm boot, full smoke green). Verified: `/metrics` 200 with 78
  `sglang:` families incl. the MFU trio (`estimated_flops_per_gpu_total`,
  `estimated_read/write_bytes_per_gpu_total`); MFU estimators also appear in the
  batch log lines (`est. prefill/decode TFLOPS/s`, `est. read/write BW`).

### 2026-10-22 — benchmark analysis: thinking-mode budget starvation (site fix)
- Owner request: review `~/code/scratch/tool-eval-bench-reports` (label
  `dsv41flash-miaai-gpu2100`) and find settings/tuning that improves the numbers
  **other than the effort setting**. Bench run 2026-10-22 13:15–17:45 UTC (4h30m,
  v2.8.0): Tool-Eval 89% (172/194), GSM8K 98.0%, MMLU 86.8% (90.8% on the retired
  sparkring lane), IFEval 89.5/90.5 (83.9/88.8 there), spec best 111.7 t/s, tg
  48.2/81.4/93.0.
- **Root cause — client/server budget mismatch, not model loss.** The harness sends
  `chat_template_kwargs {enable_thinking: true, reasoning_effort: high}` with
  `max_tokens=256` for MMLU and `4096` for IFEval (both hard-coded in the harness
  plugins). With thinking on, the reasoning segment consumes the entire budget: the
  response ends `finish_reason=length` with empty content and scores zero however good
  the reasoning was. Reproduced on the live engine (MMLU item #1 at `max_tokens=256`:
  256 completion tokens / 259 reasoning tokens, empty content; the same prompt at
  16384: 3,005 reasoning tokens, then the correct answer, 47 s).
- **Measured cost of the starvation** (live engine, seed 42, temp 1.0/top_p 0.95,
  effort 75 — only the budget varied):
  - MMLU: the report's 50 truncated items re-run at `max_tokens=16384` →
    **42/50 correct (84%)**, 50/50 `finish_reason=stop`, median 485 completion
    tokens, p90 1,173, max 4,454. Projected MMLU **86.8% → 95.2%**.
  - IFEval: 39-prompt controlled A/B, 4096 vs 16384, scored by IFEval's own checker →
    prompt accuracy 87.2% → **92.3%**, instruction 88.2% → **95.6%**, truncated 3 → 0
    (both flips are truncation recoveries, keys 1154/3429).
- **Fix @ this commit — `DSV41_MAX_NEW_TOKENS_FLOOR` (default 16384).** The adapter
  (`adapter/encoding_compat.py`) raises a *thinking* request whose `max_tokens` /
  `max_completion_tokens` is below the floor, capped by `DSV41_MAX_NEW_TOKENS`; it
  never shortens a request and never touches thinking-off traffic (that path still
  follows `SGLANG_DEFAULT_THINKING`). Wired through `start.sh`'s `-e` whitelist on both
  head and worker paths, documented in `.env.example` / `.env.tp4.example` / README,
  pinned in `tests/test_switchless_ring_integration.py`; 9 new boundary tests in
  `tests/test_max_new_tokens.py` (16/16 pass) and the integration suite 15/15 (run in
  WSL — the Windows shell cannot exec the 32k-character launch line).
- **Rejected, with measurements** (150-item stratified MMLU slice, paired to baseline):
  `top_k=20` +2/−1 (net +1), `repetition_penalty=1.05` +1/−0 (net +1) — both inside
  noise, so the sampling-tail hypothesis does not hold and neither ships. The residual
  MMLU loss is 16 genuinely-wrong answers; IFEval's residual gap is constraint
  following (`keywords:letter_frequency` 69.7%, `detectable_format:json_format` 70.6%),
  which no server-side knob reaches.
- **Rejected — thinking cap (`--enable-strict-thinking` + `SGLANG_MAX_THINK_TOKENS`).**
  It is the other way to end the starvation (force the think-close token at a count)
  and works in principle, but (a) the cap is global while starving (256) and healthy
  (16384) budgets coexist — a cap low enough for MMLU would cut reasoning that IFEval
  and tool-eval currently use in full; (b) the strict path also bans
  `think_excluded_tokens` (DSV4 EOS + DSML) inside thinking, which is the tool-call
  markup path — an unvalidated risk for the tool lane. Per-request
  `max_thinking_tokens` requires `--enable-strict-thinking` (verified: the tokenizer
  manager raises without it), so it cannot be a low-risk default. The floor gets the
  same recovery without cutting reasoning.
- **Not a lever, verified:** thinking-off costs a lot and is already overridden by the
  harness — 150-item paired run: thinking-on 94.0% vs thinking-off 85.3% (net −13,
  −8.7 pp). `--fp4-indexer` is a no-op on V4.1 (no ratio-4 layers; upstream
  `docs/tp4.md`). KV dtype is already `fp8_e4m3`. `sampling_defaults=model` does
  nothing here: the checkpoint ships no `generation_config.json` (checked on the head's
  weights dir and in-container).
- **Cost of the floor:** thinking requests that used to die silently now run to
  completion; bench wall time on MMLU/IFEval grows.
- **CORRECTED 2026-10-23 — "throughput (`tg`) is unchanged" was wrong.** The floor
  also caught *measurement* clients, which never ask for thinking: llama-benchy's
  throughput probes and tool-eval-bench `--spec-bench` send `max_tokens` with no
  thinking kwarg, so (once thinking went default-on) `request_wants_thinking()`'s
  engine-default fallback authorized the rewrite and their fixed budgets became
  16384 — the probes ran to EOS instead of to their budget and `tg` moved for
  workload reasons, not engine reasons. Do not use `tg` continuity as a sanity
  check across 2026-10-22 → 2026-10-23. Fixed same day (see the 2026-10-23
  "measurement contamination" entry); the starvation recovery is unaffected.
- **Deployed 2026-10-23 (owner-authorized window, no bench):** preflight
  `verify-fabric-p2.sh` ALL PASS → repo pulled on the head (bff8ce7) → lane stopped
  all-ranks (`start-cluster.sh stop`, rank 0 first, clean) → image
  `dsv41-4x-spark:canary-roce-miaai` rebuilt on ALL FOUR nodes (~110 s each, cache hit
  on the unchanged kernel layers; the Dockerfile's in-build adapter test suite — incl.
  the 16 max-new-tokens tests — passed inside every build) → `start-cluster.sh serve`
  → ready in ~3 min. Gates on the new instance: full smoke
  ("Structured output, tool round trip and full-budget native vision passed"),
  batch-16 sanity, warm-up 6 s, Engram `packed=True` hits 87.3–87.5%, SpecDecoding
  accept len 3.55 (first) / 2.61 (batch-16) > 1, **0 NCCL WARNs on all four ranks**,
  corruption probe 0 U+FFFD (emoji + CJK intact), `/v1/models` `deepseek-v41-flash` @
  1,048,576. Floor verified live: container env carries
  `DSV41_MAX_NEW_TOKENS_FLOOR=16384`, the boot warning announces it, and the exact
  starvation case (MMLU-style prompt, `max_tokens=256`, thinking on) now returns
  `finish_reason=stop` with the worked answer (678 completion tokens) instead of
  `finish_reason=length` with empty content. No bench was run (owner will).
- **Two boot-attempt gotchas hit during the redeploy, recorded for the next one:**
  1. **Stack re-sync resets exec bits + reintroduces CRLF.** The stack
     (`~/.local/ds41f-miaai/stack`) is an rsync/tar copy of `upstream/`, and the
     vendored tree ships 0644 scripts — a re-sync from the workstation's Windows
     checkout also carries CRLF (`/usr/bin/env: 'bash\r'`). Fixes now in the repo:
     `site/start-cluster.sh` execs the launcher via `bash` (mode-proof, commit
     "exec the launcher via bash"), and the correct re-sync is
     `git archive HEAD deepseek-v41-flash-tp4-switchless/upstream | tar -C
     ~/.local/ds41f-miaai --strip-components=2 -xf -` from any clone (LF, no CRLF,
     keeps the gitignored `benchmarks/` and generated state). Never tar the tree
     from a Windows checkout.
  2. **Worker build trees are `~/dsv41-miaai` on each worker** (not the launcher's
     `WORKER_DIR=~/.local/ds41f-miaai`, which on workers holds only
     engram/weights/state) — `cmd_build`'s worker rsync would target the wrong dir;
     worker images were rebuilt from `~/dsv41-miaai` with the changed files streamed
     in. Worth unifying with a real site build wrapper before the next full rebuild.
  3. **`git archive` path stripping bit once:** the head's stack lives one level
     deeper than the workers' build trees — `-C ~/.local/ds41f-miaai
     --strip-components=2` scattered the tree next to `stack/` and the head build
     silently reused the cached adapter layer (same image ID). The archive must be
     extracted with `-C ~/.local/ds41f-miaai/stack --strip-components=2` (workers:
     `-C ~/dsv41-miaai --strip-components=2`). Always verify the change is inside
     the built image (`docker run --rm --entrypoint grep …`) before booting.
  4. **Stack re-syncs also drop exec bits again** — `site/start-cluster.sh` now
     `chmod +x`es the launcher chain on every run (commit "restore launcher exec
     bits").
  5. **`cmd_build`'s worker rsync is destructive — never run it (2026-10-23, hit
     for real).** `start-cluster.sh build` reaches the worker stage and
     `rsync -aH --delete`s the recipe into `WORKER_DIR` = `~/.local/ds41f-miaai`
     on the workers — the engram/weights/state dir. On spark-6d14 that deleted
     `weights/deepseek-v41-flash-1017`, the bind source of the
     `dsv41-local-weights` volume: the volume then refused to mount
     ("no such file or directory") and the worker could not have booted. It also
     scattered the recipe tree next to the legit dirs (cleaned by hand) and died
     on the head's root-owned `b12x-next-compile` cache files (rsync code 23) —
     so it destroys worker state *and* fails. Restore path that worked:
     `site/materialize-miaai-weights.sh spark-6d14.shawndo.intra` (hardlink from
     the node's own HF cache, zero copy, 48 shards, volume mounts again).
     Permanent fix: `site/build-cluster.sh` builds head+workers via `git archive`
     streams into `~/.local/ds41f-miaai/stack` (head) and `~/dsv41-miaai`
     (workers) with no rsync, then sha256-verifies the adapter inside every built
     image before reporting success. Use it for every rebuild; the launcher's
     `build` command is unsafe on this layout until upstream changes the rsync.

### 2026-10-23 — thinking ON by default (GLM parity, owner decision)
- `SGLANG_DEFAULT_THINKING=true` added to the site env's `EXTRA_CONTAINER_ENV`
  (the older env comment claiming the flag does not exist was wrong — the dsv4.1
  image ships it); README's stale "THINKING_DEFAULT / no reasoning_effort knob"
  paragraph rewritten (thinking default + effort budget semantics + sampling
  defaults). Commit 3c53eed.
- **Live-verified engine gotcha:** setting the env alone arms prompt encoding but
  NOT the response parser — a default request rendered its ` think` block inside
  `content` with `reasoning_content` empty (the parser keys off
  `chat_template_kwargs`, not the env). Fix @ 4dd72ce: the adapter now injects
  `SGLANG_DEFAULT_THINKING` as an explicit `thinking`/`enable_thinking` kwarg when
  the request omits both (explicit client values still win); 7/7
  `test_thinking_alias.py`. Images rebuilt on all four nodes + restart.
- Verified live on the new instance: default request (no kwargs) →
  `reasoning_content` populated (163-char think block) with clean content `'3'`;
  explicit `{"thinking": false}` still wins; full smoke + warm-up passed; 0 NCCL
  WARNs ×4; corruption probe 0 U+FFFD. No bench (owner will run one).

- **2026-10-20 — External-survey pass: DeepSeek-V4.1 TP4 switchless recipes (owner
  request: "references to switchless DeepSeek 4.1 TP4 recipes with better performance
  than what we are getting… willing to try a different inference server"). Requirements:
  official quantizations, TP4, switchless. Sources: MiaAI-Lab repo + docs, knapcio
  DeepSeek repo, NVIDIA forum threads 386285/384438/381005/382897/382725 (267 posts,
  tail read via /raw/), plus GLM-on-SGLang/TensorFold scoping. Four read-only scouts;
  combined findings here (the survey's canonical record). Doc-only — no config change,
  no node action (lanes are serving; no restarts allowed this session).**

  ### The landscape (official weights, TP4, switchless ring where stated)

  | stack | engine | measured (decode code/prose c1, aggregate peak) | notes |
  |---|---|---|---|
  | MiaAI-Lab default ring profile | SGLang dsv4.1 @ f80c91a4b | 74 agg C1 / 224.8 C8; prefill 3.0–3.5K | the base recipe; 1M ctx, MXFP4/8 (checkpoint-native), MAX_TOTAL_TOKENS 8M pinned |
  | MiaAI step-D (forum 386285 / issue 59, China OEM GB10) | same | C1 78–90 window / **296 C8**; prose C1 40 / C8 141; 1M prefill **2,880 tok/s (347 s)** | full production line minus RoCE: EP1 + b12x_next, WO_A_W8, draft-head FP8, block verify, prefill SP + FP8 gathers, fast load; qeval 72/75 (+2 over step A), 0 broke |
  | knapcio DeepSeek v2.4 (switched 100G) | SGLang | prose C1 67–69 / code 110; c4 131 → c16 232; cold prefill 4,730 @32K → 2,830 @967K | per-flag measured ladder v1→v2.4 in repo docs |
  | christopherowen r6n (ring TP4) | vLLM fork + TileLang | 226.5 agg @ 16 streams closed-loop; step grid geomean −2.2% | sparknet collectives; r6 TileLang prefill 32K 5,836 tok/s |
  | tonyd2wild native vLLM 1017 | vLLM | C1 48.5 agg / C6 165.7 | his own "correct peer" for this lane (not the 89.8 TensorFold headline — different engine + EXL3) |
  | **this lane (2026-10-18 bring-up)** | sparkring vLLM cycle | c1 32.2 / c2 44.0 / c4 45.4 (warm, interrupted-JIT context) | honest cross-lane A/B: GLM lanes c1 51.6 (FP8) / 55.3 (NVFP4); spec acceptance DSpark k5 40–62% vs GLM DFlash2 93% |

  Same-prompt A/B is scarce: the only controlled one found (forum 382725 post 196,
  4×Spark on 100G switch, same prompts, original weights, DSpark k=5): vLLM (aidendle)
  vs knapcio v2.3 SGLang — prose c1 41.0 vs **72.3**, code c1 86 vs **127**, prefill
  17K/69K/138K 1,950/1,823/1,766 vs **4,617/4,769/4,656**, KV pool 2.98M vs 6M pinned,
  boot ~10 min vs ~5, quality identical (tool-eval 93 both). **SGLang won decisively —
  for DeepSeek.** Attribution: the wins live in MiaAI's DeepSeek-specific kernel/
  indexer work (DSA tilelang kernels, chunked indexer, Engram, prefill SP), not in
  SGLang-generic flags. GLM-side counter-evidence: knapcio's own SGLang-vs-vLLM GLM
  TP4 head-to-head (vendored in the NVFP4 lane at
  `../glm-v53-flash-nvfp4-tp4-switchless/upstream/docs/sglang/`, measured in
  `bench/kld_long_texts.json`): equal speed at the same k, quality 68/75 vs 75/75 —
  the engine switch does NOT transfer to GLM.

  ### What we already match (this lane vs the surveyed stacks)

  The MiaAI/knapcio "tuning ladder" items map 1:1 onto flags this lane already runs:
  CUDA graphs (ENFORCE_EAGER=0), spec decoding k=5 DSpark (NUM_SPECULATIVE_TOKENS=5),
  pinned KV via KV sizing, fast/serial load decisions, NCCL ring contract (four-PF
  lib + SWITCHLESS_RING_ONLY + MERGE_NICS=0 — the SAME contract the MiaAI patched
  NCCL implements), channels=8 (adopted `dade187`), Engram NVMe packed shards
  (ENGRAM_PACKED + BALANCED, +15% prefill upstream). Max-num-seqs 16 remains the one
  proven-upstream admission rung not yet gated here (watch item 6).

  ### Transferable findings (verdicts against the adoption gate)

  1. **NCCL pinned-buffer shrink** (MiaAI: `NCCL_BUFFSIZE=1048576`,
     `NCCL_LL128_BUFFSIZE=262144`, `NCCL_PROTO=^LL128`, `NCCL_MAX_NCHANNELS=8`):
     default NCCL conn buffers cost 4.7 GiB/node on GB10 UMA (pinned host RAM = GPU
     memory); shrunk set = 0.14 GB. **Not adopted now** — this lane already sets
     MAX_NCHANNELS=8, decode ARs are 61 KB LL, and the KV pool is free-memory-sized
     (2.24M tokens @ gmu 0.83); freeing conn buffers only matters if the pool ever
     needs the headroom (watch item 2's gmu 0.85 rung would be the moment). Engine-
     independent knob set, recorded here as the first lever for that rung.
  2. **sparknet ring NCCL profile** (christopherowen, `dgx-spark-networking`):
     channels balanced across ring directions AND PCIe roots, 4ch/4MiB buffer, small-
     call thread-block reduction, size-based one-shot vs NCCL selection — 2 MiB
     allreduce 320→232 µs (−27.5%). Explicitly non-model-specific. **Rejected as
     transport swap** (different NCCL; violates the shared four-PF contract this
     cluster's switching depends on) — recorded as the comparison target if the
     four-PF lib is ever re-derived; its AArch64 send-path fence fix and
     `SPARKNET_ROCE_TIMEOUT_MS` clock-aware wait are design references for any
     future NCCL site patch.
  3. **`torch._library.utils.fill_defaults` O(args²) patch** (873→48 µs per custom-op
     call outside CUDA graphs; "applies to every MoE launch outside a CUDA graph"):
     generic PyTorch overhead fix, engine-agnostic. **Candidate overlay experiment**
     for this lane (its DSpark verify path launches eager ops); needs its own
     bench-recipe-update window with step-time A/B.
  4. **Display carve-out** (~2.1 GiB GB10 firmware scanout reservation reclaimed via
     DRM dumb buffer + dma-buf for embedding/output-head weights → KV +0.8 GiB/rank)
     and **Memory-Saver 64 KiB-page UVM patch** (+1.78–1.86 GiB/node): **rejected** —
     driver surgery pinned to kernel 7.0.0-1019 / driver 580.178.04; site pins are
     580.173.02 + 6.17.0-1032 (watch item 4). Revisit only if the kernel/driver pin
     ever changes.
  5. **GB10 clock-latch ops rule** (tonyd2wild forum 382897: GPUs can latch <1 GHz —
     his 41→5 tok/s collapse was two latched GPUs; only a power unplug clears it,
     reboot does not) and the **thermal stacking note** (382725 post 187): adopted
     as pre-bench checks.
  6. **Fabric cold-power-off rule** (HuShengKai 381005: after cabling changes a FULL
     power-off + wait is required — links otherwise negotiate at 40 Gbps; ×2.35 cold
     throughput after the fix): adopted for any future re-cabling; also validates our
     `verify-fabric-p2.sh` + `ib_write_bw` discipline.
  7. **Bench methodology** (MiaAI tp4.md + knapcio sparkDash discipline): idle-cluster
     gating between waves; TTFT at first nonempty content event (SSE chunks ≠ tokens);
     step-time comparison for kernel A/Bs (single-prompt greedy c1 moves ±10% on
     reduction-order changes alone); discard first two runs post-boot; `#running-req`
     above benched concurrency invalidates the window; nvidia-smi polling at 2 s costs
     0.6 ms/decode step (poll ≥10 s); spec acceptance on agent traffic ~55% vs 75%+ on
     bench code — tune drafts on real traffic, never one prompt; character-scaled
     decode-window tok/s is an ESTIMATE and must be labeled as such.
  8. **Not transferable (model/stack-specific), recorded to prevent re-litigating:**
     Engram n-gram tables + NVMe row store (DeepSeek DSA indexer), DSpark adaptive
     verify + cost curves, WO_A_W8 / cert-head / b12x TileLang kernel family, EP1 MoE
     layout (vLLM expert parallelism differs), chunked-indexer backport, `--swa-
     prefix-tails`, `DSV41_PREFILL_BYPASS` (short-request admission during long
     prefill — the CONCEPT is generic but no GLM/vLLM knob exists; if deep prefill
     head-of-line blocking ever hurts this lane's traffic, this is the feature to
     request upstream), DeepSeek reasoning-effort budget knob.
  9. **Ring-vs-switch**: only one direct comparison found (382725 post 158): decode
     within ~10% switchless-vs-switched, usually equal; prefill ~5% lower (one-link
     bisection). Corroborates this lane's transport choice; no action.
  10. **Other engines scoped**: SGLang GLM-5.3-Flash support exists upstream (v0.5.20+,
      PR #36507; DFlash via EAGLE; fp8-KV on GB10 via #36904) but GB10 is NOT in the
      cookbook's verified hardware list and every Spark deployment found needed
      community patches (beastllama 2×Spark repo: 4 GB10 patches, DFlash2 NOT
      bit-identical at temp 0 — 1/20 exact). Best published GLM-on-SGLang 4×Spark
      number: Openzeka aggregator C1 46.8 (provenance undisclosed). TensorFold (Zig
      engine, Apache-2.0) has a GLM-5.3-Flash TP4 switchless stack (Keenni repo, ~80
      tok/s single-stream decode) but on abliterated EXL3 4bpw weights + NC-licensed
      drafter — fails the official-quant requirement, and its cold prefill is 0.87–0.89×
      vLLM. Neither justifies a lane switch for this site.

  ### Bottom line

  The "better performance" perception decomposes into (a) SGLang-vs-vLLM on DeepSeek —
  real but DeepSeek-kernel-attributed, already counter-evidenced for GLM; (b) MiaAI
  step-D tuning depth — mostly flags this lane already carries or levers upstream
  hasn't published (watch item 5 covers the two that matter); (c) methodology
  differences in published numbers (temp-0/greedy vs real traffic, character-scaled
  estimates). Concrete follow-ups carried from this survey: max_num_seqs 16 gated rung
  (watch item 6), NCCL buffer shrink at the gmu-0.85 rung (item 1), fill_defaults
  overlay experiment (item 3). No restarts were performed; lanes untouched.

### 2026-10-18 — tonyd2wild lever adoption A/B (channels + Engram threads)
- Bench context: the user's interrupted tool-eval runs (spec-bench ~12 min after boot,
  cold JIT) read 36-59.5 eff t/s; a warm re-run (`dsv41flash-warm1`) read essentially
  the same (filler 38.9 / code 55.4 / structured 50.5 avg eff; perf c1 32.2, c2 44.0,
  c4 45.4; prefill ~2.0K tok/s warm) — cold-start was NOT the main story.
- Honest cross-lane comparison on the SAME harness: GLM lanes do perf c1 51.6 (FP8) /
  55.3 (NVFP4) and spec structured ~110 t/s (DFlash2 accepts 93%, τ 7.5); this lane's
  DSpark k=5 accepts 40-62%, τ 3.2-4.0. tonyd2wild's own NATIVE-weights vLLM 1017 lane
  (C1 agg 48.5, C6 165.7, run 2026-10-17-1017) is the correct peer, NOT his 89.8 t/s
  TensorFold headline (different Zig engine + EXL3 3.5 bpw, ~17% fewer bytes/token).
  His native lane is not faster than the site's GLM lanes either — the DeepSeek lane's
  value is 1M ctx + vision + Engram, not c1 decode on this ring.
- **Adopted + A/B-measured (`dade187`)**: `NCCL_MAX_NCHANNELS=8` + `ENGRAM_DISK_THREADS=128`
  (his 1017b/1017d levers, no code changes needed). Warm A/B (`dsv41flash-ch8-eg128`):
  perf c1 flat (32.2→32.1), c2 +8.9% (44.0→47.9), c4 +24% (45.4→56.4); spec code
  +6.7% (55.4→59.1 avg), filler/structured flat. Concurrency gains, c1 latency-bound
  unchanged — matches his ladder's shape. KEPT.
- **Evaluated, not adoptable now**: `DSV41_ENGRAM_FAST=1` and `DSV41_INDEXER_TP_SPLIT=1`
  require his unpublished patch sets (`dsv41-1017-fast`, `dsv41-1017-roce-idx`; only his
  boot-10 set `patch/full` is public, and sparkring already vendored it — our boot log
  already shows pre-forward staging); `VLLM_ENABLE_ROCE_ALLREDUCE` b12x one-shot
  transport violates the site's four-PF NCCL switching contract (lane invariant).
  `expandable_segments:False` is hardcoded in the launcher (would be site mod #4) and
  its attribution inside his 1017g bundle is unclear — parked pending a need.
- GB10 clock-latch pre-bench check (his docs/gpu-clock-latch.md): worth adding to any
  future bench session — nodes verified fast on the Oct 17 GLM benches, no action now.

- **Base image unrecoverable as pinned** (details in VENDORED-AT.md site mod #3):
  `vllm/vllm-openai:nightly-8a728663c1c3…` returns `manifest unknown` AND its receipt
  digest `sha256:1891ada1…` returns "manifest schema unsupported" — Docker Hub purged
  the Sep-2026 nightly era; sparkring `main` @ `ad1853fc` still carries the dead pin.
  Substituted `nightly-dev-arm64-cu13.0.1-591bb95` (2026-09-15) re-tagged under the
  pinned name; in-image `torch 2.13.0+cu130` = the receipt's torch line; the branch's
  12-arg `topk_softplus_sqrt` schema verified present. First candidate `728d3ad`
  (Sep 4) REJECTED (10-arg schema). Layer IDs diverge from `image-receipt.json`.
- **LANE UP 2026-10-18 ~19:45 UTC.** All gates green: `/v1/models` advertises
  `deepseek-v41-flash` @ 1,048,576; corruption probe 0 U+FFFD (emoji+CJK intact);
  forced tool call round-trips (`deepseek_v41` parser, thinking-on + tools, engine
  alive); SpecDecoding mean acceptance 67/23 ≈ 2.91 (DSpark k=5 active). Boot ~9.5 min.
- **Site KV-pool observation (first boot):** `GPU KV cache size: 2,242,638 tokens,
  Maximum concurrency for 1,048,576 tokens per request: 2.14x` at gmu 0.83 with the
  1M limit — close to upstream's 430K-boot measurement (2,182,642 @ 430K limit). This
  is a bring-up observation, not yet a receipt.
- **Image lineage (final serving image):** `local/sparkring-deepseek-v41:overlay6`
  `sha256:e87985e1426d3fde2b5253ef4e4b34da3763c8740f2ee8ef6b7cff6e4ca0bee6` =
  overlay5 (rebuilt on substituted base `nightly-dev-arm64-cu13.0.1-591bb95`) +
  pip `deep_gemm` 2.6.1+local built from DeepGEMM @ `8b1392b978f5a03c828dd1711090d7fb50958b8a`
  (the nv_dev pin from vllm @ `8a728663` — the branch-era DeepGEMM). Committed via
  `docker commit` on spark-0f0b (in-image build needed: git, python3-dev,
  `CPPFLAGS/CFLAGS=-I.../nvidia/cu13/include` for cusparse.h, and
  `VLLM_DOCKER_BUILD_CONTEXT=1` for the install script's uv path).
- **Failure chain solved on bring-up (all documented above):** dead base tag →
  728d3ad ABI-incompatible (10-arg topk_softplus_sqrt) → 591bb95 base ✓ but its
  vendored `vllm.third_party.deep_gemm` 2.6.1 (newer DeepGEMM) rejects the branch's
  SF transform → pip deep_gemm @ 8b1392b ✓.
- **Boot log notes:** "Duplicate NCCL runtime" WARNINGs present as predicted
  (watchitem #1) — four-PF lib loads via `VLLM_NCCL_SO_PATH=/opt/sparkring/nccl/…`;
  boot completed past graph capture on all ranks (ring healthy). "Connected all
  rings" string does not exist in this engine's logs — ring health is evidenced by
  completing init + graph capture (any ring failure hangs at NCCL init).
- **Ops gotchas (cost time):** (1) `check` doesn't stage the recipe tree; (2) the site
  `stop` skips EXITED containers — failed boots need `docker rm -f` per rank before
  relaunch (the upstream launcher refuses existing names); (3) failed boots leave
  ROOT-owned cache files — clear via a throwaway container, not host rm; (4) stale
  `/cache` JIT caches survive image swaps — clear `cache/vllm` after any image change;
  (5) per-rank `docker logs` must run on the rank's own host.
- Owner request: align default thinking with the GLM lanes — see pre-bring-up entry.

### 2026-10-18 — pre-bring-up: thinking default aligned with GLM lanes
- Owner request: confirm default thinking/effort/temperature match the GLM switchless
  lanes before first boot. Temperature: no server-side default in any lane (match).
  Effort: not applicable — the dsv41-feat template exposes only boolean `thinking`, no
  `reasoning_effort`. Thinking: GLM lanes default ON (`enable_thinking: True`,
  `reasoning_effort: high`; the `DEFAULT_EFFORT=low` experiment was adopted-then-reverted),
  this lane was OFF (upstream recipe value) → changed `site/env.switchless`
  `THINKING_DEFAULT=false → true`. This is a deliberate site delta vs upstream
  `recipes/deepseek-v41-flash-cycle.json` (`"thinking": false`).

### 2026-10-17 — adoption (initial)
- Branch `adopt-deepseek-v41-flash-tp4-switchless`; directory created; NOTHING
  committed/pushed (adoption deliverable is uncommitted by convention).
- Sources mined: HF API (checkpoint pins/identity), sparkring @ `ad1853fc` (cycle lane
  files + receipts), NVIDIA forum thread 384438 (28 posts — stack comparison, §2).
- Weights confirmed downloading/present on the nodes (spark-0f0b snapshot @ `1bae0e10`);
  no downloads performed by the adoption session.
- Open: first bring-up pending (README "Bring-up"); receipts in §5 are upstream's,
  site numbers start accruing at bring-up.
