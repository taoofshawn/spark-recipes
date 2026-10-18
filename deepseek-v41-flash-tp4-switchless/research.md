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

## Changelog

### 2026-10-18 — bring-up day 1: base-image substitution + thinking default
- **Base image unrecoverable as pinned**: `vllm/vllm-openai:nightly-8a728663c1c3…`
  returns `manifest unknown` AND its receipt digest `sha256:1891ada1…` returns
  "manifest schema unsupported" — Docker Hub purged the Sep-2026 nightly era; sparkring
  `main` @ `ad1853fc` still carries the dead pin (checked 2026-10-18). Substituted
  `nightly-dev-arm64-cu13.0.1-591bb95` (2026-09-15, arm64), locally re-tagged under the
  pinned name. Verified in-image `torch 2.13.0+cu130` = receipt's exact torch line AND
  the `topk_softplus_sqrt` 12-arg schema the branch requires. Layer IDs diverge from
  `image-receipt.json`; the receipt remains the upstream comparison target. Detail:
  VENDORED-AT.md site mod #3.
- **First boot failure (728d3ad base, rejected)**: every rank died in
  `profile_run → speculator.propose` with `_moe_C::topk_softplus_sqrt() expected at
  most 10 argument(s)`. Root cause: build-image.sh rebuilds only `_C_stable_libtorch`;
  `_moe_C_stable_libtorch.abi3.so` (which registers the `torch.ops._moe_C` namespace)
  comes FROM THE BASE — the Sep-4 base bundles an Aug-19 build with the pre-Engram
  10-arg schema. Base must postdate the branch's fork-era main; the Sep-15 base's
  schema matches exactly (probed before rebuild). Lesson recorded in VENDORED-AT #3.
- Build gotcha (cost one build attempt): docker `--memory` is unit-sensitive —
  `MEM=24` = 24 bytes = rc 125; use `MEM=24g`.
- **bring-up ops gotcha**: `start-cluster.sh check` does NOT stage the recipe tree
  (`sync_recipe` only runs in `serve`) — first `check` failed on a missing staged
  launcher; stage manually or run serve once. Watcher note: per-rank `docker logs`
  must run on the RANK's host (the site `logs` helper does this; don't grep all ranks
  from the head).
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
