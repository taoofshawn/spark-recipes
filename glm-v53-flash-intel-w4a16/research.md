# research.md — glm-v53-flash-intel-w4a16: maintenance notes

Working notes for future update/maintenance sessions on this recipe. The
README is the deploy doc; this file is the memory.

## Lane receipts (author-reported, different images/protocols — do not mix)

> The legacy `dflash2` lane was **removed 2026-09-11** (see "Cleanup: drop the
> dflash2 lane" below); its receipts are retained below for reference only.

| | dflash2pmu (default) | mtp3 |
|---|---|---|
| spec decoding | DFlash2 k=7 + `disable_eagle_block_drop` | native MTP3 |
| prefix matching | `--prefix-match-unit 128`, patches baked | `--prefix-match-unit 128`, patches baked |
| drafter rev | `bf582e4e…` (08-31) | none |
| KV pin / pool | 13.5 GB → ~1.81–1.87M tokens | 13.5 GB → 1.92M tokens |
| seqs | 6 | 6 |
| receipts | florianbrede: TEB **90** (158/176), PP 1,679, TG 33.1/54.7 @C1/C4, ~82-token recompute | florianbrede: TEB 91, 108 tok/s @ C6 |

**Removed lane (reference only):** legacy `dflash2` — DFlash2 k=7 (no
`disable_eagle_block_drop`), block-2304 prefix matching + a boot-time
hybrid-APC patch, drafter `dc77ff1c…` (08-28), 12.52 GB → 1.75M pool,
maxseq 8. Receipts: miken TEB 90, code accept 0.68–0.71. It was the original
miken reference row and the only lane that ran the base image with no build,
but it lost prefix hits under DFlash2 spec decoding (the drafter's SWA group
zeroed the hybrid min), which is exactly the problem the dflash2pmu default
was created to fix. Dropped to reduce the lane surface.

## Cleanup: drop the dflash2 lane (2026-09-11)

Branch `cleanup-glm53-drop-dflash2` (uncommitted). The legacy `dflash2` lane
was removed from `.env`, `docker-compose.yml`, and `README.md`; only this
`research.md` retains a reference. Specifics:

- **`.env`**: dropped the `LANE=dflash2` alternative block (and its
  `dc77ff1c…` drafter pin / 12.52 GB / maxseq-8 row).
- **`docker-compose.yml`**: removed the boot-time hybrid-APC patch
  (`patch_hybrid_prefix_hit.py`) mount + its `LANE=dflash2` gate, removed the
  `else` dflash2 spec-config branch (the one without
  `disable_eagle_block_drop`), and pointed the compose defaults at the
  dflash2pmu values (`LANE=:dflash2pmu`, `MAX_SEQS=6`,
  `DFLASH_REVISION=bf582e4e…`) so an unset var can't silently fall back to
  the legacy row.
- **`patches/patch_hybrid_prefix_hit.py`**: deleted — dead (only dflash2 used
  it). The remaining lanes bake their patch series into the image.
- **`README.md`**: profile table reduced to dflash2pmu + mtp3; removed the
  dflash2 switch block, the "legacy dflash2 profile" block-size note, the
  12.52 GB KV references, and the Legacy-APC reference.
- `prepare-model.sh` surgery and the `prepare-model.sh`-produced config are
  identical across the two remaining lanes.

## On-cluster verification (2026-09-11, this cluster)

Brought up from `main` on both nodes; both containers healthy on
`glm53-intel-dflash2-pmu128:20260908`:

- Boot markers: `Resolved architecture: DFlash2DraftModel`,
  `speculative_config=SpeculativeConfig(method='dflash', …
  num_speculative_tokens': 7 …)`, `Using Eagle3 auxiliary layers from
  config: (6,15,25,34,43)`, `Setting attention block size to 4608`,
  `GPU KV cache size: 1,867,536 tokens`, rejection-sampler warmup, Model
  loading 84.68 GiB / 302.6 s.
- Acceptance **0.28–0.32** overall, per-position deltas
  `[35,21,13,5,4,3,3]` — monotonic decay, consistent with florianbrede's
  0.34/0.37 (his were long concurrent benchmarks; these are short
  single-prompt reads, so the lower overall figure is expected).
- PMU128 replay: 171-token prompt →
  `usage.prompt_tokens_details.cached_tokens=128` on first and repeat (one
  128-block hit + 43-token residue — matches the lane's documented residue
  behavior). Sub-128-token prompts report `cached_tokens=0` — expected,
  they never fill a PMU unit.
- Response `usage` in this build: non-stream chat responses omit `usage`;
  stream with `"stream_options": {"include_usage": true}` to get
  `prompt_tokens_details.cached_tokens` / `created_cache_tokens`.
- This build's `/metrics` spec-decode counters:
  `vllm:spec_decode_num_draft_tokens_total`,
  `vllm:spec_decode_num_accepted_tokens_total`,
  `vllm:spec_decode_num_accepted_tokens_per_pos_total` (with
  `position="N"` label), plus `_created` variants.
- The model-name proxy flapped "unhealthy" while :8000 was down mid-switch
  (its healthcheck targets the backend) and recovered automatically once
  serving resumed.

## Current state (2026-09-11)

- **`dflash2pmu` is now the DEFAULT lane** (`.env` ships `LANE=dflash2pmu`
  + `IMAGE=glm53-intel-dflash2-pmu128:20260908`); on-cluster verification
  above. The README has been reorganized as a fresh-runner deploy doc:
  lane feature comparison + deploy steps up front, historical material
  (receipts, A/B history, update passes) lives here in `research.md`.
  The legacy `dflash2` lane was removed (see the cleanup note above);
  `mtp3` remains the opt-in alternative documented in the README's
  "Switching profiles".

- Model pinned `5eee1846…` (Intel HEAD, 2026-09-01 upload; re-verified
  2026-09-07 via HF API — UNCHANGED, no new revisions), image digest
  `4def0ef6…` (sm121-v11-dflash2, single-arm64 manifest, verified via
  `docker manifest inspect --verbose` 2026-09-06).
- **2026-09-08 update pass (forum+HF sweep):** Intel HEAD re-verified
  UNCHANGED (`5eee1846…`, lastModified 2026-09-01). DFlash2 drafter
  upstream still `bf582e4e` (files untouched since 08-31). Adopted
  florianbrede's third recipe `tp2_glm53flash_autoround_dflash2_k7_pmu128`
  (external DFlash2 k7 + PMU128 on the SAME base digest) as the opt-in
  `LANE=dflash2pmu` row: vendored verbatim into `dflash2-pmu128/`
  (SHA256SUMS-verified), compose gated (baked-patch lanes skip the boot-time
  hybrid-APC patch), drafter pin bumped to `bf582e4e` for that lane. Receipts
  TEB 90 (158/176), PP 1,679 tok/s, TG 33.1/54.7 @C1/C4, ~82-token recompute.
  Watch items from 381350 (not adopted, measured on other images/stacks):
  klement's k=4-over-k=7 DFlash2 A/B (mixed workloads), eugr fork
  `fix-tool-choice-enforcement` mod (TC-45; fork-specific, PR #380), rodman80
  dual-fabric NCCL (+~5% long prefill, both-CX7-NIC rails), rodman80
  chat-template refresh (tool-result-ID dedup — ours verified byte-identical
  already), voktolom post 434 (PMU128 + retention-0 on eugr b12x: warm TTFT
  seconds → ~0.2 s; retention flag behavior confirmed image-specific),
  sakra0616 post 444 (Anthropic-adapter `--chat-template` MUST be explicit
  for CC prefix caching — we already mount the vendored template, covered).
- **2026-09-07 update pass (forum+HF sweep):** thread 382041 has no posts
  after 09-04 (nothing to adopt). The actionable finding is
  florianbrede-ayet's `tp2_glm53flash_autoround_mtp3_pmu128` recipe (forum
  382632): same Intel quant, same digest-pinned base, native MTP3 +
  PMU128 + 13.5 GB KV pin (1.92M pool) + seqs 6, receipts tool-eval 91/100,
  108 tok/s @ C6, 82-token avg reprocessing. Vendored verbatim into
  `mtp3-pmu128/` and wired as the opt-in `LANE=mtp3` row (see README). Our
  indexer overlay was already byte-identical to tonyd2wild's latest
  (`8a3ecfb0…`) — the kpool patch chain moved to per-request tail-cache
  handling on 09-02 and we already had it.
- DFlash2 drafter: upstream main moved to `bf582e4e` (README/figure only;
  config/weights identical). Our pin `dc77ff1c` still resolves — no bump
  needed.
- Forum 381350 post 340 (robert287, 09-05): NVFP4 quants (except RedHat's)
  have a ModelOpt/vLLM issue; "EXL and AutoRound quants do not exhibit the
  same problem" — quality corroboration for this lane.

## Provenance: what @miken actually references (post 5)

The thread (382041) post 5 contains **no URLs** (verified against the cooked
HTML; only a bare `https://huggingface.co/Intel/GLM-5.3-Flash-W4A16-AutoRound`
appears in the earlier posts). @miken's result stack, decoded from his text:

1. **Model**: `Intel/GLM-5.3-Flash-W4A16-AutoRound` — AutoRound 0.15 INT4
   W4A16 (sym, g128), `packing_format: auto_round:auto_gptq`, MIT. 34 shards +
   `model_extra_conv.safetensors` + `model_extra_tensors.safetensors` (both
   referenced by `model.safetensors.index.json`; the MTP/extra weights live
   there — no separate `model_mtp.safetensors`).
2. **Image**: "Tony sm121-v11 lineage" = `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2`
   (alias `radixark/vllm-glm53-flash:sm121-v11-dflash2`; identical per
   rodman80). Build chain in tonyd2wild's repo: `Dockerfile.glm53-sm121-v1..v9`
   then `docker/dflash2-overlay/Dockerfile` (NOT v9 — the overlay is `FROM
   sm121-v8`) + 4 dflash2 patches. vLLM at
   `/usr/local/lib/python3.12/dist-packages/vllm`
   (`0.1.dev20051+g487ecf187` per miken's tool-eval receipt).
3. **Patches**: "MiaAI decode-floor/APC/indexer patches" — the APC piece is
   `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks`'s hybrid-prefix-cache
   coordinator fix (`overlay/patch_hybrid_prefix_hit.py`), ported with anchors
   verified against **this** image by rodman80 (`docs/patch_hybrid_prefix_hit.py`
   in his repo); the indexer piece is tonyd2wild's SM121 kpool overlay
   (`docker/sparse_attn_indexer_kpool_sm121.py` = our
   `patches/sparse_attn_indexer_kpool.py`, byte-identical bar the header);
   `GLM53_INDEXER_WORKSPACE=rightsize` is MiaAI's env.
4. **A/B numbers**: post-5 table + `brandonmusic`'s KLD panel (EXL3 lane) and
   LIL `378ca545` NVFP4 (0rand lane). The full harness that reproduces the A/B
   protocol is **rodman80/glm-5.3-flash-w4a16-2x-DGX-Sparks** (created
   2026-09-04, pushed 2026-09-06; `benchmarks/{RESULTS,COMPARISON}.md` +
   `bench/`). It serves the **sibling** `canada-quant/glm-5.3-w4a16-mtp`
   quant (compressed-tensors W4A16 + BF16 MTP head), not the Intel one — same
   image, same patches, same flags lane. Nothing public exists that boots the
   **Intel** quant other than miken's post itself; this recipe is that port.
5. **The qzeros-sym fix** (eugr, `fix-autogptq-sym-qzeros` mod; forum 381350
   post 390, 2026-09-06) was published AFTER miken's run and is for **newer**
   vLLM (upstream Dockerfile step 2026-09-05). The pinned image threads
   `may_have_zp=not is_sym` (auto_gptq.py@487ecf187) — see "Surgery mechanics
   notes" below. Keep as an image-bump gate.

## Surgery mechanics notes (verified against sources)

- **Why it works**: vLLM `AutoGPTQConfig.from_config` reads
  `dynamic` (auto_gptq.py@487ecf187 lines ~196-213);
  `get_dynamic_override` (quantization/utils/gptq_utils.py) treats a `-:`
  prefixed key as a regex exclusion — `re.match("<pattern>", layer_name)`
  returns False for that module, leaving it on the unquantized (`BF16`) path.
- **Rule source**: `Intel/GLM-5.3-Flash-W4A16-AutoRound` `config.json`
  `quantization_config.extra_config` = **679** rules (miken said "≈680"; the
  standalone `quantization_config.json` duplicate holds 666 — a stale subset,
  the surgery reads config.json), all `{bits:16, data_type:float}`; shapes
  include bare module keys
  (`model.visual.blocks.0.mlp.gate_proj`), per-layer keys
  (`model.language_model.layers.N.self_attn.q_proj`), and a few glob keys
  (`.*model\.language_model\.layers\.N\.mlp\.gate.*`). We transform each
  verbatim: `"-:" + key`.
- **Checkpoint layout**: 37,152 `qweight` AND 37,152 `qzeros` int32 tensors
  (e.g. `model.language_model.layers.3.mlp.experts.0.gate_proj.qweight`),
  46 layers (0–45). Only routed-expert GEMMs are INT4 (36,288 of the 37,152
  match the canada-quant card's count; difference = extra conv/tensors).
- **qzeros sentinel**: auto-round sym writes unused qzeros filled with
  `0x77777777`. On THIS image the MoE method passes `may_have_zp=not is_sym`
  to backend selection and `w1_zp=... if use_zp else None` to the kernel config
  (auto_gptq.py `get_fused_moe_quant_config`), so marlin never consumes them —
  consistent with miken's clean-quality receipts. Do NOT add the eugr mod
  unless the image is bumped (then it is required before anything else).
- **`model_extra_conv.safetensors` / `model_extra_tensors.safetensors`** are in
  the index `weight_map` (36 files total) — the surgery hardlinks them; do not
  prune.
- **Template**: Intel ships a complete `chat_template.jinja` (10,644 B) with
  image/video/audio macros, `reasoning_effort` (low|high, else → max) and
  `clear_thinking` — but it has **no `enable_thinking` branch** (thinking is
  structurally always-on). We vendor rodman80's `chat_template_mm.jinja`
  (11,213 B, validated on this image with the same base model) and pass
  `--chat-template` + `--default-chat-template-kwargs {"enable_thinking":
  true, "reasoning_effort": "high"}` (explicit pins; `THINKING`/`REASONING_EFFORT`
  env). Default-path output is identical to the Intel template (both emit the
  full think block); rodman80's additionally honors `THINKING=false` (empty
  think block) and carries tool-sort hardening. miken's receipts ran the
  Intel-shipped template — the default path is unchanged.

## What differs between @miken and rodman80 (open A/B items)

Both were measured on the SAME image+model-class, different configs. Do not
mix:

| knob | miken | rodman80 (validated) |
|---|---|---|
| `--kv-cache-memory` | 12,520,000,000 (1.75M pool) | 9,663,676,416 (1.34M pool) |
| `--max-num-seqs` | 8 | 6 |
| CUDA graphs | on (`graphs 1..64`) | `--enforce-eager` (neutral/worse) |
| `--max-num-batched-tokens` | not stated | 8192 (A/B over 16384) |
| clock lock | not stated | 2400 MHz (+5–7% decode) |
| acceptance | 0.68–0.71 (code, per-position?) | 0.418 overall / 0.75–0.78 per-position |

Watchers: (1) graphs-vs-eager A/B with restart-between (per repo benchmark
discipline); (2) KV pin 12.52 vs 9 GiB — both validated, the difference is
pool (1.75M vs 1.34M); (3) MNBT: only rodman80's data, keep 8192; (4) does
`GLM53_INDEXER_WORKSPACE` actually get consumed? (grep the image at boot —
`grep -r GLM53_INDEXER_WORKSPACE /usr/local/lib/python3.12/dist-packages/vllm`).

## Watchlist

- **florianbrede-ayet/spark-recipes (mtp3-pmu128 lane)**: the vendored lane's
  upstream — watch for new patches to the #53388/#53906 series, PMU changes,
  and fresh A/B numbers vs DFlash2. The A/B on THIS cluster (dflash2pmu k=7
  vs mtp3+PMU128, both on the same image/quant) is the standing open question;
  florianbrede claims MTP3 wins reasoning-heavy workloads (forum 382632).
- **vLLM #53388 / #53906 upstreaming**: both patches in `mtp3-pmu128/patches/`
  are exact upstream hunks against `487ecf187`; when the image's vLLM picks
  them up natively the vendored series becomes a no-op — verify with
  `apply_runtime_patches.py --verify-only` before any image bump.
- **tonyd2wild / GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark**: new image tags
  (`sm121-v12`+ / `-hybrid-*`), any Dockerfile change to the dflash2 overlay or
  patch_v7/v8 — every image bump re-runs the "new vLLM?" gate (qzeros mod,
  flag renames, vLLM path).
- **Intel/GLM-5.3-Flash-W4A16-AutoRound**: model updates (new `sha`, new
  shard layout, template changes) → re-pin `MODEL_REVISION`, re-run
  `prepare-model.sh` on both nodes.
- **canada-quant/glm-5.3-w4a16-mtp**: the sibling quant's card is the most
  current A/B source; its README tracks vLLM-main GLM support
  (vllm-project/vllm#53906 merged 2026-09-03) — if vLLM-main ever exceeds the
  pinned image's numbers, revisit the image pin (with the qzeros gate).
- **MiaAI-Lab EXL3 repo**: PR77 E2 kernels lifted EXL3 prefill +20% (Sep 1) —
  the EXL3 rows in every comparison table get stale; re-check before citing
  miken's table as current.
- **Forum 381350** (main GLM thread): post 390 eugr qzeros reference, post 273
  vision+text crash, posts 333/334 "1 tok/s stutter" (seen on NVFP4 + Intel
  W4A16 — informational), and any new W4A16 reports.
- **Driver**: 580.173.02 current; 610.43.02 unstable/bigger UMA (repo-wide).

## Revision-pin procedure (for any bump)

1. Image: `docker manifest inspect --verbose ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2`
   → replace digest in `.env`; then check the four serve-arg knobs still exist
   (new image may retire a flag) and the qzeros-sym gate (new vLLM → add
   eugr's mod to the boot sequence).
2. Model: new Intel `sha` → `.env` `MODEL_REVISION` + `hf download` + re-run
   `prepare-model.sh` on both nodes; diff the new `extra_config` count (679
   today) and the template.
3. Drafter: only if a new `incoai/GLM-5.3-Flash-DFlash2` revision is blessed
   upstream; the dflash2 overlay is sensitive (acceptance changes). The
   dflash2pmu lane pins `bf582e4e…` (HF HEAD, 08-31). The dropped dflash2
   lane's `dc77ff1c…` pin is retained here for reference only.
4. `GLM53_INDEXER_WORKSPACE`/APC anchors: any vLLM-file drift breaks the
   fail-closed boot — by design; re-derive or retire then.

## 2026-09-17 — production-night troubleshoot: "crawling" agent traffic (mtp3 lane, live container)

**Deployment (live, not repo defaults):** `glm53-intel-w4a16` =
`glm53-intel-mtp3-pmu128:20260907`, `LANE=mtp3`, `PMU=1`, started
2026-09-17 02:19 UTC — first production night on the mtp3 lane.
Report: agents capped at 4 concurrent, "performance crawling".

**Method:** troubleshoot-slowness protocol (live env → metrics → engine
timeline → host/GPU → spec-decoder). All timestamps UTC.

**Evidence:**
- Serial giant prefills: pp 5.2K→10.5K tok/s in single 10-s windows every
  ~30–60 s, tg≈0–3 during them. Avg prompt = 3.74M/55 req ≈ 68K tok, avg
  gen ≈ 200 tok — agent-style full-history turns.
- Decode healthy when it runs: tg 20–35 tok/s at run=1 (02:49–02:52
  sustained burst); MTP3 mean acceptance 2.4–3.9 (normal variation).
- run ≤ 2, waiting=0 the whole hour → the client cap of 4 never bound;
  traffic was effectively serial.
- Prefix-cache cumulative hit rate collapsed 25.1% (02:36) → 3.5%
  (03:22); live delta over 4 min = 532,273 queries / 13,824 hits = **2.6%**.
- PMU128 retention proven live (same 655-tok prompt twice): shot 1
  `cached_tokens=0` (16.7 s wall — queued behind an in-flight prefill),
  shot 2 `cached_tokens=640/655` (128-token precision, 15-token residue).
  **The lane does NOT lose prefix hits** — the dflash2-on-base-image
  failure mode does not apply to native MTP3.
- Host/GPU exonerated: P0, empty throttle counters on both nodes; head
  118/121 GiB + 3 GiB swap allocated (calm si/so), worker 116/121 (calm);
  JIT lines one-off (02:35 warmup, 03:21 new run=2 shape).

**Root cause:** workload, not hardware/config — prefix-unique prompts
(fresh subagent-style conversations per request). Every turn cold-prefills
~68K tok ≈ 7 s TTFT at the ~9.5K tok/s W4A16-marlin prefill ceiling, and
decode starves during those windows (MNBT 8192 chunked prefill). KV pool
never above 7.8% → no eviction pressure; the low hit rate is prompt
uniqueness, not cache loss.

**Cap arithmetic:** min(MAX_SEQS=6, 1.92M ÷ 68K ≈ 28 sessions, MTP3
efficient batch ≈ 6) = 6 — the client cap of 4 sits below cap; the cap was
never the lever.

**Fix applied:** none server-side (nothing broken). Client-side
recommendation delivered: append-only conversation history + stable system
prefix so PMU128 replays history at 128-token precision — turns become
decode-bound at 25–35 tok/s.

**Watch:** headroom 3–5 GiB/node at GMU 0.85 + 13.5 GB KV pin (DGX OS
7.5.0 OTA boots with ~7.2 GiB less RAM) — do not raise GMU/KV pin without
re-measuring host headroom through a ~950K prefill. The mtp3-vs-dflash2 A/B
watcher above gains a production data point: PMU128 cache mechanics and
acceptance validated live.

## 2026-09-18 — async A/B × 2 lanes + compose ASYNC-wiring bugfix; final = mtp3 @ ASYNC=0

User-requested one-knob async A/B on BOTH intel lanes (seqs 6 untouched),
same bench as the 0rand/vision A/Bs (warm-up req; c1 tg1024 ×2; c4 = 4×
tg512 ×2; stream:false, real completion-token counts; cold-cache boots,
drop_caches ritual; identical prompts across all boots).

**BUGFIX (blocking find): the compose `environment:` block never mapped
`ASYNC` into the container** — `.env`'s `ASYNC=1` was inert in EVERY
prior intel deployment (only the runtime check `[ "${ASYNC:-0}" ... ]`
exists at the command block; with no env var it always evaluated 0). The
dflash2pmu "ON" side below is the FIRST boot ever to run
`--async-scheduling` on this recipe. Fix: `ASYNC: "${ASYNC:-0}"` added to
the environment block (verified in argv + printenv). The OFF side was
benched pre-fix (no flag — runtime-identical to ASYNC=0 post-fix).

| lane | shape | async OFF | async ON |
|---|---|---|---|
| dflash2pmu (k=7) | c1 full-1024 | 23.0 / 17.4 tok/s | 32.4 / 20.9 tok/s |
| dflash2pmu (k=7) | c4 aggregate | 44.4 / 39.8 tok/s | 43.5 / 45.6 tok/s |
| mtp3 (native MTP) | c1 full-1024 | 21.9 / 22.3 tok/s | 24.8 / 21.7 tok/s |
| mtp3 (native MTP) | c4 aggregate | 46.4 / 52.2 tok/s | 46.1 / 48.3 tok/s |

KV pools: dflash2pmu 1,867,536 both sides; mtp3 1,920,956 both sides
(pins unchanged; pools match the documented rows). All boots healthy,
zero errors.

**Verdicts:**
- Async: **no significant delta on either lane** (c4 overlaps within the
  ±4-7 tok/s round spread; c1 is acceptance-dominated — spread 17-32
  tok/s across samples of the SAME config). Consistent with the 09-17
  vision-0rand and 0rand A/Bs: async is a no-op knob on this stack. The
  knob is now at least WIRED for future re-tests.
- Lane: **mtp3 ≥ dflash2pmu at c4 in every round** (46.1-52.2 vs
  39.8-45.6 agg, ~+8-17%) with a slightly larger KV pool; c1 too noisy
  to rank. Matches last night's production choice of mtp3.

**Final serving state (left up): mtp3 lane @ ASYNC=0** (winning lane at
its marginally-better setting; the currently-running boot IS the final
config — no extra recreate). Branch also re-points proxy
`BACKEND_MODEL=glm-5.3-flash` (both GLM recipes share the served name;
proxy was still targeting deepseek-v4-flash).

## 2026-09-18 — mtp3 made the DEFAULT lane (repo defaults/docs flip)

Follow-up to the async A/B above (final serving state = mtp3 @ ASYNC=0):
the recipe's shipped default lane is now mtp3, matching what production
has been running since 09-17. dflash2pmu becomes the documented
alternative. Uncommitted edits at flip time. Files touched:

- **`docker-compose.yml`**: `LANE` compose default `dflash2pmu` → `mtp3`;
  `PMU` default `0` → `1` (both shipped lanes are PMU128 lanes, so 1 is
  correct for either); drafter-resolution and spec-config conditionals
  re-pointed at the new default (`[ "${LANE:-mtp3}" = ... ]`); lane
  comments + launch echo updated. No flag/logic change — with the current
  `.env` (LANE=mtp3, PMU=1) the resolved argv is byte-identical to the
  validated 09-17/09-18 production boots.
- **`.env`**: IMAGE/LANE/PMU comment blocks re-labeled (mtp3 default,
  dflash2pmu alternative); `LANE=mtp3`, `IMAGE=glm53-intel-mtp3-pmu128:
  20260907`, `PMU=1` were already the live rows from the 09-18 final
  state — no value changes. Fixed a truncated DFLASH_* comment in the
  process.
- **`README.md`**: default/alternative labels swapped everywhere (top
  bullet, profile table, why-default paragraph now cites the 09-18 A/B
  numbers, build step builds the mtp3 image with dflash2 commented as
  alternative, drafter download marked dflash2pmu-only, mtp3 boot marker
  `method='mtp'` added + DFlash2 marker relabeled, KV-pool line split per
  lane, PMU check relabeled "both lanes", switch block now targets
  dflash2pmu, references section labels).
- **`mtp3-pmu128/README.md`**: the "kept as alternative until A/B'd here"
  note replaced with the A/B outcome.

No serving-profile values changed; this is a defaults/docs flip only. The
"Lane receipts" table above keeps its historical (dflash2pmu-default-era)
header for the record — the current default is mtp3.
