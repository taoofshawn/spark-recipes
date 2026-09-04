# research.md — glm-v53-flash-miaai: maintenance notes

Working notes for future update/maintenance sessions on this recipe. The
README is the deploy doc; this file is the memory. Anything not needed to
maintain or re-derive the recipe was pruned — git history retains the full
adoption narrative (see `git log glm-v53-flash-miaai/`).

## Current state (2026-09-04)

- Deployed and validated on the 2-node cluster (spark-0f0b / spark-6d14).
  Serves as `glm-5.3-flash` on :4000, 1M ctx, DFlash2 k=7. Boot-log health
  markers and what they mean: see README "Boot-log health markers".
- Single-stream ~24-28 tok/s decode, 4-way aggregate ~59 tok/s (upstream
  documents 33-74 on their kit). Prefill ~600-750 tok/s.
- **2026-09-04 research pass applied:** threshold 1024→3584 (PR#112),
  mixed-prefill gate skip→2048 (#119), EXL3_FAT_KERNEL=1→0 (no-op on this
  image), image digest-pinned, kpool-tail OOB patch vendored (new hotfix,
  see below). Image, weight and drafter pins still byte-valid.
- The DS4-vision recipe (`deepseek-v4-flash-vision-miaai`) shares this
  cluster and port; only one may serve at a time.

## Environment facts (do not rediscover these)

- Each DGX Spark exposes ONE GPU; TP=2 always spans the two nodes. A
  single-node TP=2 diagnostic boot is impossible (`local_rank <
  device_count` assert).
- Sparks have no `hf` CLI, no host `huggingface_hub`, pip is PEP-668-locked:
  `python3 -m venv /tmp/hfvenv && /tmp/hfvenv/bin/pip install huggingface_hub`
  for any cache work.
- The GLM image bakes `HF_HOME=/root/.cache/huggingface` — the host cache
  mounts there (the DS4/Anemll image uses `/cache/huggingface` instead).
- `docker-compose.override.yml` is AUTO-LOADED when no `-f` is given. If a
  boot behaves strangely, check for a leftover override file on BOTH nodes.
- Worker first, leader ~35 s later. During the leader's ~6 min weight load
  the worker spams `TCPStore.cpp sendBytes failed` / NCCL heartbeat warnings
  — wait-window noise. Judge boot health by the LEADER log.
- GPU contention: `docker compose down` the other recipe on BOTH nodes
  before starting one (a surviving old leader container causes GPU
  allocation crash-loops).
- Worker-local debugging trick: `docker exec glm53-exl3-miaai ps -eo` —
  `VLLM::EngineCore` + `VLLM::Worker_TP0` alive = healthy; a couple of
  defunct `python3` zombies (ppid 1) are cosmetic either way.
- **Image path inside the container: vLLM lives at
  `/usr/local/lib/python3.12/dist-packages`** (NOT /opt/venv, NOT /opt/env —
  this is the MiaAI/Entrpi preview lineage, not the DSv4 vLLM lines). Every
  boot patch and every future anchor must use this path.

## The KV hotfix — `hotfix_kv_check_glm5.py`

The GHCR image's baked vLLM (`vllm/v1/core/kv_cache_utils.py`) predates the
overlay's DFlash2 KV work. Three defects, all inside the image, none
reachable from compose flags:

1. **Stale builder branch**: `_get_kv_cache_groups_glm5_next`'s else branch
   is STANDALONE — leaves the drafter at block_size=16 / window 16384, so a
   1M request needs ~1306 block ids vs a ~621-block pool. Symptom: boot
   ValueError (34.15 GiB) or silent capacity starvation
   (`waiting{reason="capacity"}` forever, GPU ~0%, small requests fine).
2. **Detector/builder mismatch**: `_glm5_next_tensor_layout` rejects
   `page_size_padded` specs the builder emits.
3. **No glm5 branch in the capacity check**
   (`_max_memory_usage_bytes_from_groups`) — the stock "GPU KV cache size"
   line is window-unaware and not an honest capacity figure (upstream
   issue #94).

The hotfix edits `kv_cache_utils.py` in place at boot (patch loop in
docker-compose.yml, before `vllm serve`):

- **A**: detector accepts drafter specs with `page_size_padded == mla_page`.
- **B**: early glm5 capacity branch — allocator-consistent per-block math,
  window-bounded cdiv, mamba groups excluded (length-independent SSM state).
- **E** (load-bearing): replaces the STANDALONE builder branch with the
  overlay's padded slot-share rescale (block=64, page padded to mla_page →
  313 blocks/req → 1M schedulable).
- **D**: one-line debug dump when the detector still returns None.

Fail-closed by design: it preflights every anchor and aborts the boot
(nonzero exit) if the image text no longer matches. That failure is the
signal the image changed — do not serve until the hotfix is re-derived or
retired.

Why upstream doesn't hit this: their validated boots use locally built
images with builder/detector/accounting in sync (their issue #97 documents
GHCR-vs-rebuild drift).

## The K-pool tail hotfix — `patch_kpool_tail_slotmap.py`

Added 2026-09-04 (vendored from
`vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe` docs/KPOOL_TAIL_BUG.md,
MIT). A **silent correctness bug** in the image's vLLM, verified present in
our exact image (extracted + dry-run patched 09-04):

- GLM-5.3 is hybrid (KDA + sparse MLA); its attention metadata is built by
  `vllm/v1/worker/gpu/model_states/mamba_hybrid.py`, which calls
  `build_attn_metadata(...)` **without `positions=`** (default None) — the
  plain-transformer path (`default.py`) passes it.
- `KpoolTailSpec` declares a ONE-block circular scratch cache per request,
  but with `positions=None` the indexer skips
  `compute_kpool_tail_slot_mapping` (its guard is `if positions is not
  None`) and the tail group uses the generic paged mapping, which indexes
  a one-entry block-table row by `pos // block_size` → OOB:
  `_kpool_tail_seed_kernel` / `_kpool_decode_update_batched_kernel` never
  bounds-check. Manifestation is pool-geometry-dependent: intermittent
  garbage destination blocks, or silent corruption of a neighbour layer's
  sparse-attention index; long generations trigger it most reliably.
- Fix (two halves): pass `positions=input_batch.positions` in the hybrid
  path; and make `compute_kpool_tail_slot_mapping` write the persistent
  buffer IN PLACE (the old `.clone()` is captured by CUDA graphs at a
  transient address → illegal memory access on replay).
- Upstream measured: 48 overruns → 0 (ctx 8192); 19,575 decode-path tail
  updates 0 OOB under `--enforce-eager`; 57,551-update soak 0 OOB.
- Boot marker: grep `positions=input_batch.positions` → "mamba_hybrid.py
  now passes positions=input_batch.positions". Fail-closed like the KV
  hotfix (compose loop runs it with `|| exit 1`; anchors checked).

## Known-benign boot/log lines (all verified on this cluster)

| line | verdict |
|---|---|
| `SymmMemCommunicator: Device capability 12.1 not supported` | expected on GB10; PYNCCL fallback |
| `Custom collectives are disabled because this multi-node ...` | expected for TP across 2 nodes |
| `Sparse MLA impl has no dense-MHA prefill path; using the top-k MQA path only` | by design; only SM12x kernel |
| `Draft model DFlash2Qwen3ForCausalLM does not support external multimodal embeddings` | by design; drafter is text-only |
| `Disabling fine-grained prefix-cache hits ... KpoolTailManager` | upstream default; 3584-token block-aligned hits still work |
| sampling defaults overridden by `generation_config.json` (temp 1.0, top_p 0.95) | checkpoint ships tuned defaults; clients may override |
| one-time Triton JIT `_topk_topp_kernel` latency spike | first sampled request only; cached after |
| worker TCPStore/NCCL spam during leader load | rendezvous noise; judge by leader log |
| defunct `python3` zombies (ppid 1) | cosmetic; verify workers alive via `ps -eo` |

## Revision-pin procedure (use for any model/drafter bump)

Pins live in `.env` (`MODEL_REVISION`, `DFLASH_REVISION`) — both nodes'
caches must hold exactly those revisions (`HF_HUB_OFFLINE=1` means vLLM
uses whatever snapshot dir resolves first).

1. Diff old vs new revision trees via the HF API
   (`https://huggingface.co/api/models/<id>/tree/<rev>`; compare LFS oids).
   The DFlash2 repo has a pattern of same-day "Checkpoint update" commits
   with byte-identical serving files — often only the README differs and
   re-pinning is a no-op.
2. If weights differ: proper `snapshot_download` in the /tmp venv on BOTH
   nodes (a `cp -a` of an existing snapshot only works for byte-identical
   trees; symlinks to ../../blobs must survive — verify none broken).
3. If the tree matches but the snapshot dir for the pinned rev doesn't
   exist, materialize it by `cp -a` of the cached snapshot, then
   rglob-verify no broken symlinks.

## Upstream watchlist (research these on every update pass)

**MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks** (source of image + profile;
HEAD eb0469fb 2026-09-02 — FROZEN since; all forward progress is in open
PRs/issues; owner paused new-PR merges per #104. Reviewed 2026-09-04):

**DO NOT miss (the 3 same-image, same-geometry findings — all adopted):**

- **#110 + PR#112 refined (09-03, danielbanu, measured on EXACTLY our
  geometry: TP=2, MAX_NUM_SEQS=4, MNBT=7168, same `:exl3` image)** — the
  PR's fixed **1024 was over-taxing MNBT=7168 kits** (−44% contended
  prefill: 182.3s vs 102.9s baseline). **`threshold = MNBT/2 = 3584` is the
  sweet spot**: freeze 150s→11s (14×) AND prefill 95.2s (best, 1046 tok/s).
  ADOPTED 09-04 (supersedes the 1024 from 552e88a). Table (A = warm-poll
  freeze of other sessions, B = cold 100k prefill): unset 150.2/102.9s |
  1024: 6.9/182.3s | MNBT 2048 unset: 145.1/98.5s | **3584: 10.7/95.2s** |
  MNBT 4096 + 2048: 4.5/104.0s. Shrinking MNBT is NOT an alternative
  (starvation is scheduling-rights, not chunk size).
- **#119 (zankie 09-03, same image)** — `GLM53_MIXED_PREFILL_CHUNK=skip`
  (our old default) made a 9.5k newcomer mid-decode wait the WHOLE running
  decode → 67-71s TTFT; at `2048` → 10.1s (running stream dips to
  ~1.9 tok/s). Soak at gate=2048 also delivered the #106 NEGATIVE
  (no freeze in 75 min, 175 req, 0 errors). ADOPTED 09-04 (was `skip`).
- **#122 (09-03)** — each request holds ~90k pool tokens BEFORE its prompt
  (3584 hybrid pages + DFlash slot-share), so `MAX_NUM_SEQS=4` never fits a
  ~356k pool. THEIR pool is 356k @262k ctx; OURS is ~1.05M @1M ctx → the
  90k fixed cost is ~7% of our pool vs ~25% of theirs. **Our 4×100k
  session load fits: 4×(90k+100k) = 760k < 1.05M pool.** Do NOT blind-drop
  MAX_NUM_SEQS to 3; watch `vllm:num_preemptions_total` on the leader
  (`curl localhost:4000/metrics | grep num_preemptions`) instead — only
  drop if preemptions show up under sustained 4-stream load.

- **#110 + PR #112** (`LONG_PREFILL_TOKEN_THRESHOLD`) — see above;
  superseded by the 09-03 refinement. Our EXTRA_ARGS now carries 3584 and
  the launch arg plumbing stays in place for future re-tuning.
- **#108** — the `padded slot-share ... exact-fit page mismatch
  draft_bytes/token=2048` line we treat as a HEALTH MARKER is upstream
  issue: ~5.2% structural padding per co-owned page, reproduced every
  boot. Cosmetic-but-real; if upstream lands a rescale, re-check our
  marker text.
- **#106** — prefix cache stops hitting GLOBALLY after ~50-60 min of
  concurrent agentic load. 09-03 counter-data: a 2-node 75-min soak at
  `GLM53_MIXED_PREFILL_CHUNK=2048` did NOT freeze (175 req, 0 errors,
  prefix hits advanced every window) — that gate may itself prevent the
  stall; we now ship it. Watch `vllm:prefix_cache_hits_total` for a frozen
  counter under load (symptom: full re-prefills on byte-identical prompts
  at low KV usage). Reederey87 also attributes zero-hits to chunk ends
  missing the page boundary (they align MNBT=page-size 3584 + 64-token
  fine-grained hash) — their MNBT=3584 == our hybrid page size, and our
  MNBT=7168 is exactly 2× it, so chunk ends DO align; keep MNBT.
- **#113** — 4-Spark DCP=4 data: on LONG agentic contexts k=3 beat k=7
  (81.4 vs 64.0 aggregate x4, accept 50% vs 27%); k=7 remains right for
  structured/code output (high-accept regime). **09-04 cross-lane
  corroboration (still NOT our geometry):** punkjazz-labs (TP4, same
  weights rev 25a44fd) k=7→3: accept 30%→53%, decode 27→31-33 tok/s prose;
  jnardiello production k=3; gitcommit90 default 7→5 (prose/code peak at
  k=5, k=7 best only for pure structured); Reederey87 k=8: prose −9%.
  A/B QUEUE: `DFLASH_TOKENS=3` on our kit for the multi-session profile
  (one variable at a time; k affects structured/code output — our serving
  target is agent traffic, which is exactly the mixed regime k3 wins).
- **#111** — changing reasoning_effort still invalidates the whole prefix
  cache (effort line sits at ~token 8). Reederey87's fix = emit the
  `Reasoning Effort` line UNCONDITIONALLY in the chat template so the
  off-shape is a strict extension of the on-shape (50k prompt: 56.8s →
  0.26s toggle). **We diffed their template vs ours: ONE comment-line
  difference only — our vendored `files/chat_template.jinja` ALREADY emits
  it unconditionally** (effort line is not wrapped in the thinking check),
  so our template already has this property. No change needed; verify
  prefix hits still advance if effort is toggled.
- **#97** — GHCR vs local-rebuild drift. CONFIRMED and now ACTIONED:
  public `:exl3` (2026-08-28) lacks PR#77's `exl3_fat_gemm`; two community
  digests floated for `:exl3` (9bb1557a vs 0f4798ac — the latter is what
  issues #106/#119/#122 cite, but BOTH our nodes and the GHCR package page
  resolve 9bb1557a). Image now **digest-pinned** in .env; re-verify before
  any bump. Rebuilt images are NOT the retirement path — #121 shows
  locally-rebuilt main images corrupt long-form generation; stay on GHCR.
- **#94** — honest KV-capacity boot line. If merged into the image, it
  supersedes our fix B.
- **#102** — `EXL3_FAT_KERNEL=1` + MNBT=7168 head-rank silent death at
  multimodal warmup (did NOT reproduce here — and PR#77 confirms the flag
  is a NO-OP on the public image; we now ship `EXL3_FAT_KERNEL=0` +
  comment). If a future boot dies without a traceback right after
  CUDA-graph capture: set `EXL3_FAT_KERNEL=0` and `MAX_NUM_BATCHED_TOKENS=2048`.
- **#86** — `GLM53_INDEXER_WORKSPACE=rightsize` reclaims ~4.5 GiB for KV
  (+26-28% capacity). We ship `stock` (do NOT flip — see next).
- **⚠️ #86 REVERSAL (09-04, punkjazz-labs bisect)** — the PR86
  indexer-workspace `rightsize` patch is now IMPLICATED as the long-prefill
  stall cause on their 4-Spark EXL3 kit (they bisected the stall to the
  rightsize patch and adopt c190db1 WITHOUT it; also confirmed the stall
  without a drafter). We ship `stock`, which is the safe side. Keep
  `stock`; do NOT flip to rightsize "for KV headroom".
- **#88 / #85 / PR #100 / #99 (KV disk tier)** — read before touching
  SPEC_METHOD / max_num_seqs / KV-tier knobs. Still open, no new data.
- **NEW 09-03 issues/PRs**: #114 (1M+vision boot init nondeterministic
  11min→never — we run vision+1M; treat as operational risk, not adoption),
  #115 (docs-only TP4 `.env.tp4.example`: GMU 0.75/seqs 8/k3/MNBT 2048 —
  all measured TP=4, does NOT transfer; also documents a TP4-only DFlash2
  draft+96k-chunked-prefill full-rank hang), #116 (EXTRA_DOCKER_ARGS hatch),
  #117 (bench_decode.py 401 w/ API key — tooling bug), #118 (shared-head
  UMA accounting omits ~5.6 GiB — only if our head is shared; it's not),
  #120 (Reederey87 fork claim — see below), #121 (rebuilt image corrupts
  long-form gen; GHCR clean — validates our no-rebuild), #122 (above).

**Entrpi** (https://github.com/Entrpi — parallel DGX-Spark serving line;
v2.3-tier1 as of 2026-09-02):

- **`vllm-glm-5.3-flash-spark`** — HEAD f223ff9 (09-02):
  `VLLM_B12X_MXFP8_MAX_M=0` disables the FlashInfer fallback in the apply
  path; per-M dispatch (a3c9ab3) already tracked. Still Entrpi's OWN image
  lane — informational, not drop-in.
- **`glm-5.3-flash-exl3-2x-spark`** — v2.3-tier1; registry digest now
  published in ANNOUNCEMENT/BUILD (their own image). FINDINGS.md §18-19:
  per-M dispatch/rowwise-fp8 draft head/NCCL 8 channels. §19 measured OUR
  drafter pin bf582e4e at accept 2.53 vs lab MXFP8 2.48 / first-pub 2.47
  — parity; our drafter pin is fine. Read their FINDINGS §8 ("what does
  NOT help") BEFORE spending time on speculative tuning here.
- **`dgx-spark-serving-mode`** — unchanged since 06-29; ~10-15 GB UMA
  headroom, host-level systemd toggle. Only relevant if we need more KV.
- **`ds4` / `ds4-on-spark`** — DeepSeek/Metal lanes; out of scope.

**Community/parallel lanes (09-04 sweep). Most are different topology or
different image — treat native-kernel numbers as non-transferable:**

- **`Reederey87/glm53-flash-exl3-2x-dgx-spark`** (2-node TP2 EXL3 4bpw,
  digest-pinned base = same day-0 vLLM preview image we run!) — the
  highest-signal sibling. HEAD 09-02: `LONG_PREFILL_TOKEN_THRESHOLD=1792`
  (= their MNBT/2: MNBT=3584 page→1792; consistent with our 3584 rule),
  unconditional Reasoning-Effort line (we already have it), MNBT=page-size
  alignment + fine-grained 64-token APC hash + per-group retention
  (fork-level patches — NOT vendorable to our public image), drafter
  pinned to 7d74cdd (an EARLIER DFlash2 rev; we pin the latest
  bf582e4e — keep ours, Entrpi measured it parity).
- **`vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe`** v0.3.1 (09-03):
  **`docs/KPOOL_TAIL_BUG.md` — ADOPTED as patch_kpool_tail_slotmap.py**
  (correctness fix, same base-image lineage, see hotfix section above).
  Also validates `--long-prefill-token-threshold 1024` on K2/TP1 — different
  quant/topology; the fixed VALUE doesn't transfer, the fix does. Their
  native EXL3 kernels + 1.91M-token KV pool are TP1/K2-2bit specific.
- **`punkjazz-labs/glm-5.3-flash-exl3-4x-dgx-spark`** (TP4, same weights
  rev 25a44fd as ours): GMU 0.75 (0.87-0.88 left <2 GiB host mem → NVRM
  NV_ERR_NO_MEMORY engine deaths on THEIR 4-node box; we're TP2 and
  0.8848 is image-tuned — do NOT copy the number), k7→3 (see #113),
  MNBT 7168→2048 (+16-18% fat-kernel prefill at 64k, coding first-token
  11-22s→2.9s — but fat kernel is TP4-rebuilt; our MNBT stays 7168 per
  the #112 table), MAX_NUM_SEQS 4→8 (+40% agg, −10% KV — only with a
  bigger pool; ours is the binding constraint), and the #86-rightsize
  stall bisect above. Authoritative two-node cross-check only.
- **`gitcommit90/glm-5.3-one-spark`** (TP1 2.05bpw): K sweep prose/code
  peak at K=5, default 7→5. Confirms k-trend; TP1 numbers don't transfer.
- **`jnardiello/tp4-glm53-fp8-gx10`**: k=3 production; k=5 +53% under 4
  streams, prose −10-19% (not promoted). FP8/TP4 lane.
- **`Plaaasma/glm53-flash-dual-dgx-spark`** (forum 382120): NVFP4 KV lane
  (288B/token → 2.2M pool) — different KV dtype, needs custom Triton.
  NOT adoptable to our fp8_ds_mla image. But: `EXL3_TEMP_ROWS_FUSED=192`
  mixed-step starvation fix (830→480ms, agg 10-13→24 tok/s) — env knob,
  **UNKNOWN whether our image exposes it; worth a boot-log grep**.
  GB10 unified-memory livelock fixes (zram 6G prio 100, 0.2s watchdogs
  SIGTERM below 0.75 GiB, swappiness=180) — HOST-level; note only.
- New image tags: none. `:exl3`/`latest`/`20260828-dflash2` all = same
  08-28 build (digest-pinned now). A future PR#77-equipped rebuild would
  be a NEW image (EXL3_FAT_KERNEL=1 + E2 kernels, MNBT 2048 recommended
  by the PR author, KV pool 1.9M vs 1.35M tokens at 7168) — adopt ONLY
  after #121's rebuild-corruption story is resolved, and expect hotfix
  anchor drift (fail-closed boot = re-derive time).

## Other-recipe crossover findings (from the in-repo recipe sweep)

- **Forum 378890 CORRECTED (08-02)** — `--max-num-batched-tokens` IS a
  fairness lever: 2048 vs 8192 → decode share 1.7%→5.0% (2.9×), p95 gaps
  6.10s→1.64s, prefill −7.7%, KV pool +63-67% (1.60M→2.61M). BUT chunk
  size does NOT restore new-request admission during a big prefill (still
  ~145-159s vs 1.66s warm) — the long-prefill threshold is the primary
  fix; MNBT is the supplementary fairness lever. The DSv4 aiden recipe
  adopted MNBT 2048+seqs 6 for agent traffic; for GLM the #112 table says
  keep 7168+3584 (no prefill tax) — A/B MNBT 2048 only if decode fairness
  during long prefills becomes the pain point.
- **CUDA-graph hot-shape (aiden, PR#5)**: auto-generated
  `cudagraph_capture_sizes` omitted the steady-state shape (36 = 6×(5+1));
  adding it = +9-14% at c4-c6. GLM never sets `--cudagraph-capture-sizes`
  and its hot shape is 4×(7+1)=32. **Check the boot log's captured-size
  list**; if 32 is absent, add it via EXTRA_ARGS (aiden's number is
  v0.11.2-specific, but the mechanism is the same vLLM).
- **`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`** (aiden): stops the
  profiler reserving full graph memory so KV gets the budget. GLM wires
  it as `CG_ESTIMATE` (default 1 = estimate on). Flipping to 0 is a KV
  lever COMPLEMENTARY to #86-rightsize (which we'll keep at stock given
  the #86 reversal). A/B only if KV pressure appears.
- **DSv4 #42 stop-in-reasoning guard arm** (tonyd2wild 09-02): guard only
  armed when the prompt's last token was the thinking start marker. GLM's
  template DOES emit the marker in-prompt (line 261) and we ship
  `GLM53_SUPPRESS_STOPS_IN_REASONING=1` — pattern check done, low risk.
  Worth one hostile-stop test (stop string in a self-opened think turn).
- **Silent-fallthrough discipline (DSv4 #37/#44/#46)**: DSv4 recipes
  learned that silent no-op patches (dtype fallthrough, dropped mounts,
  missing patch) run at half speed with NO error. GLM already fail-closes
  (`|| exit 1`) and greps boot markers; the kpool patch follows the same
  contract. If server throughput drops without a log error, re-verify all
  markers — don't trust a healthy-looking boot.
- **`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800`** (#51) — ALREADY in GLM
  compose (matches patch_spinwait's 96%@18W tell).
- **NCCL GID (DSv4 #38)**: leave unset, let NCCL pick (drift risk). GLM
  auto-detects the RoCEv2/IPv4 GID at boot — equivalent; don't pin.

## Open research queue (post-09-04-pass, in priority order)

1. **Decode fairness / admission recheck on OUR kit after the two adopted
   knobs** — A/B `DFLASH_TOKENS=3` (#113 + 4 lane corroboration) and
   `MAX_NUM_BATCHED_TOKENS=2048` (378890 corrected data) ONE AT A TIME
   against the 09-04 baseline, replicating the multi-session + subagent
   burst that timed out. Metric: `num_preemptions_total`,
   `prefix_cache_hits_total`, per-stream TTFT, aggregate tok/s.
2. Check the boot log's `cudagraph` capture-size list for 32
   (hot shape 4×8); add via EXTRA_ARGS if absent.
3. Grep for `EXL3_TEMP_ROWS_FUSED` support in our image (forum 382120's
   mixed-step fix); if exposed, A/B 192.
4. Watch #106 prefix-hits freeze under load; watch #114 boot-time variance
   (vision+1M); re-read the watchlist before every boot change.
5. If KV pressure appears: A/B `CG_ESTIMATE=0` (not #86-rightsize — see
   reversal above), or the host-level UMA squeeze.
6. Re-check GHCR for a NEW image (PR#77-fat-kernel equipped) — but only
   after #121's rebuild-corruption is resolved; a new image means
   re-deriving BOTH hotfixes (fail-closed boots are the signal).

## Serving concurrency (2026-09-04 user-reported) — RESOLVED IN PASS

**Symptom.** The user ran multiple omp sessions against this GLM endpoint and
one session spawned subagents (burst of concurrent streams); serving felt
"way too slow" and they switched sessions back to the DeepSeek recipe. Live
sample during such a burst: 3 running requests, KV usage 96–97%, generation
crawling at 0.3–5.3 tok/s, and at least one client timeout → full-context
re-send (the "extra slow" amplifier).

**Root cause found (same-image measurements, 09-03 upstream data):** this was
NOT primarily capacity. The 09-04 math shows 4×100k sessions use 656k of the
~1.05M pool (63%) and fit. The killers were:

1. **Long-prefill starvation (#110/PR#112)** — one long chunked prefill
   (a research session with a growing context) claimed the whole per-step
   token budget and froze every other session (~0.3-5 tok/s). Our 09-04
   earlier fix (`threshold 1024`) had the right mechanism but the WRONG
   value for MNBT=7168 (PR fixed-default 1024 → −44% contended prefill).
   **Now: `--long-prefill-token-threshold 3584` (= MNBT/2).**
2. **Mixed-prefill gate `skip` (#119)** — a newcomer arriving mid-decode
   (subagent burst) waited the whole running decode: 67-71s TTFT. With
   `skip` the running decode is protected but the newcomer is starved.
   **Now: `GLM53_MIXED_PREFILL_CHUNK=2048` → 10.1s TTFT** (running stream
   dips to ~1.9 tok/s only during the newcomer's own prefill).
3. **K-pool tail OOB (KPOOL_TAIL_BUG)** — a latent write-OOB in the image's
   hybrid attention path, made worse by long generations; intermittent
   corruption/crashes masquerading as "slow"/"weird". **Now patched**
   (`patch_kpool_tail_slotmap.py`, see hotfix section).

**What we deliberately did NOT change** (and why): `MAX_NUM_SEQS=4` stays —
#122's ~90k fixed KV/request is 25% of THEIR 356k pool but only ~7% of OUR
1.05M pool; #119's 165-preemption soak (MAX_NUM_SEQS=4) was on that tighter
pool, and ours fits 4×100k + outputs. Keep `stock` indexer workspace (#86
REVERSAL — rightsize now implicated in the stall). Keep MNBT=7168 (#112
table: 7168+3584 has no prefill tax; 2048 is the fairness fallback only).
Keep k=7 for now (A/B `DFLASH_TOKENS=3` is queued, see below).

**Still open (measure on the next GLM boot, one knob at a time):**
- k=3 A/B (best cross-lane evidence for mixed/prose agent traffic; only
  our geometry is untested).
- MNBT 2048 A/B if decode fairness during long prefills is still painful
  (378890: 2.9× decode share, −7.7% prefill, +63-67% KV pool).
- Boot-log check that `cudagraph_capture_sizes` includes 32 (hot shape
  4×8) — aiden's +9-14% hot-shape fix is the same vLLM mechanism.
- Monitor `num_preemptions_total` — drop MAX_NUM_SEQS to 3 only if
  preemptions appear under sustained 4-stream load.

Adopt nothing else without a measured A/B against the 09-04 baseline; the
validated invariants (worker-first, port 4000, offline serving, GMU wiring,
boot markers) are non-negotiable. Prefix-cache note: live hit rate was
81.9% during the burst; with gate=2048 in place #106's freeze did not
reproduce in a 75-min upstream soak — keep watching the counter.

## Research playbook for a fresh session

1. Snapshot this repo's state for the recipe (`git log -5 -- .`), confirm
   the running container's image digest still equals the pinned
   `:exl3@sha256:9bb1557a…` (inspect + registry HEAD), and confirm ALL FOUR
   boot markers on the leader (kvcheck-hotfix, padded slot-share, GPU KV
   cache size, positions=input_batch.positions).
2. MiaAI-Lab pass: new commits / open PRs / issues (the watchlist above);
   note anything touching the overlay, KV code, or serve flags.
3. Entrpi pass: same for their repos; diff their KV fixes against ours.
4. Image pass: GHCR tag list; if `:exl3` moved, pull on one node, boot in a
   scratch project name, and run the marker checks before touching the
   serving setup.
5. Adopt only what (a) applies to the running image/revision, (b) doesn't
   break the validated invariants (worker-first, port 4000, offline
   serving, GMU/hotfix wiring), and (c) carries a measured claim. Document
   the why in the commit, and update this file + README markers when the
   hotfix or profile changes.
