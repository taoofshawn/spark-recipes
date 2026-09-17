# research.md — deepseek-v4-flash-vision-0rand: maintenance notes

Working notes for future update/maintenance sessions on this recipe. The
README is the deploy doc (active-running content only); this file is the
memory — dated changelog entries, update-pass findings, and TODO/watch
items live here (AGENTS.md convention).

## Changelog

### 2026-09-17 — incident: boot-3 pairing (async ON + seqs 4) collapses decode at batch 2 under real agent load; profile restored

Troubleshoot-slowness pass on the live deployment (container up ~15 h = boot 3
of the 09-16 bring-up session). User report: "active session is very slow"
(~21:29 UTC). This is the missing receipt for boot-3 (6666086), which flipped
to the exact upstream pairing after the 09-16 entry was written.

**Root cause: concurrency-arrival decode collapse on `ASYNC_SCHEDULING=1` +
`MAX_NUM_SEQS=4`.** Whenever ≥2 requests ran, aggregate decode fell to
1.5-3 tok/s (~1 tok/s per stream vs 14-40 tok/s single-stream the same day).
Hardware, host memory, KV capacity and spec acceptance all exonerated; JIT
warm after the first batch-2 shape.

Evidence (UTC, container logs + /metrics on the head):

- 21:28:37 `_gumbel_sample_kernel` Triton JIT on BOTH TP workers — first
  batch≥2 sampling shape of the boot (warmup gap, one-off; jit_monitor's
  "extend warmup" advice is a cheap follow-up: warm a 2-seq shape).
- 21:28:40 run=3→2; 21:28:50-21:29:50 tg aggregate 1.5-2.4 sustained at
  run=2 (KV 16.6→18.7%, waiting=0). Snapped back to 16.7 tok/s at run=1
  the moment the intruder requests completed (21:30:00).
- Live probe (21:33-21:37, 96-token text completions, stream:false, ZERO
  new JIT lines in the window): single co-running with the session's turn
  53.0 s = 1.8 tok/s; concurrent pair 61.1/61.4 s = **1.6 tok/s per stream
  at batch 2, warm**; single after = 14.7 s. ~20× per-stream loss at
  batch 2, reproducible warm — not a warmup artifact.
- 21:35:40 run=1 wait=2 at KV 17.4% (pool ~2.9M): admission deferral with
  healthy capacity — off-profile behavior; mechanism not captured
  (async-scheduler deferral vs chunked-prefill budget; by-reason gauge not
  sampled in time). Second data point that the pairing misbehaves beyond
  decode rate.
- Host: si/so ≈ 0 (VmSwap 4.1 GiB = boot-time allocation), 6 GiB
  available, CPU ~85% idle. GPU: P0, 2184 MHz, no throttle reasons — but
  96% util at 1.7 tok/s = GPU spinning on per-step overhead, not stalled.
- SpecDec acceptance: 5.2-6.2 at run=1, 3.0-4.8 at run=2 — inside the
  normal content swing; not the driver.
- Prefix cache (watch item, NOT diagnosed): long-run
  700416/15303997 = **4.6% hit rate**; every agent turn re-prefills the
  full 11-15K context (pp spikes 12-15K all day; KV residual ~2.6-3.0%
  between turns) — vs 100/100 replay HITs in both 09-16 boots with
  co-le's cache-pressure. All three mods ARE applied this boot (`[fix-*]`
  done on head). Either the real agent prompt shape defeats replay
  (unstable prefix head?) or the fixes don't cover this workload — needs
  the cache-pressure tool against a captured agent prompt; no verdict.

**Cap arithmetic (skill formula):** min(MAX_SEQS, KV pool ÷ per-session,
spec-decoder efficient batch). One session turn allocates ~14% of the pool
(measured all day) → ~7 sessions fit; MAX_SEQS 4; measured spec-efficient
batch under the live pairing = **1** → interim client cap 1 (no restart).
Restored profile (async OFF + seqs 8, twice measured 09-16): spec-efficient
batch = 4 (c4 agg 59-68 tok/s, fixes ON) → cap 4.

**Fix (branch `dsv4-vision-0rand-restore-validated-profile`):** `.env`
ASYNC_SCHEDULING 1→0, MAX_NUM_SEQS 4→8 — the exact twice-measured config.
Attribution is correlation + elimination, flagged [INFERENCE]: async alone
is exonerated (boot-2 ran async ON at seqs 8, healthy at c4: 61.5/68.2);
seqs 4 is the only config delta vs both healthy boots; the causal mechanism
at seqs 4 is unproven. Deploy needs container recreate on BOTH nodes
(worker first, ~6-11 min warm boot). Verify after restore: c4 96-token
probe ≈ 55-70 tok/s aggregate, plus a cache-pressure pass.

**Addendum — boot-4 (restored profile) deployed on this branch + validated
(same day, per user call; node trees checked out on the recovery branch
without merging — deliberate deviation, both trees otherwise pristine):**

- Bring-up per the skill: teardown both nodes → worker first (GID=3
  auto-detected) → head ~40 s later → warm boot ~6 min (22:25:5x start →
  22:32:01 UTC /health 200). Markers green on BOTH ranks: three `[fix-*]`
  mods applied, `B12X_MXFP4_MXFP8` MoE, DSpark drafter loaded (k=6,
  probabilistic), no `--async-scheduling` in argv, `--max-num-seqs 8`,
  **KV pool 2,919,967 tokens** (the ~12K async tax gone), zero
  errors/tracebacks. model-name-proxy went transiently unhealthy during
  the boot window and recovered to healthy once :8000 served (its
  healthcheck probes the backend — expected, no action).
- Validation (stream:false, warm-up request first, tg = completion
  tokens actually generated; head, 22:33-22:36 UTC):

  | probe | boot-4 (restored) | 09-16 A/B receipt | boot-3 (broken) |
  |---|---|---|---|
  | c1 tg1024 | **59.1 tok/s** (first run hit EOS at 167 tok → 34.4) | 32.3-33.5 | 14-40 bursty |
  | c4 = 4× tg512 | **72.0 / 64.9 tok/s aggregate** (two rounds) | 59.4-68.3 | 1.6-3.1 aggregate |

  The c4 rounds traverse batch 1→2→3→4 with no collapse at any size —
  the exact shape that collapsed ~20× per-stream on boot-3 (1.6 tok/s
  per stream there; 16-18 tok/s per stream here). Only first-touch JITs
  fired this boot (`_topp_*`, one CuTeDSL gemm) with no sustained
  impact. c1 59.1 vs the bench's 32-34 is content/acceptance variation
  (different prompt class); not investigated further.
- Watch item stays OPEN: prefix-cache hit rate under the real agent
  workload (4.6% long-run observed on boot-3, full-context re-prefill
  every turn). Needs co-le's cache-pressure against a captured agent
  prompt — not tested in this boot.
- Ops note: nodes are on the recovery branch, NOT main. After the PR
  merges, `git checkout main && git pull origin main` on both nodes; the
  running containers already match the merged config, so no recreate is
  required at that point.




Bring-up per the skill (branch `dsv4-vision-0rand-0915-updates` checked out on
both nodes; GLM intel recipe torn down; checkpoint pin `86f746b3` verified
cached on both). Warm boots ~6-11 min; all three mods applied on BOTH ranks
at every boot (log lines `[fix-*] applying patch ... done`).

- **Tier 1 (shm)**: `docker exec ... df -h /dev/shm` on both nodes shows
  **61G tmpfs = the host's /dev/shm**, identical inside and outside the
  container — the compose `shm_size: "64gb"` is a **no-op under `ipc: host`**
  (Docker only sizes private /dev/shm). Effective ceiling is the host default
  (~50% RAM). The #325 SHM-reduction mitigation cannot be applied via
  compose; would need host remount or `ipc: private` + sized shm (own boot
  test). Recipes repo-wide share this dead knob (aiden 32g, glm 32g, here
  64g) — cleanup candidate, untouched in this branch.
- **Tier 2 A/B** (same bench: c1 tg1024 ×2, c4 = 4 parallel tg512 ×2;
  fixes ON in both boots; warm-up requests before measuring):

  | config | c1 tok/s | c4 agg tok/s | KV pool | cache-pressure |
  |---|---|---|---|---|
  | boot 1: async OFF, seqs 8 | 32.3 / 28.2 | 59.4 / 68.3 | 2,982,037 | 100/100 replay HIT |
  | boot 2: async ON, seqs 8 | 33.5 / 31.6 | 61.5 / 68.2 | 2,969,792 | 100/100 replay HIT |

  Verdict: **within noise** — no measurable async-scheduling win at c1/c4,
  and it costs ~12K tokens of KV pool. Final serving config = **async OFF**.
  The two-knob upstream pairing (async + seqs=4) remains untested; seqs=4
  contradicts the agent-serving profile and stays rejected. `ASYNC_SCHEDULING`
  knob kept (default 0) for future one-knob retries.
- **Prefix-cache fixes validated**: cache-pressure (`co-l/cache-pressure`,
  100 × ~8K contexts, --kv-size = logged pool) shows **100/100 replay HITs,
  ttft ~0.2 s, zero misses** on both boots — the 1-in-4 dead-zone signature
  is absent with the mods on. Fix ON left serving.
- Final boot markers: KV pool 2,902,514 (2.77× @1M ctx; pool varies
  boot-to-boot with free RAM at profiling — 2.90-2.98M observed). Proxy
  restarted on the leader with `BACKEND_MODEL=deepseek-v4-flash`; end-to-end
  through :4000 returns the real backend name (mode A) + sane reply.
- Watch for the soak: the #307 mid-decode stall → EngineDead pattern (SHM
  topic) and #259/#262 corruption-under-concurrency (b12x MoE) only surface
  under hours of real load — review after several days.

### 2026-09-15 — update pass: adopt prefix-cache fixes as opt-in mods; hold PilcoTHINK 0.29 image

Sources swept (window 2026-09-11 → 09-15): NVIDIA forum thread 381911 posts
#251–#348 + board sweep (cats 721/723) + full-text searches; GitHub (0rand
primary — zero commits since `d0c8584`; oselivanov/ollie-gb10-serving-stacks;
gpdev-Pilcothink Dockerfile lane — new `0.29/DSV4F-Vision-exp` lane @
`8bd44e88`/`bf701f7a`); Docker Hub (Dickson tag unmoved; **new**
`pilcothink/vllm_spark_dsv4fv:0.29` @ `sha256:ac497c0a…`, pushed 2026-09-15);
HF model (unchanged, pin still current).

**Adopted — prefix-cache fixes as opt-in mods (default OFF):**

- The shared 0.28.1 image has a measured **prefix-cache dead zone** (forum
  #296/#302, co-le's cache-pressure tool): a prompt ending 1..64 tokens past
  a 256-token boundary retains nothing reachable under sparse retention +
  EAGLE drafting → exact replay AND follow-up turn get 0 cached tokens (~1 in
  4 prompt lengths; stu.miller measured 0.82% retention under pressure).
  Present in BOTH Dickson's and Ollie's builds → in the shared engine, i.e.
  our image. Plus a same-content duplicate-block leak in `block_pool`.
- Vendored verbatim from oselivanov/ollie-gb10-serving-stacks into
  `mods/fix-dsv4-prefix-replay-tail/` (stu.miller's fix: retain the tail at
  the last reachable boundary `num_prompt - 1 - slack`; offline-verified
  zero-hit cases 192/768 → 0 across all 256 prompt-end offsets @ 8K/65K/262K,
  MNBT 4096 and 2048) and `mods/fix-vllm-prefix-cache-dedupe/` (port of
  co-l/ds4-prefix-cache-fixes 03-dedupe). Patches target
  `vllm/v1/core/single_type_kv_cache_manager.py`, `kv_cache_coordinator.py`,
  `block_pool.py` — our exact engine rev `0.28.1rc1.dev475+g6fbb00b18`
  (Ollie's fork is the same base). SHA256SUMS recorded. Wired as
  `FIX_DSV4_PREFIX_REPLAY_TAIL=0` / `FIX_PREFIX_CACHE_DEDUPE=0`
  (compose env + boot hooks, same fail-loud pattern as
  `FIX_MM_PREFIX_SPAN`); README section added.
- Receipts behind "on by default" in Ollie's stack: #312 (stu.miller, 553
  requests all OK, heavy agentic to 500K), #313 (co-le, "everything is
  perfect cache-wise", decode 45 t/s stable). Not enabled here yet — the
  2026-09-13 `FIX_MM_PREFIX_SPAN` bring-up precedent applies: enable +
  validate (cache-pressure tool) in a bring-up pass, not this one.

**Not adopted — `pilcothink/vllm_spark_dsv4fv:0.29` image (held):**

- PilcoTHINK shipped his own DSv4-vision lane today (forum #334): official
  vLLM **v0.29.0** + vision backports, DSML wrapper/streaming parser fixes,
  SM121 o_proj, packed FP8 linear (`VLLM_SPARK_PACKED_FP8_LINEAR=1` default),
  `FULL_AND_PIECEWISE` graphs ≤48 (our compose already matches), MNBT 8192,
  seqs 10, GMU 0.85, **k=3**, effort max. Measured: pp2048 ~1857 t/s (vs
  1577 on his 09-13 build), tg128 ~45 @0ctx, tg1024 ~44-46 @d1K-d4K,
  **TEB hardmode 91/100** (161/176, engine `0.29.1.dev0+g98dff2a81`).
- Why held: (a) quality receipt 91 < our image's 93/100; (b) k=3 default is
  the value upstream measured as thinking-damaging (#171) — our k=6 receipt
  config contradicts it; (c) provenance: vLLM v0.29.0 tagged 2026-09-09,
  BEFORE `9e257065` (#56141, 09-10) — GitHub compare
  `9e257065...98dff2a81` shows the engine does NOT descend from main's
  parser fix (diverged; 14 commits = upstream pre-09-10 cherry-picks), so
  the "parser fixes" are lane patches of unclear rev content; (d) the flag
  set diverges from our validated invariants (MNBT/seqs/GMU/k). Re-adoption
  trigger: a TEB ≥93 receipt on the lane at our shape, or prefill need
  (pp2048 2× our stack), with parser-fix provenance confirmed.

**Watchlist updates:**

- **vLLM #56141 (`9e25706`)** — still not confirmed in ANY consumable image:
  Dickson's is pre-09-10; Pilco's 0.29 lane likely backported it ("DSML
  wrapper … fixes") but unverifiable without a boot. Watch stays open;
  hand-backport into the digest-pinned image remains out of scope.
- **oselivanov/ollie-gb10-serving-stacks** — competing stack on the SAME
  engine base (`0.28.1rc1.dev475+g6fbb00b18.d20260907`): stability fixes
  (b12x MoE kernels blamed for the multi-stream degradation/corruption
  reports #259/#262 — #336 "DON'T recommend using it for Vision Exp"),
  dedup + replay-tail fixes on by default, TEB 93/100 (#308), 91 avg over 6
  runs (#335). The two prefix fixes are adopted above; his b12x warning and
  stability work are the reason corruption reports exist — if we ever see
  the #259 pattern, drop the moe backend first.
- **Kernel 7.0.0-1019 (DGX OS 7.5.0 OTA)** — NCCL/RoCE
  `ibv_reg_mr_iova2` ENOMEM → OOM deadlocks, no stack change needed to
  trigger (#315/#316; giles8's dedicated regression thread). Corroborates
  the repo-wide OTA watch; recorded in README ops notes. Stay on 6.17.x.
- **`index_topk` 512→1024 in the checkpoint's config.json** (#328,
  andriizahorui) — single unverified report of improved coherence; HF-
  config surgery, no second receipt. Watch only.
- **Mid-decode stall → EngineDead** (elvisnwh #307, #323) — unresolved;
  0rand's SHM-reduction tip (#325) recorded in README ops notes.
- Unchanged holds: async-scheduling + seqs=4 (#256 re-raises the perf
  question; still no measured claim at our seqs=8/batch-4096 profile);
  K=5 DSpark; DSv4.1 successor lane.

### 2026-09-12 — adopt upstream PR #1 mm-prefix span fix (opt-in mod)

Upstream merged oselivanov's PR #1 (`d0c8584`, 2026-09-12T16:13Z): the V2
model runner derived DSV4 vision mm-prefix bidirectional ranges from
`PlaceholderRange.extract_embeds_range()` (per-row-pair IMAGE-embed runs of
the N-layout) instead of the full sentinel block `[pad + IMAGE_START …
IMAGE_END]` — with the N-layout interleaving rows, y leaked into the causal
128-token window → y-only grounding bias ("center collapse"). Measured on
our exact image/rev: ball-sweep y 416/501/501 → 250/350/750, interior
errors ≤0.006 (PR #1 comments + INVESTIGATION-vision-grounding-bias.md
§12.7).

Adopted as an OPT-IN knob mirroring upstream (`FIX_MM_PREFIX_SPAN`, default
0 — upstream's default too):

- vendored `mods/fix-dsv4-mm-prefix-span/` verbatim from upstream @
  `d0c8584` (`dsv4-mm-prefix-span.patch` + `run.sh`, SHA256SUMS recorded).
  Python-only patch of `vllm/v1/worker/gpu/attn_utils.py`
  (`compute_mm_prefix_ranges`) + `vllm/v1/worker/gpu/model_states/default.py`
  (`DefaultModelState.prepare_attn`); idempotent `patch -p1 -N` with dry-run
  guards; applied on BOTH nodes at boot (compose mounts `./mods:/mods:ro`
  and runs the mod via `bash` — upstream stores `run.sh` non-executable,
  100644 — before `exec vllm serve` when `FIX_MM_PREFIX_SPAN=1`).
- patch failure is fatal by design: an explicitly requested fix must not be
  silently missing. If a future image changes the two files' context, boot
  stops loudly — re-vendor from upstream or flip the flag off.
- `.env` ships `FIX_MM_PREFIX_SPAN=0`; README documents the flip + both-node
  recreate requirement.

Caveat carried from the PR review (unanswered before merge): the V2 port
reads `mm_prefix_clamp_sliding_window` only from `hf_text_config` (V1 also
reads the module — "so it can't silently no-op later"). Harmless at our
pinned rev (attributes exist; the V1 path proves the mechanism); re-verify
on any image rev bump. Watch-list entry for PR #1 closed; vLLM #56141
(parser follow-up) remains the top image-refresh watch item.

### 2026-09-11 — upstream review: no runtime changes adopted

Sources swept (window 2026-09-10 → 09-11): NVIDIA forum thread 381911
(all posts >#214) + board sweep (categories 721/723) + full-text searches;
GitHub (0rand primary + siblings, PilcoTHINK lane, Dickson, vLLM
`deepseek_v4` parser); Docker Hub tags; HF model revisions.

**Pins verified current (no action):**

- Image `dicksondickson/vllm_spark_dsv4:0.29-b12x` @ `sha256:899d174e…` is
  still the only tag (count=1 at page_size 25 and 100), digest unmoved
  since its 2026-09-10T06:26Z push; 0rand confirmed this exact pin on the
  forum (#229).
- 0rand's own hub image (`0rand/vllm_spark_dsv4-0.29-b12x:latest` @
  `85eb91ee`) unchanged since 2026-09-08 — still the superseded build.
- HF `DeepSeek-V4-Flash-Vision-Exp`: HEAD `6821d6ad` (lastModified
  2026-09-01) vs pinned `86f746b3` — git-tree OID diff shows `README.md` +
  `.eval_results/` only; `encoding/encoding_dsv4.py`, tokenizer/config
  files, and all 48 shards byte-identical (same LFS sha256s). Keep
  `86f746b3`; a pin==HEAD bump would buy nothing.
- PilcoTHINK Dockerfile lane: zero commits after `38db013b` (pin = HEAD);
  no 0.29/0.30 lane directory exists.

**Upstream delta since our pin (`74e9f11` → `b94f1fbd`):** one functional
commit, `0cf86aa` (2026-09-10): image → Dickson's build (already adopted
09-10), `MAX_NUM_SEQS` 8→4, and `--async-scheduling` (commit message:
"was mistakingly omitted" — i.e. intended config, absent at our pin by
accident).

**Not adopted (re-evaluated):** `--async-scheduling` + seqs=4 pairing.
Still no measured claim at our seqs=8 / batch-4096 profile; the seqs flip
contradicts our validated agent-serving profile; the DSpark +
BREAKABLE_CUDAGRAPH interaction is unverified on this image. Revisit if
numbers land at a comparable profile.

**Watch list (adoption triggers):**

- **vLLM #56141 (`9e25706`, merged 2026-09-10T07:04Z)** — "[Bugfix]
  Tolerate misspelled DSML tool_calls wrapper": production-observed DSML
  opener misspellings (`<｜DSML｜tool>` vs `<｜DSML｜toolcalls>`) emitted by
  DSv4-Flash itself; direct follow-up to the `d98c8c03` parser fix baked
  into our image. Merged 38 min AFTER Dickson's image push → NOT in any
  published vision image; PilcoTHINK's newer engine lane (vLLM
  `0.28.1rc1.dev637+g9e2570656`, forum 382957) contains it but has no
  DSv4-vision image yet. → Adopt on the next image refresh that contains
  `9e25706`; do NOT hand-backport the multi-file parser change into the
  digest-pinned image (untestable without a boot; outside the update
  skill's no-launch scope).
- **0rand PR #1** (oselivanov) — DSV4 vision mm-prefix span fix in the V2
  model runner (bidirectional range derived from the full sentinel block,
  not just the IMAGE-embed runs); measured grounding repair (ball-sweep y
  416/501/501 → 250/350/750, interior error ≤0.006), gated by
  `FIX_MM_PREFIX_SPAN=1`. **MERGED 2026-09-12 (`d0c8584`) — adopted here as
  the opt-in `FIX_MM_PREFIX_SPAN` knob; see the 2026-09-12 entry.**
- **vLLM #52865** (open) — DSv4/V3.2 tool-argument streaming: long string
  args buffered until `</parameter>` instead of streaming incrementally;
  touches our pinned parser file.
- **vLLM #50753** (open) — surface unterminated reasoning as content
  (opt-in `force_nonempty_content`); author validated on 2× GB10 TP=2.
- **0rand's B12X bug hunt** (forum 382939 #2/#4) — B12X bugs "affect both
  GLM and DS4FVE with high probability"; quality testing so far shows no
  impact ("it just folds to a different path"). Results pending — could
  touch the B12X env family.
- **K=5 DSpark** (forum 382675 + `hxfhd/deepseek-v4-dspark-k5-workaround`)
  — validator-patched k=5 measured on aiden `production-3.73-vision`
  (draft work −15–20%, avg 47.5 tok/s): NOT our image, author calls it
  unsupported, and #171 is precedent for off-standard-k thinking damage.
  HOLD. Side-receipt: k=6 is the smallest stock-legal k for this
  checkpoint (n_predict=3, dspark_block_size=5) — validates our k=6 pin.
- **DSv4.1-Flash successor lane** — vLLM #56208 frontend merged;
  tonyd2wild 4× TP4 measured 77.2 tok/s peak / 1M ctx (forum 382897);
  2-Spark feasibility unknown (the "552B" is int8 double-packed per
  381911 #217–#225; 2–3-bit quant would be needed). Track as a future
  lane, not this recipe.

**Recipe hygiene in this pass:**

- compose command-block fallback for `MAX_NUM_BATCHED_TOKENS` aligned
  2048 → 4096 (matches `.env`, the `environment:` block, and the README
  shipped value; no behavior change — `.env` always sets it).
- The 2026-09-10 changelog block moved here from the README per the
  AGENTS.md convention (README = active-running doc only).
- Ops knowledge for the running cluster distilled into the README's new
  "Ops notes" section (prefill headroom, client max-output-tokens
  footgun, GB10 clock latch, GB10 cgroups gap).

### 2026-09-10 — switch to Dickson image (upstream tool-call parser fix)

*(moved from the README per the AGENTS.md convention)*

**What changed:** the image pin moved from `0rand/vllm_spark_dsv4-0.29-b12x`
@ `85eb91ee` (PilcoTHINK `9df77f2e` lane) to `dicksondickson/vllm_spark_dsv4`
@ `899d174e` (Dickson's build of the PilcoTHINK `38db013b` lane). Same vLLM
base `6fbb00b1`, same model pin, same `o_proj` donor — the only functional
delta is that `38db013b` **bakes in the upstream DeepSeek V4 tool-call parser
fix** (`vllm/parser/deepseek_v4.py` from vLLM `d98c8c03`, "[Bugfix] Parse DSML
tool calls when the model omits the tool_calls wrapper" #55954). The earlier
0rand image predates that commit.

**Why:** the parser fix is the measured delta behind the higher tool-call
quality on this image. stu.miller's locked v1.8.0 comparison (forum #212):
**pilco/0rand 86 vs Dickson 93** tool-eval-bench (Dickson 93/100 post #176,
92/100 post #194). 0rand himself switched to the same image (forum #214) but
has not yet pushed his own hub tag.

**Not adopted:** `--async-scheduling` (Dickson's only other change, post #194,
paired with `MAX_NUM_SEQS=4`, post #200). No measured claim at our seqs=8 /
batch-4096 profile, and it wasn't shipped in 0rand's `.env.sample` — left off
to avoid an un-validated knob on a fragile cross-recipe dimension.
*(2026-09-11 note: `0cf86aa` has since shipped it upstream paired with
seqs=4; still holding — see the 2026-09-11 entry.)*

**Gotchas:** image is arm64 (GB10-correct), ~24.2 GiB pulled; cold-boot JIT is
unchanged. No flag/env changes — kv-cache-dtype fp8, FLASHINFER_MLA_SPARSE_DSV4,
AOT=0/BREAKABLE=1, GMU 0.87, k=6, port 8000, worker-first all preserved.
