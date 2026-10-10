# research.md — glm-v53-flash-nvfp4-0rand: maintenance notes

Working notes for future update/maintenance sessions on this recipe. The
README is the deploy doc; this file is the memory.

## Provenance (2026-09-14 adoption pass — adopt-recipe skill)

Every non-cluster value traces to a reviewed source:

| value | source | receipt |
|---|---|---|
| `IMAGE=pilcothink/vllm_spark_glm53:0.28` | 0rand `.env.sample` + README; jetspark post 17 ("sha256:e99cb670…") | `start.sh` default |
| `MODEL=nvidia/GLM-5.3-Flash-NVFP4` @ `09b04e5e74bca08ca8549fc736d4cdd8624bfde3` | upstream `docs/FINDINGS.md` §1 verified config | — |
| `--moe-backend b12x --linear-backend b12x`, `--reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice`, `--mamba-cache-mode align`, `--kv-cache-dtype fp8 --block-size 256`, `--enable-prefix-caching --enable-chunked-prefill`, `--skip-mm-profiling`, `--no-enable-flashinfer-autotune`, `--dtype bfloat16` | upstream `start.sh` `vllm serve` block (@ `0227b8df`) | — |
| spec-config shape (`attention_backend: TRITON_ATTN`, `kv_cache_dtype: auto`) + `SPEC_EXTRAS` extras (`draft_sample_method: probabilistic`, `rejection_sample_method: standard`, `enable_adaptive_verification: false`, `disable_eagle_block_drop: false`) | `start.sh` SPEC_CONFIG python block (`RECIPE_SPEC_EXTRAS`) | — |
| GMU 0.88 / KV pin 9,663,676,416 / seqs 4 / batch 1024 / capture 16 / async ON / estimator OFF / ctx 900,096 | `.env.sample` defaults + `FINDINGS.md` §1 | — |
| K=5 default; k=4 and k=7 as tuning rows | `.env.sample` (K=5); posts 17/39 (k=4); post 1 (k=7) | — |
| drafter `incoai/GLM-5.3-Flash-DFlash2` | upstream README "Provenance" | upstream does NOT pin a revision (`DRAFT_NAME=glm53-dflash2-orig`) → we pin `bf582e4e` (the intel recipe's dflash2pmu pin; HF HEAD as of 2026-09-08) — see Watchlist |
| env block: `CUTE_DSL_ARCH=sm_121a`, `TORCH_USE_RTLD_GLOBAL=1`, `TORCHINDUCTOR_COMPILE_THREADS=1`, `VLLM_ENGINE_READY_TIMEOUT_S=3600`, `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, `VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=auto`, `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | `start.sh` `-e` flags | — |
| `PORT=8000` (upstream 8100), `MPORT=29521` (upstream 29503) | repo convention (AGENTS.md) | skill-default override |
| `SERVED_MODEL_NAME=glm-5.3-flash` (upstream: empty = model path) | skill default (same parent model) | skill-default override |
| temp 1.0 / top_p 0.95 / thinking true / effort high pins | skill defaults; jetspark post 39 ran TEB with exactly these request-side (`chat_template_kwargs`) | skill-default override |
| NCCL block beyond 0rand's five vars | reference recipe `glm-v53-flash-intel-w4a16/docker-compose.yml` | cluster-level RoCE-v2 fabric wiring, image-independent |

## Receipts (author-reported; different hosts/configs — do not mix)

- **0rand 900K profile** (repo README, 2026-09-13): boot ~980 s; KV pool
  1,080,115 (1.20×); pp1024/tg1024 c1 = 32.0 t/s; pp1024/tg512 c1 =
  40.3/28.8/33.7 t/s @ depth 0/2K/8K; API smoke "DRAGON ONLINE". **No hardmode
  on this profile by 0rand** — "the 94/100 result below belongs to the prior
  700,160-token profile and is not silently transferred."
- **paxren2020** (post 41, 2026-09-14): **95/100** (167/176; 81 pass/5 partial/
  2 fail), Max context 900,096, engine vLLM 0.28.1rc1.dev475+g6fbb00b18,
  responsiveness 25/100 (median turn 6.1 s), deployability 74, weakest M
  Autonomous Planning (83%), 2286.4 s. Independent replication of the 900K
  profile.
- **0rand 700K** (`FINDINGS.md` §1): 94/100 (166/176, Hard Mode 38/38), e2e
  996 s, pool 892,139 (1.27×); spec-bench warmed: structured 49.7 (α=88%) /
  code 43.0 (α=71%) / filler 33.5 (α=52%); failures TC-43/TC-68, safety flag
  TC-47 (reproduces across runs — model behaviour, not config).
- **pilcothink** (post 1): GMU 0.9 @262K → 743,165-token pool, 2.83×
  concurrency; TEB 94/100 (166/176) @262K; TG128/TG1024: MTP=5 26.8/23.7,
  DFlash k=5 35.7/28.0, k=7 39.1/30.8 t/s.
- **jetspark** (post 39): memory zones — weights 90.6 GiB (meas), OS+CUDA
  13/11 GiB, KV 6.5 GiB → 668,803 tok (meas), activations ≈ MNBT ×
  ~800 KiB/token (8192 → ~6.4 GiB est), draft k=4 ~2.2 GiB (est), graphs-off
  0.46 GiB (meas), FREE 2.3/4.3 GiB. Spec-bench ladder d0→d690000 (tg
  27.3–50.4 eff t/s; TTFT 1.0 s @d0 → 746 s @d690K). Extended-TEB 97/100
  (170/176) @524K parallel 3, median turn 11.0 s. "I wasn't able to achieve
  more than two parallel requests because of the batch size"; "DFlash scales
  really poorly with concurrency, MTP beats it".
- **jetspark** (post 17): nvidia W4A4 vs LibertAIDAI W4A16 — W4A4 needs four
  starts, drops video + concurrency, cannot use its own MTP head; TEB 90 vs
  94 (within test noise); k=4 ≈ 20% faster decode than k=7.
- **pilcothink** (post 44): 1M context running on this stack, "currently
  delivers the fastest speed" — no config receipts attached.
- **paxren2020** (post 45, 2026-09-14): agent-serving impressions at 900K —
  GLM ~2× slower than DeepSeek, only 4 concurrent agents vs 8; **2 of 4 agents
  dropped out mid-task** (6 h run). Real-world confirmation of the concurrency
  ceiling on this stack.
- **jetspark** (post 53, 2026-09-15): **98/100** on the nvidia NVFP4 quant with
  his customized TEB (system-prompt tweak only) — the result that convinced him
  off EXL3. Same model, yet another bench build — quote the evaluator (§6).

## Memory model on GB10 (from FINDINGS — applies to the pilcothink build)

1. **GMU is a fraction of whole-system RAM** and the pilcothink build enforces
   it as a **fatal** pre-load check (~45 s in). Ceiling was 0.88 on 0rand's
   host; **re-measure on ours before trusting it** (the intel recipe ships
   0.85 on the same boxes — a sane fallback). `--kv-cache-memory` does not
   bypass the gate.
2. **CUDA-graph memory estimator tax**: −2.1 GiB of KV in graph mode unless
   `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` (shipped default).
3. **Batch/seqs are KV levers**: the 700K breakthrough was +4.11 GiB of pool =
   estimator off (+2.11) + batch/seqs trim (+2.00).
4. Failure catalogue (FINDINGS §5): GMU gate (fatal, pre-weights); KV-too-small
   for ctx (dies ~15 min in — the engine names the largest ctx that fits);
   `${VAR:+}` expanding on 0 (why the compose uses explicit `= "1"` tests).

## Watchlist

- **Drafter revision**: upstream unpinned (`glm53-dflash2-orig`); we pin
  `bf582e4e`. At first boot compare `/metrics` spec-decode counters
  (`vllm:spec_decode_num_accepted_tokens_total` vs drafts) against upstream's
  α≈88% structured / 71% code — a big gap means the drafter revision differs.
- **`SPEC_EXTRAS`**: semantics known only from `start.sh`'s JSON. Note
  `disable_eagle_block_drop: false` here vs `true` in the intel recipe's
  dflash2 lanes — DIFFERENT IMAGE, do not port between them without an A/B.
- **`--privileged`**: upstream's `vendor/run_cluster_dual.sh` ran the container
  privileged; this compose uses house style (gpus + `/dev/infiniband` +
  IPC_LOCK). If NCCL/IB init fails at boot, that is the first knob to revisit.
- **`ATTN_BACKEND`** left empty = image default (sparse-MLA). Only set it if
  upstream documents a value.
- **b12x bugs**: 0rand (post 2) is hunting b12x bugs affecting GLM and
  DS4FVE; his verdict so far: "it just folds to a different path, not
  inferior" (post 4). Track his findings before image bumps.
- **Concurrency for 4 sessions**: `MAX_SEQS` ships at 4 (the cap) and the pool
  is only 1.20× at 900K — four concurrent long sessions must shrink context
  (~225K each at 900K-pin). The intel recipe's pool (~1.87M @1M) is the
  concurrency-friendlier stack; this recipe is the long-context/quality stack.
- **TEB version churn**: the evaluator moved 40 commits in one day during this
  thread — always quote the bench build (FINDINGS §6).
- **Driver**: 580.173.02 current; 610.43.02 unstable/bigger UMA (repo-wide).
- **tonyd2wild `GLM53_INDEXER_WORKSPACE` / SM121 indexer overlay**: NOT used
  here — that is the other image lineage's crash fix. Do not copy it onto this
  stack without evidence.
- **DGX OS 7.5.0 OTA (forum 383222)**: boots with **~7.2 GiB less RAM**
  (119.5 → 112.3 GiB kernel-available; UEFI 5.36_0ACUM027 + kernel
  7.0.0-1019-nvidia). The pilcothink GMU gate is a **fatal** check against
  whole-system RAM — post 36 shows free-at-startup ≈108.2 GiB passing 0.85 but
  failing 0.9; a −7.2 GiB drop would fail the shipped GMU 0.88 (107.1 GiB).
  Do NOT take this OTA on the cluster without lowering GMU and re-measuring.

## Changelog

### 2026-09-23 — bring-up A/B: 1M profile validated on-cluster, MNBT 4096 wins c2, 0.29 hold stands (branch `glm-nvfp4-0rand-0922-updates`)

Four cold-cache boots (down-both → drop_caches → worker-first → head),
identical bench each side: warm-up ×2, c1 = 2× single-stream tg1024,
c2 = 2 rounds of 2-parallel tg512 (the 09-18 async-A/B lane shape), temp 0,
`stream:false`, real `usage.completion_tokens`, EOS-short rounds excluded
(none occurred), α from `/metrics` counter deltas. ~2K-token VARIED prose
prompt (deterministic shuffle, baked in `~/bench_glm_ab.py` on the head).
Gotcha hit: the first attempt used a repeated-paragraph prompt — temp-0
continuation of repeated text pins per-position acceptance at ~1.0 (α=0.991,
c1 58 t/s, flat 832/832/831/829) and inflates tok/s ~2×; replaced with varied
prose before any verdict. c1 rounds within a boot are ultra-stable (<1%);
ACROSS boots, c1/c2 track α (0.89–0.98) — the content-acceptance swing the
±10% rule warns about. Never benchmarked right after boot (health→bench gap
≥2 min plus warm-up; boots below are health-time, not serving-time).

| boot | config | KV pool | boot (head) | c1 tg1024 | c2 2×tg512 | α |
|---|---|---|---|---|---|---|
| A | shipped 700K, MNBT 2048, 9 GiB pin, 0.28 | 1,027,894 (1.47×) | ~609 s | 53.36/53.67 | 62.93/62.19 | 0.930 |
| B | 1M, MNBT 1024, 10.24 GB pin | 1,172,644 (1.12×) — **exact upstream match** | ~990 s | 51.23/52.47 | 64.86/61.55 | 0.952 |
| C | 1M, MNBT 4096 | 1,150,684 (1.10×) | ~983 s | 52.67/52.80 | **66.06/77.20** | 0.980 |
| D | 0.29 image @ shipped 700K | 1,027,894 (1.47×) | ~974 s | 47.75/47.65 | 66.26/61.41 | 0.895 |

- **GMU gate (the main risk item): PASSED first try.** GMU 0.88 with the
  10.24 GB pin cleared the fatal whole-system-RAM pre-load check on BOTH
  1M boots (B and C) — no stepwise pin lowering (9.5→9 GiB) was needed and
  0.85 was never touched. Host free RAM was ~117 GiB at boot time (well above
  the ~107.1 GiB the 0.88 gate needs). 0rand's 1M profile is launch-verified
  on THIS cluster now.
- **MNBT 1024-vs-4096 verdict: 4096** (adopted). C vs B: c1 +1.7% (noise),
  c2 +13.3% with non-overlapping ranges (C min 66.06 > B max 64.86). The
  c2 gain is not pure α luck: C's α (0.980) exceeds B's (0.952), but C's c1
  stayed flat while c2 rose — acceptance boosts both cells proportionally,
  so a c1-flat/c2-up split points at real batching behavior under 2 streams.
  Cost: pool 1,172,644 → 1,150,684 (−21,960 tokens, larger activation
  reserve). Matches 0rand's post-82 daily-driver batch.
- **0.29 image verdict: NO — hold stands.** c2 +2.0% (overlap, NOISE),
  c1 −10.9% (no overlap, but α fell 0.930→0.895 — decode tracks acceptance).
  Same pool as boot A (1,027,894). Consistent with ttsiodras post 92
  ("~same as 0.28"). The held-image item keeps: no reason to bump.
- **Config left serving: boot C** (1M ctx, MNBT 4096, 10.24 GB pin, GMU 0.88,
  seqs 4, ASYNC=1, k=5, image 0.28). Rationale: c2 aggregate (the
  discriminator cell) beats boot A by +14.5% median (beyond the ±10% band,
  ranges disjoint) with c1 flat; ties-or-beats B; adds 1M ctx over the
  shipped 700K at no measured speed cost. B vs A was a tie (c1 −3.1%, c2
  +1.0%).
- α vs upstream's ~88% structured / ~71% code: NOT a mismatch signal —
  upstream's receipts are spec-bench structured/code; our bench is temp-0
  prose continuation, which the drafter finds easy (α 0.89–0.98 across all
  four boots, per-position decay healthy 1707→1624). Report-only, per plan.
- Display-KV: OUT OF SCOPE (user decision; documented, not wired). Async:
  not relitigated (two A/Bs on 09-18, no delta at seqs 4).
- Boot markers (all four boots): `B12X NvFp4 MoE`, `DFlash2DraftModel`,
  `SpeculativeConfig(method='dflash', num_spec_tokens=5)`, Eagle3 aux
  `(6, 15, 25, 34, 43)`, graph capture ~10–12 s, proxy `BACKEND_MODEL`
  `glm-5.3-flash` matched.
- Reference-number caveat: absolute c1 here (47–54) far exceeds the 09-18
  async A/B's c1 (22–27) — different prompt/API shape (raw `/v1/completions`,
  ~2K varied prompt, no chat-template thinking segment), so only intra-matrix
  comparisons are meaningful; the 09-18 lane's shape was not reproduced.
- Post-A/B (same day, user request): the winner config was made the shipped
  default — README default-profile bullet, receipts table (1M row added,
  900K demoted to prior default), verify/boot-marker examples (max_model_len
  1048576, pool ~1,150,684 @MNBT 4096), and tuning rows (prior-900K and
  MNBT-1024 rows added) all synced to `.env`.

### 2026-09-23 — update pass: adopt the 1M production profile as a tuning row + document the upstream display-KV variant (held)

Sources swept (window 2026-09-15/18 → 09-23): upstream repo (0rand primary —
one new commit `4ec03bf`, 2026-09-22T10:59Z), forum 382939 posts 56–92 (last
post 2026-09-23), Docker Hub (`pilcothink/vllm_spark_glm53` — NEW tag `0.29`
@ `sha256:82807abe…`, pushed 2026-09-17; pinned `0.28` @ `e99cb670` unmoved),
HF model/drafter, board sweep.

**Pins verified current (live, no action):** HF weights
`nvidia/GLM-5.3-Flash-NVFP4` @ `09b04e5e` (lastModified 2026-09-11, exactly our
pin; 33 shards + index unchanged) and drafter `incoai/GLM-5.3-Flash-DFlash2` @
`bf582e4e` (lastModified 2026-08-31, exactly our pin). PilcoTHINK Dockerfile
lane: `0.29/GLM53-flash` guide update `6eedaa37` (2026-09-17) = the `0.29`
image push above; `0.28` lane untouched. Sibling stack watch: Ollie's
ollie-gb10-serving-stacks switched his GLM/DSv4 stack to the **marlin linear
backend** (`1e0297f`, 2026-09-17) — no config change here; see the DSv4-vision
recipe's b12x watch. `local-inference-lab/GLM-5.3-Flash-NVFP4` requant pushed
2026-09-16 (`175ae8ce`, QAD "quantization-aware-distillation") — 0rand tested
it on the forum (post 80): "slower and constantly choked" — watch only.

**Upstream delta since our pin (`e1de4ab` → `4ec03bf`):** one functional commit,
`4ec03bf` (2026-09-22): "display-KV variant: +18.4% KV via display-reserve
unlock (experimental)". Files: `start-display-kv.sh` (+249), `display-kv/`
(shim `display_kv_glm.py` + `sitecustomize.py` + `libdisplay_kv_glm.so` +
toolkit + AGPL allocator source from coolbho3k's DeepSeek-v4.1-Flash repo),
README sections, `.env.sample` re-framing. **`start.sh` is untouched** —
`0227b8df` remains the authoritative production flag set.

**Adopted — the launch-verified 1M production profile as a tuning row (docs +
`.env` comment; shipped profile unchanged):**

- Upstream README now documents a **launch-verified 1M profile (2026-09-22)**:
  ctx 1,048,576, KV pool 1,172,644 production, GMU 0.88, **10,240,000,000-byte
  KV pin** ("11 GB"), seqs 4, batch 1024, async ON, boot ~13 min; max
  concurrency at full 1M ctx 1.12×. Post 82 (0rand, 2026-09-22) confirms it as
  his daily driver: "1m kv cache with 11GB kv cache pin, 4096 batch, 1m
  session. 120/122gb on head node, stable, no oom, no issues, 30-35 t/s on
  mixed" — note the batch discrepancy (README 1024 vs post 4096).
- `.env.sample` now frames the 900K/9 GiB profile as a "conservative FIRST-BOOT
  demo profile" and points production at 1M + 10.24 GB. Our shipped `.env`
  stays at the operator's middle-ground profile (700K / MNBT 2048 / 9 GiB pin,
  the deliberate 09-16 revert state) — flipping `MAX_LEN` to 1M is a one-line
  change + bench (the 10.24 GB pin must clear the fatal whole-system-RAM GMU
  gate on OUR hosts; upstream's ceiling was measured on his). Added as a tuning
  row in README + a commented alternative in `.env`.

**Documented (HELD) — the display-KV variant (EXPERIMENTAL):**

- Mechanism: runtime-only `nvidia_drm` reload (`modeset=1 fbdev=0`, never
  persisted; reboot restores) on both nodes, then a fail-closed allocator shim
  replaces vLLM's final KV `torch.zeros` backing with a contiguous UVA span:
  production 10.24 GB ordinary pin (never exceeded) + 1.75 GiB DRM display
  carveout per rank. Measured A/B (2026-09-22): KV pool 1,388,762 vs 1,172,644
  (+216,118, **+18.4%**); concurrency at 1M ctx 1.32× vs 1.12×; TEB hardmode
  91/100 (160/176, parallel 4, seed 42), 0.0% error rate; no decode tax within
  fluctuation (fill run ~10% faster, in the noise band). Bandwidth receipts:
  display segment SM reads 163–168 GB/s vs 235–269 GB/s ordinary cudaMalloc;
  copy-engine DMA catastrophic (0.9–2.3 GB/s) — attention kernels use SM loads,
  which is why KV works.
- Why held, not adopted: (a) requires **host-level changes** the compose cannot
  express — the runtime DRM reload with **passwordless sudo on the worker**
  (head prompts once per boot); (b) the allocator is **AGPL-3.0** — vendoring
  into this repo is a licensing decision not taken this pass; (c) the shim is
  coupled to upstream's `start.sh`/`run_cluster_dual.sh` launcher lane, not
  our docker-compose conventions; (d) driver-dependent (verified 580.x,
  fails on 595.84 upstream) — a boot-test + A/B at our shape is needed first.
- Re-adoption trigger: the operator wants +18.4% KV at 1M ctx and accepts the
  sudo/DRM host prerequisites → vendor `display-kv/` verbatim + a host DRM
  runbook step + compose opt-in env (`PYTHONPATH=/opt/display-kv`,
  `DISPLAY_KV_*`, `KV_CACHE_MEMORY=10240000000`, `MAX_LEN=1048576`) in a
  bring-up pass with before/after bench. Documented in README
  "Display-KV variant".

**Not adopted — `pilcothink/vllm_spark_glm53:0.29` (held):**

- New tag pushed 2026-09-17T16:24Z (`sha256:82807abe…`); the 0.29 GLM53 lane
  adds the `GLM53_CONTEXT_CUDA_GRAPH` / `GLM53_GROUPED_CONTEXT_STORE` /
  `GLM53_FUSED_DFLASH_TAPS` / `GLM53_CONTEXT_GRAPH_CACHE_V2` env family.
  ttsiodras's receipt (post 92, 2026-09-22): GMU 0.9, batch 8192, seqs 10,
  262K ctx, graphs ≤64 — TEB + llama-benchy ~same as 0.28.
- Why held: the new flags are unvalidated at our shape (seqs 4 / batch 2048 /
  1M ctx); the pinned 0.28 digest is unmoved and its receipts unchanged; an
  image bump requires a bring-up A/B, not this update pass. Recorded as a
  README known-issues hold.

**Forum receipts adopted as documentation (no config):**

- Post 63/64 (pilcothink): W4A16-vs-W4A4 workload fit — NVFP4 W4A4 shines at
  high concurrency on datacenter Blackwell; on GB10 "support for NVFP4
  computation still appears to be somewhat limited" — W4A16 can win at low
  concurrency. Context for the intel-vs-nvidia recipe question; no config.
- Post 79/80 (0rand): tested the **local-inference-lab NVFP4 requant** —
  "slower and constantly choked switching between prefill and decode. I have
  image and recipe but not using it and no point publishing". Receipt for
  staying on the nvidia quant; watch item closed.
- Post 82/84 (Ama5u): same recipe at 900K running stable for a week on his
  nodes — independent stability receipt for the shipped profile family.
- Watch items updated: **driver 580.178.04 random hard freezes** (post 83 +
  amasu/glm53-flash-cluster diagnostics; random, days into uptime) added to
  the README driver watch — our cluster is on 580.173.02, stay put. New
  quants spotted (brandonmusic/GLM-5.3-Flash-tr3-4bpw,
  canada-quant/GLM-5.3-Flash-W4A16-MTP): watch only, no receipts.

**Gotchas hit:** the upstream `.env.sample` re-framing is comment-only (the
`KV_CACHE_MEMORY=9663676416` demo default and all real keys are unchanged) —
no flag drift to mirror. Upstream `start.sh` blob is byte-identical to our
pinned `0227b8df` state (`4ec03bf` did not touch it), so no spec/serve flag
changes propagated into the compose.

- **2026-09-14** — adopted from
  `0rand/glm-5.3-flash-nvidia-nvfp4-dflash-2x-dgx-sparks` (README 2026-09-13
  state; `start.sh` @ `0227b8dfd13397eee04051817b03b9c04e808a67`;
  `docs/FINDINGS.md`; `.env.sample`) + NVIDIA forum thread 382939 (posts 1–44
  reviewed; post 44 = pilcothink's 1M-context claim). Recipe generated by the
  `adopt-recipe` skill on branch `adopt-glm-v53-flash-nvfp4-0rand`, left
  uncommitted by design. Skill-default overrides applied: port 8000, served
  name `glm-5.3-flash`, temp/top_p/thinking/effort pins; drafter revision
  pinned where upstream left it unpinned.
- **2026-09-15** — first update pass (recipe-update skill, branch
  `adopt-glm-v53-flash-nvfp4-0rand`). Upstream repo: one new commit `e1de4ab`
  (README-only: drops the "no hardmode on 900K yet" caveat — the 95/100 is post
  41's replication; fixes ZMQ bind-address prose). `start.sh`, `.env.sample`,
  `docs/FINDINGS.md` byte-identical to the pinned `start.sh` blob `0227b8df`.
  Image `pilcothink/vllm_spark_glm53`: still only tag `0.28`. HF: weights
  `09b04e5e` and drafter `bf582e4e` unchanged. Forum 382939 posts 45–55: no new
  flags; adopted as receipts — post 45 (2 of 4 concurrent agents dropped at
  900K) and post 53 (jetspark 98/100 on the NVFP4 quant). General sweep of cats
  721/723 since 09-14: no recipe-relevant config; new repo-wide watch item —
  DGX OS 7.5.0 OTA −7.2 GiB boot RAM vs the fatal GMU gate (forum 383222).
  No config changes required.
- **2026-09-16** — TEMP direct-:4000 experiment ended; config reverted (branch
  `revert-glm-nvfp4-temp-port-4000`, left uncommitted by request). Commit
  `3fd441b` (merged via #61) had temporarily moved serving to `PORT=4000` with
  `SERVED_MODEL_NAME=spark-llm` and the model-name proxy disabled (not
  started), plus a middle-ground profile `MAX_LEN=700160` / `MNBT=2048`.
  Reverted back to the cluster convention: `PORT=8000`, served name
  `glm-5.3-flash`, proxy re-enabled for the next bring-up (the bring-up skill's
  step 3b re-points proxy `BACKEND_MODEL` and restarts it on mismatch). The
  middle-ground profile knobs are KEPT as-is — only the port/served-name part
  of the experiment was rolled back.

## 2026-09-18 — async A/B (user-requested): no measurable delta at seqs 4; ASYNC=1 kept

One-knob A/B, seqs 4 untouched, identical bench both sides (warm-up
request; c1 tg1024 ×2; c2 = 2× tg512 ×2 — the lane's validated cap-2
shape, since the DFlash2 ≥3 cliff would dominate a c4 bench;
stream:false, real completion-token counts). Cold-cache boots
(drop_caches ritual, ~26 min each). Both boots healthy, KV pool
identical (1,027,894 tokens), zero errors.

| shape | ASYNC=1 (boot-1) | ASYNC=0 (boot-2) |
|---|---|---|
| c1 full-1024 | 21.9 / 24.8 tok/s | 23.7 / 27.2 tok/s |
| c2 aggregate (2 streams) | 35.6 / 36.8 tok/s | 34.5 / 32.4 tok/s |

Verdict: **within noise** — c1 favors OFF by ~10% (inside the ±10%
content-acceptance swing measured on the vision lane the same day), c2
favors ON by ~8%, directions oppose between shapes. No reason to deviate
from 0rand's upstream default (ASYNC=1). First measured async receipt
for this recipe (the knob was upstream-inherited, previously untested).

Ops note: boot-2's first attempt died `DistStoreError: 1/2 clients` —
orchestration error (worker reached NCCL handshake while the head's old
container was still tearing down), NOT config; retried clean with the
down-both → up-worker → up-head order.
