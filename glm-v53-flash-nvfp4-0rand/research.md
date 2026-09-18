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
