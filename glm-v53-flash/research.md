# GLM-5.3-Flash — Research & Handoff Notes

**Status: launched and measured on this cluster 2026-08-28** (see the README
audit trail: MTP3 lane PASS, 23-28 tok/s single-stream). Everything below was
gathered 2026-08-28 (model released 2026-08-26) from the NVIDIA Developer
Forum, GitHub, Docker Hub, and HuggingFace (sizes measured from safetensors
index metadata, not README claims). It is the context for `README.md` plus
the watch-list for future improvement passes.

Quick links: recipe `README.md` · primary upstream
[tonyd2wild/GLM-5.3-Flash-NVFP4-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-2x-DGX-Spark) ·
forum threads 381433 / 381429.

---

## 0. 2026-08-28 upstream sync — DFlash2, template fix, RedHatAI (adopted)

Upstream moved: the repo was renamed to
`tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark` and added **DFlash2** —
inco.ai's block-diffusion drafter (vLLM PR #52816 port) — **proven at TP2 on
our exact lane** (fp8 KV, marlin, block-size 2304): 46.9 tok/s single-stream
at 74.1% acceptance (2.15x MTP-4), 56.2 tok/s aggregate @ C5, zero failures;
structured/agentic output 54-61 tok/s (~2.5-2.8x). The drafter costs zero KV
pool (slot-shares the MLA tensors). "dspark2" (as seen on the forum) is this —
not a separate method, and NOT TP4-only: TP2 was the first-proven lane; the
TP4 flagship (68.5 tok/s @ 1M ctx) lives in a sibling repo.

**Adopted (this sync):**
1. Patch layer 9 — the 4-file DFlash2 overlay vendored into
   `patches/overlay-dflash2/` (FROM the v8-equivalent = our stack), inert
   unless `--speculative-config method "dflash"` is selected. Opt-in via
   `DFLASH2=1` (drafter `incoai/GLM-5.3-Flash-DFlash2` rev `7d74cdd`, 2.34 GB,
   **CC-BY-NC-ND-4.0 non-commercial** — MTP3 stays the license-free default).
   `num_speculative_tokens` must be 7 (block 8 - 1); KV pin auto-drops to
   3,221,225,472 for concurrency headroom (upstream shipping value).
2. Patch layer 10 — chat template fix (`53912b4`): the shipped template
   ignored `enable_thinking` (structurally always-on). Vendored fixed
   `chat_template_mm.jinja` + `--chat-template`; `THINKING` env is now a real
   toggle (default true = pre-fix behavior, all-recipes parity).
3. `tools/fleet_watchdog.sh` — adapted for our 2-node compose layout;
   probes `/health` NOT `/v1/models` (200 with a dead engine); orchestrated
   worker-first relaunch (vLLM v1 cannot recover a dead engine core).
4. Verified our `sparse_attn_indexer_kpool.py` gate matches upstream
   `a5c4b19` (multi_processor_count >= 78 -> persistent_topk, else
   top_k_per_row_decode) — no change needed.

**Cautions:** upstream issue #7 (OPEN) — another 2x GB10 pair with an overlay
built FROM v9/InstantTensor gets only 28-31 tok/s @ 0.35 acceptance; our stack
is FROM the stable v8-equivalent, but treat 46.9 as unmeasured-here until
benchmarked. DFlash2 on the NVFP4-KV lane is partial (chunked prefill >3K
kills rank-0) — stay on fp8 KV. First inference after enabling is ~10 tok/s
(drafter JIT), measure warm. Acceptance ~0.15 = broken aux capture.

**Second sync same day (16:02Z): TP2 KV ceiling raised — then WITHDRAWN.**
Upstream walked the `--kv-cache-memory` pin up with DFlash2 attached,
watchdog-free: 10 GiB killed warmup, 7.5 GiB served then died on the FIRST
real request, and **7 GiB = 727,583 tokens** seemed to survive serving + a
500-token generation. Our compose adopted that pin. **Then upstream corrected
itself the same evening (18:45Z, commit 53853387): the 727,583 figure is
withdrawn** — pinning `--kv-cache-memory` makes vLLM skip subtracting the
measured activation peak (gpu_worker.py:475-495), so the pool allocates,
warms, answers a SHORT prompt, then **dies on the first long request**;
reproduced at 7.5 GiB, 300K ctx, 700K ctx and 12 GiB (locked the node), and
no log of the 727,583 measurement survived. **Operationally: let vLLM's
profiler size the pool; profiler-verified figures at 262K context are
581,040 tokens with DFlash2 (survived a 28,818-token prompt) and 965,166
without a drafter.** Our compose's DFLASH2=1 pin is back to upstream's
shipped 3221225472 (3 GiB). MTP lane stays at 4.14 GiB (the MTP draft head
costs ~5 GB; the DFlash2 drafter slot-shares MLA tensors so costs zero KV
pool tokens, but ~4.8 GiB of KV headroom — a real trade: +91% decode for
-40% pool). Also in this sync: the repo's "four defects" commit (3a2b7930)
is docs-only (bind-mount + /health-polling instructions) — our recipe already
handles both by baking the top-k fix into the image and probing /health; and
a tuning note that `enable_thinking: false` buys +8% acceptance but emits
untagged reasoning-prose into `content` (README tool-calling notes updated;
THINKING default stays true).

**Third sync same day (18:48Z): docs/OPEN-PROBLEMS.md.** Upstream opened a
failure catalog with next probes. Notable for us: (1) the TP worker rank
profiles 4-5 GiB LESS KV headroom than the head (min-across-ranks binds the
pool; not a config error, looks upstream-vLLM); (2) InstantTensor
direct-I/O loads are 15x faster but silently unstable multi-node (rank dies
~1 min post-load in 4/4 TP2 boots — we already avoid it); (3) UVM driver
livelock under memory pressure — `vm.swappiness=0` mandatory on every node
(does NOT survive reboot; /etc/sysctl.d/), plus `swapoff -a && swapon -a`
before launch; (4) vision is not speculated on either drafter (text-only
draft inputs); (5) CUDA-graph trap vllm#53030 can pin acceptance at exactly
1.00 (we run --enforce-eager, unaffected); (6) temp-0 is free throughput
(+13-21%) via the rejection sampler's exact top-1 match. Condensed into the
README's DFlash2 section.

**RedHatAI/GLM-5.3-Flash-NVFP4 vs LibertAIDAI (user question).** Verdict: keep
LibertAIDAI. RedHatAI is W4A4 (compressed-tensors, FP4 weights AND FP4
activations, LLM Compressor); LibertAIDAI is weight-only (ModelOpt, FP4
experts / BF16 everything else). Activation-FP4 buys zero memory (expert
weights ~175 GB either way) and is a datacenter speed lever with quality risk
on this KDA + sparse-MLA arch. Zero 2x GB10 deployments use RedHatAI (all 8
community recipes pin LibertAIDAI; RedHatAI's own recipe is TP4
datacenter-only); format mismatch means an untested load path on sm_121.
RedHatAI publishes evals (GPQA-D 90.57, AIME25 86.67) while LibertAIDAI
publishes round-trip cosine 0.99665 + every GB10 pitfall. Full report:
`/tmp/redhatai-report.md` (2026-08-28 snapshot).
**⚠️ SUPERSEDED 2026-08-30 — see section 0b: ModelOpt builds emit corrupted
token IDs (vllm#54150); RedHatAI is now the pinned checkpoint.**


## 0b. 2026-08-30 checkpoint switch — RedHatAI compressed-tensors (adopted, research-only)

Upstream tonyd2wild switched the recipe's default checkpoint from
`LibertAIDAI/GLM-5.3-Flash-NVFP4` (ModelOpt weight-only W4A16) to
`RedHatAI/GLM-5.3-Flash-NVFP4` (compressed-tensors W4A4) in commits
`7497e96b` + `5a4df199` (2026-08-29/30). Reason: **ModelOpt NVFP4 builds of
GLM-5.3-Flash emit intermittent corrupted token IDs** —
[vllm#54150](https://github.com/vllm-project/vllm/issues/54150) (first seen
SM120), confirmed on **2× GB10 / SM121 TP2 DFlash2 — our exact lane — by
ajclark in upstream issue #10**. The failure is quiet (HTTP 200, normal
throughput, English mostly fine) but a corrupted token inside a tool-call
block desyncs the parser and can spiral into repetition lock. The
compressed-tensors conversion of the same model is clean from the same image.

**Adopted 2026-08-30:** pin `RedHatAI/GLM-5.3-Flash-NVFP4` rev `36c184c6`.
Drop-in: same flags (`--moe-backend marlin`, fp8 KV, DFlash2 k=7); marlin
dequantizes to bf16 and never consumes the activation scale, so the W4A16→W4A4
difference likely costs little on our backend. Trade-offs: expect a few points
lower on hard reasoning; RedHatAI ships `chat_template.jinja` but NOT
`chat_template_mm.jinja` (our `TEMPLATE_FIX` layer covers it); ~2× faster load
(11 shards vs 120); 184 GiB. This **supersedes the 08-28 "keep LibertAIDAI"
verdict above**, which was written before the corruption evidence existed.
**Not yet measured here** — next bounce must verify: boot + `/v1/models`, the
standing `glm45` vs `deepseek_r1` reasoning-parser item, and a tool-call
stress test to confirm the corruption class is gone.

**2026-08-30 recipe-update verification pass (second agent review).** Every
checkable claim in the switch verified against sources: upstream commits
`7497e96b`/`5a4df199` exist and match; vllm#54150 title/contents match
(ModelOpt NVFP4 emits invalid-UTF-8 byte tokens on SM120, compressed-tensors
clean); upstream issue #10 verified — filed by ajclark, with the 2× GB10
TP2 DFlash2 A/B (identical image/flags/KV pool; LibertAIDAI 3–6 U+FFFD per
run vs RedHatAI 0/0/0) contributed by **todoriri** in the comments (the
attribution in the audit trail above was ajclark's report + todoriri's
measurement). RedHatAI rev `36c184c6` live on HF (lastModified 2026-08-28,
compressed-tensors tag, 10 model shards + `model_mtp.safetensors`);
`chat_template_mm.jinja` indeed absent from the repo (only
`chat_template.jinja`) and our `patches/chat_template_mm.jinja` →
`/opt/glm53/chat_template_mm.jinja` COPY + `--chat-template` wiring covers
the vision template (upstream README warns vision 500s without it).
`--moe-backend marlin` dequantizes to bf16 so W4A4 vs W4A16 activations
should not bite on our backend — matches upstream's "drop-in" claim.
Compose/docs leftovers: none (all LibertAIDAI refs that remain are
historical context). Attribution note recorded; no code changes needed.

Also this pass: upstream #11/#12 (tmooch, closed/superseded-open PR)
reports TP2 DFlash2 tuning — KV pin 3→6 GiB (678,661-token pool, kills 6
preemptions), k=7→5, `--max-num-batched-tokens` 8192 (C6 aggregate 47.7→
60.6→64.9 tok/s, −26% wall) with a robust 84-request harness. **Watch, not
adopted**: unmerged upstream PR; k=5 contradicts our validated k=7 config;
KV pin contradicts our TP2 no-pin guidance; batched-tokens 8192 contradicts
our agent-serving 2048 profile. Revisit if tonyd2wild merges it. Forum
general sweep: no new GLM-relevant topics since 2026-08-29 15:45Z review.

## 0a. 2026-08-29 upstream review — pins moved, KV-pin philosophy, EXL3 lane (review pass)

Full review pass against all sources: tonyd2wild DFlash2 repo (new), two EXL3-lane
repos (new to watch list), NVIDIA forum general sweep, Docker Hub, HF revisions.
**No invariant broken; no emergency.** Findings ranked:

### Checkpoint pin `aa28e1f5` has 3 newer upstream commits (RE-PIN CANDIDATE)
`LibertAIDAI/GLM-5.3-Flash-NVFP4` HEAD is `357b45cc` (2026-08-28T16:52Z):
1. `b2abefa1` — documents the **ModelOpt NVFP4 MoE activation-scale trap**
   (vllm#54189): weight-only checkpoint + uninitialized `input_scale` ⇒ alpha = 0
   ⇒ degenerate one-token output with NO error. **Our `--moe-backend marlin` does
   not read the activation scale — we are not in the failure path** (card says so
   explicitly). Our measured 74% acceptance corroborates a healthy lane.
2. `24c04b86` — **`--reasoning-parser glm45` is SGLang-only; vLLM needs
   `deepseek_r1`** (vLLM's glm45 alias expects prompt-side `-describedby`
   mismatch and silently discards replies). Our compose ships
   `REASONING_PARSER=glm45`, but our measured load tests show correct
   content/reasoning split — our 10-patch image diverges from stock vLLM. VERIFY
   on next bounce (one curl with `reasoning_content` in the response) before
   changing; if broken, switch to `deepseek_r1`.
3. `357b45cc` — **weight change**: adds per-expert `input_scale=1.0` (the
   vllm#54189 fix, for flashinfer_cutlass users). Marlin users unaffected.
   Re-pin candidate for the flashinfer_cutlass option only; marlin lane can
   stay on `aa28e1f5` until a re-bounce.

### Drafter pin `7d74cdd` has a newer "Checkpoint update" (RE-PIN CANDIDATE)
`incoai/GLM-5.3-Flash-DFlash2` HEAD `dc77ff1c` (2026-08-28T21:37Z, "Checkpoint
update") — file-level diff not exposed by the HF API; same single-safetensors
shape (1.17B BF16). Treat as weight-or-card unknown. **Re-pin candidate
requiring acceptance re-test** ( Drafter sha in tonyd2wild issue #7 was
`8931dc52...` — different prefix, possibly the dc77ff1c artifact; unresolved).

### KV-pin philosophy: upstream now says DO NOT PIN (WATCH, experiment candidate)
tonyd2wild withdrew the 7 GiB ceiling AND now recommends never pinning
`--kv-cache-memory` at all: pinned pools never subtract measured activation peak
(`gpu_worker.py:475-495`), so the first long prompt can OOM the engine. Profiler-
sized pool at our 262K ctx = **581,040 tokens** with DFlash2 (vs our 3 GiB pin).
Our 3 GiB pin is deliberately conservative (leaves activation headroom) and
issue #7 shows pin value does not affect acceptance. **Document as deliberate
deviation; candidate experiment: drop the pin, re-measure pool size + 28K-prompt
stability.** Corroborating: forum 381755 (743B TP4) still recommends pinning for
deterministic boots — practice is split, not settled.

### Forum general sweep (new threads since 08-27)
- **381703** (SGLang-path DFlash2): same model+drafter on SGLang; 29.4 C1 bf16 KV;
  fp8 KV now works via sglang PR #36904; mamba-cache cap fix → 83.5 aggregate @ c12.
  **Caution for us: silent worker-node death on >~32k-token prefills (no
  traceback, rank just dies)** — treat >32k prompts as unverified on our lane
  until tested. Watch.
- **381541** (closest peer: NVFP4 TP2 compose, same day-0 image + patch chain):
  512K ctx + FP8 KV + MTP-4, 9 GiB pin, 440K needle-exact. Confirms worker-first,
  `--language-model-only` is load-bearing at 512K (mm processor adds ~15.7 GiB),
  sparse-MLA indexer guard needed at 512K, batched-tokens 8192→4096 at those
  shapes. Watch (applies if we ever raise context).
- **381755** (GLM-5.3 743B TP4): drop_caches-eats-UMA corroboration, degraded
  clocks after crash (−25% TP4), DFlash2 next. Watch.
- 381534 (TP3), 381543 (TP4 FP8), 381350 (23 tok/s NVFP4 measurement matches ours):
  corroborations, no action.

### EXL3-lane repos (new watch-list entries; quant tricks NOT transferable)
- **Entrpi/glm-5.3-flash-exl3-2x-spark**: 33-35 prose / 72 structured c1;
  per-workload acceptance tables; memlog.sh 1 Hz memory-floor sampler (adopt:
  cheap, guards our KV floor); MTP-4 fallback rollback lane (watch); relaunch
  ordering fix confirms worker-first.
- **MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks**: 62.9 structured / 26.9 prose;
  **correctness canaries for DFlash2 lanes**: grammar/FSM stalls under concurrent
  structured output (#19), tool-call blank-args under concurrent DFlash2+prefix
  caching (#10, solo not reproducible), TRITON_ATTN acceptance collapse (keep
  FLASH_ATTN for draft attention — our config already does), spec-window
  reasoning-gate race. **Wide-propose/narrow-verify (7 propose / 3 verify) =
  +21.6% prose** (PR #12; needs vLLM upstream PRs #52559/#53542). Watch.
- KLD panel (brandonmusic HF discussion): NVFP4 0.0605 vs EXL3-4bpw 0.0246 vs
  FP8 0.0206 — only quant-quality datapoint suggesting an NVFP4 gap. Watch.

### Image/HF status
- Docker Hub `vllm/vllm-openai`: **no new glm53-flash tags** since 2026-08-26;
  base digest `905c0293` unchanged. Nightly churn only. No action.
- tonyd2wild image is now also on GHCR (`ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2`,
  `4def0ef6...`) — reference only; our local build matches its base.

**Actions from this pass:** (1) ~~verify reasoning-parser behavior on next
bounce~~ **VERIFIED 2026-08-29 on the live server**: `glm45` on our patched
image splits correctly under DFlash2 (`reasoning` populated, clean `content`,
41 ct) — the card's stock-vLLM failure mode does not apply to our 10-patch
image; keep `glm45`. (2) drafter re-pin + acceptance re-test when convenient;
(3) checkpoint re-pin only if we ever leave marlin; (4) document 3-GiB KV pin
as deliberate deviation from upstream's no-pin guidance; (5) long-prompt
(>32k) test before trusting long-context serving. None block current service.

### STANDING GOAL: raise context to 500K — but NOT at the cost of vision (user directive, 2026-08-29)
The user wants this recipe eventually at ~500K context while **keeping
multimodal (vision) enabled** — `--language-model-only` is a dealbreaker, which
rules out the straightforward 512K profile today (forum 381541: 512K + fp8 KV +
9 GiB pin + MTP-4, but load-bearing on `--language-model-only` because the mm
processor costs ~15.7 GiB). Until that memory can be found elsewhere, we stay
at 262K + DFlash2 (current validated config).

Rough path when enabling knobs land (config-only, ~half a day, one bounce):
`MAX_MODEL_LEN=524288`, `KV_CACHE_MEMORY` ≈ 9 GiB/rank (fp8 ≈ 122K tok/GiB),
`MAX_NUM_BATCHED_TOKENS` 8192 → 4096, GMU 0.85 unchanged — with vision kept ON
the pin must shrink to fit under GMU 0.85, so the open question is whether a
smaller pool (e.g. 6-7 GiB ≈ 750-850K tok fp8) still serves 500K at low
concurrency, or whether the ~15.7 GiB mm cost can be recovered another way
(quantized mm tower, kernel savings, future upstream improvements).

**Watch for:** (1) any upstream fix shrinking the multimodal processor's
~15.7 GiB resident cost (or an offload/quantized-mm option); (2) DFlash2
validated at >262K on the NVFP4 lane (381541 ran MTP-4 only; the >32k-prefill
silent-death caution from 381703 is unresolved); (3) tonyd2wild / peer recipes
shipping 512K with vision intact; (4) our own >32k prefill test results. When
any of these land, re-run the numbers and reconsider.

**2026-08-29 15:45Z sync — TP4 KV-pool breakthrough (same day, watch for TP2 port).**
tonyd2wild's 4-node repo (`GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark`, commit
`d26be7df` 15:12Z) raised its default KV pin 16 → 24 GiB/rank = 3,895,606 fp8
tokens (+54.8%), gate-passed (2 deep decodes @ ~41K ctx, 3× concurrent 32,879-tok
prefills, vision, /health 200 throughout; head-rank residual 15 GiB). The key
discovery: the long-standing "phantom KV backing" above 16 GiB/rank was **the
page cache** — a *threshold-triggered* flusher can sit below its threshold and
still starve the NVRM allocator; an **unconditional flusher, running for the
entire boot on every node, started before the launcher** took the same 24 GiB
pin from boot-death to gate-pass. Our cluster runs the ritual flusher +
`cache_flusher.sh` but NOT an unconditional whole-boot flusher — this is the
missing piece for any KV-pin raise (our standing 5.5 GiB upgrade idea, and any
future 500K push).

Same repo, `37385c56` (08-27): fp8 vs NVFP4 KV head-to-head at equal 32 GiB/rank
TP4 — NVFP4 KV 6.65M tok vs fp8 5.03M (1.32×) but decode ~37 vs ~55 tok/s
(fp8 ~1.5× faster) and prefill 1449 vs 3530 tok/s. Confirms our earlier read:
NVFP4 KV trades speed for density; fp8 stays our lane.

TP2 repo unchanged (no-pin guidance stands for TP2; the TP2 head-rank worker
4–5 GiB KV-headroom asymmetry remains unexplained upstream). Relevance to our
500K-with-vision goal: the unconditional flusher removes one blocker (bigger
safe pin without phantom backing), but the ~15.7 GiB mm-processor cost still
binds at TP2 — vision remains the gating constraint, as documented above.


## 1. Why NVFP4 (and not the official BF16/FP8 releases)

The official `zai-org/GLM-5.3-Flash` repos are **FP8 (main)** and **BF16** —
and neither fits 2× DGX Spark. The community NVFP4 quants are the only option.
Sizes measured from `model.safetensors.index.json → total_size`.

| checkpoint | size | per node @TP2 | vs ~121 GB usable/node | fits? |
|---|---:|---:|---:|---|
| official BF16 (`zai-org/GLM-5.3-Flash-BF16`) | 642.6 GB | 321.3 GB | −200 GB (−165%) | **NO** |
| official FP8 (`zai-org/GLM-5.3-Flash` main) | 328.3 GB | 164.2 GB | −43 GB (−36%) | **NO** |
| `RedHatAI/GLM-5.3-Flash-NVFP4` (W+A FP4) | 190.2 GB | 95.1 GB | +25.9 GB (+21%) | **YES** |
| **`LibertAIDAI/GLM-5.3-Flash-NVFP4` (W4A16)** | **194.6 GB** | **97.3 GB** | **+23.7 GB (+20%)** | **YES** |
| `local-inference-lab/...-NVFP4` (ModelOpt mixed) | 198.0 GB | 99.0 GB | +22.0 GB | YES |
| `vcruz305` / `axiomofmind` / `bullerwins` NVFP4 | 205.1 GB | 102.5 GB | +18.5 GB | YES |

Mechanism: the routed experts are ~97% of the 321B params (311.7B), so
4-bit-quantizing experts alone cuts ~623 GB → ~175 GB while keeping the
outlier-sensitive parts (attention/indexer/vision/MTP/embeddings) in BF16.
GB10 has native FP4 tensor-core kernels, so NVFP4 cuts both resident memory
and bytes/token on the bandwidth-bound MoE decode.

**Recipe choice: `LibertAIDAI/GLM-5.3-Flash-NVFP4`** — the only quant with
published 2×GB10 validation (SGLang-verified by the quant author; used by
every working vLLM 2× recipe). Weight-only (W4A16) keeps activations exact.
Revision `aa28e1f54130286c95fee10d0705c74ce8743734` (this recipe's pin).

**Headroom caveats (define what this recipe can and cannot do):**
- ~20% headroom must cover KV cache + the KDA linear-attention state (34
  recurrent states per request — often the binding constraint, not KV) +
  activations + engine overhead.
- **1M context is NOT achievable on 2 nodes.** tonyd2wild's TP2 ceiling is
  262,144 ctx; the model-native 1M requires TP4 (4 nodes, 35.7 tok/s).
- LibertAI's SGLang envelope on 2×GB10: `--context-length 65536
  --max-running-requests 2`; `8 @ 131072` fails (mamba/KDA cache too small).
  vLLM TP2 is better (262K @ 6 seqs / 507-672K KV) but still not 1M.
- Other NVFP4 quants exist (RedHatAI = activations also FP4, no GB10
  validation; local-inference-lab is misleadingly named — actually BF16
  mirror per engine research; vcruz305 quantized on a single Spark).
- `incoai/GLM-5.3-Flash-DFlash2` (0.4B block-diffusion drafter) is faster
  than MTP and now **works at TP2 on this stack** (see section 0) — adopted
  as an opt-in lane gated on its **CC BY-NC-ND (non-commercial)** license;
  native MTP (BF16 head, in-checkpoint) remains the license-free default.

### Revisions to pin (all public HF repos)

| artifact | repo | sha |
|---|---|---|
| Official FP8 (default) | `zai-org/GLM-5.3-Flash` | `04c4e9e95c5da8862dced7e5056455116f83a7e0` |
| Official BF16 | `zai-org/GLM-5.3-Flash-BF16` | `f12e0fe1f6b2ea274c11a569582edfd99d993c5e` |
| **NVFP4 (this recipe)** | `LibertAIDAI/GLM-5.3-Flash-NVFP4` | `aa28e1f54130286c95fee10d0705c74ce8743734` |
| DFlash2 draft (non-commercial) | `incoai/GLM-5.3-Flash-DFlash2` | `7d74cdd881ed7e32c31175984a67823127b66cfe` |

License: official model is **plain MIT** (no hosting restrictions — unlike
older GLM modified-MIT licenses). DFlash2 drafter is the exception
(CC BY-NC-ND).

---

## 2. Upstream evaluation (tonyd2wild) + what was adopted

`tonyd2wild/GLM-5.3-Flash-NVFP4-2x-DGX-Spark` is the world-first 2× DGX Spark
GLM-5.3 deployment (51★). Every other working 2× recipe forks/derives from it.
It uses **vLLM** (day-0 per-model image + an 8-stage patched Dockerfile ladder
v1→v8, with v9 = experimental InstantTensor loader). The repo's committed
launcher references `radixark/vllm-glm53-flash:sm121-v9` which is **NOT on
Docker Hub (404)** — unreproducible as-is, and v9 is unstable in TP2 per the
repo's own README. This recipe therefore builds the patched image locally from
the public base, consolidating v1→v8 into one Dockerfile.

**Adopted (all eight patch layers):**
1. SM121 NoPE-MLA backend (v1) — extends vLLM's SM90 NoPE sparse-MLA backend
   (plain bf16 cache) to capability 12 + FA2 off-Hopper. The only stock SM12x
   backend forces DeepSeek's packed `fp8_ds_mla` layout (`pe_dim=64`), which
   GLM's NoPE MLA (`qk_rope_head_dim=0`) cannot use → assert death in warmup.
2. FlashInfer 0.6.18 nightly (v3) — 0.6.17's FA2 MLA scheduler NaNs on
   64–256-row batches on SM121 (normal prompt sizes).
3. NCCL 2.30.7 re-pin (v4) — nightly silently downgrades to 2.29.7, which
   fails `ncclCommInitRank` on the Spark IB fabric.
4. cutlass-dsl 4.6.2 re-pin (v5) — nightly leaves a mixed 4.7.0/4.6.2 install
   that ICEs the CuTeDSL warmup.
5. PDL gated off SM12x (v6) — Programmatic Dependent Launch races the KDA
   state Triton kernels on unvalidated SM121 (boots NaN or don't by timing).
6. Indexer top-k hardening (v7, `patch_v7.py`) — kpool top-k dest
   `torch.empty` → `torch.full(-1)` + pool-id clamp (uninitialized ids → NaN
   lottery).
7. fp8 KV cache on SM12x (v8, `patch_v8_fp8.py`) — cap the FA2 fp8
   CTA_TILE_KV (Hopper's forced 32 over-requests GB10's ~101KB smem) + relax
   the FlashInfer fp8 MLA gate. First fp8 KV for a NoPE-MLA model on consumer
   Blackwell.
8. persistent_topk SM121 gate (`sparse_attn_indexer_kpool.py` overlay) — the
   DSA indexer's persistent_topk hard-crashes ANY decode past ~24K context on
   GB10 (FilteredTopK fallback needs 128KB smem/block; SM121 has ~101KB).
   Routes small-SM parts to `top_k_per_row_decode`. This fix is NOT in the
   Dockerfile ladder — it came from the later crash forensics and is deployed
   as a full-file overlay.

**Deliberately NOT adopted:**
- **v9 / InstantTensor loader** — 15× faster loads but a rank dies silently
  ~60-90 s after load in ALL four TP2 test boots (also downgrades NCCL;
  re-pinned but still unstable). Re-test when upstream moves.
- **v2 (NaN debug hooks)** — debug-only tooling, not production.
- tonyd2wild's committed launcher values — the committed `launch-glm53-vllm-tp2.sh`
  still uses v9 + `--load-format instanttensor` + `--kv-cache-memory 4445787956`,
  contradicting the README's own stability notes (stable = v8, KV up to
  5905580032 with local weights). The README + docs are authoritative; the
  launcher is stale.

### Other 2× Spark repos (all vLLM unless noted) — worth checking for tuning

| repo | approach | measured (TP2) |
|---|---|---|
| `kingjones30/GLM-5.3-Flash-2x-DGX-Spark` | **stock image + zero-pad rope mod** (`VLLM_MLA_NOPE_PAD_ROPE=1`), no rebuild; MTP-5, marlin, `--language-model-only`, `fp8_ds_mla` KV | 24.7 code / 30.3 structured / 19.6 prose; 738K KV @ GMU 0.85; ~1150 tok/s prefill @200K |
| `chishiki37/glm-5.3-flash-nvfp4-dgx-spark` | auto-search over KV × MTP depth on tonyd2wild base | **TP2 winner: fp8 KV 4.14 GiB + MTP3 = 28.3 tok/s (+30% vs MTP4)**; C8 agg 73.4 |
| `sfxnz/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark` | local v8 build; MTP-4, fp8 KV 4.14 GiB | 25.6 C1 / 39.3 C2 / 64.9 C4 |
| `drowzeys/keys-vLLm...-NVFP4KV-1M` | Zero-RoPE shim + Luke Alonso b12x `B12X_MLA_SPARSE` + **NVFP4 packed KV** | 1M ctx on 2× TP2 (1.22M-token pool), 21.2–26.6 tok/s; C8 agg 84/75 |
| `amasu/glm53-flash-cluster` | host-agnostic docker-compose TP2, `glm53:v8` | research-grade, reproduces the patches |
| `MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark` | Ray-orchestrated TP2, multimodal | — |
| `Tutanka01/glm5.3-flash-2x-dgx-spark-nvfp4` | **SGLang** TP2 | 29.0 tok/s with MTP (C4 41.8; TTFT degrades under speculation) |
| `0xSero/glm-5.3-flash-sglang-sm121` | pinned SGLang runtime (DSA flashinfer_sparse_mla, FP8 KV, NEXTN MTP5) | 131K ctx / 2 reqs profile |
| `jack6464` (forum) | Marlin + MTP5 | 23.08 C1 / 60.68 C5, tool-eval 86/100 |

**Key insight from drowzeys:** NVFP4 packed KV (`nvfp4_ds_mla`, 368 B/tok/layer)
is 1.8× denser than fp8 (656 B) but ~33% slower decode on the b12x path — fp8
KV is the daily driver, NVFP4 KV is the capacity flex (that is how anyone gets
1M ctx on 2 nodes). Requires the zero-pad rope shim + b12x backend, neither of
which this recipe ships (yet).

---

## 3. Engine decision: vLLM (not SGLang) — and the SGLang alternative

**Chosen: vLLM** (day-0 image + SM121 patch stack). Reasons:
- 8+ working/measured vLLM 2× recipes vs thinner SGLang GB10 validation.
- SGLang has a critical open bug for this recipe's shape: **#36653 NEXTN/MTP
  spec-decode fails to load MTP MoE weights under TP>1** (directly blocks the
  main speed lever at TP2), plus #36550 (abort past 262K prefill), #36596
  (NVFP4 crash at load), #36597 (EP>1 scale slicing), #36669 (thinking
  degenerates to `!` under multi-tool prompts).
- The "vLLM produces wrong output on sm_121" claim in the LibertAI model card
  refers to the **stock** image (NoPE-MLA `pe_dim=64` assert) — the patch
  stack fixes exactly that. The card is SGLang-blessed because its author
  shipped the SGLang recipe, not because vLLM can't work.
- The repo's existing recipes (DSv4) are vLLM — same flag/overlay vocabulary.

**SGLang alternative (documented, if we ever want it):** base
`lmsysorg/sglang:glm-5.3-flash-arm64`, flags from the LibertAI card:
`--attention-backend dsa --dsa-prefill-backend tilelang --dsa-decode-backend
tilelang --moe-runner-backend flashinfer_cutlass --kv-cache-dtype bfloat16
--disable-shared-experts-fusion --reasoning-parser glm45 --tool-call-parser
glm47 --mem-fraction-static 0.84 --context-length 65536 --max-running-requests 2`
plus a **TileLang shared-memory patch** (stock requests 169,984 B > GB10's
101,376 B opt-in; set `block_I=32, num_stages=1, threads=128`). Envelope is
much smaller (64K ctx @ 2 reqs). 0xSero's pinned runtime
(`ghcr.io/0xsero/glm-5.3-flash-sglang-sm121`) is the audited variant.

**Spec-decode:** native MTP (BF16, layer 45, license-free) is the only usable
path. **MTP3 is the measured TP2 winner** (chishiki37: 28.3 tok/s vs 21.8 for
MTP4; per-position acceptance [0.83, 0.59, 0.34, 0.18] — draft positions ≥3
mostly pay verify cost). MTP5 can gibberish (forum 353069). DFlash2 has since landed for GLM-5.3 at
TP2 and is adopted as an opt-in lane (section 0) — 2.15x single-stream over
MTP-4 upstream.

---

## 4. Where to watch for updates (future improvement passes)

### 4.1 NVIDIA Developer Forum (highest-value source)
Discourse JSON API: `https://forums.developer.nvidia.com/<path>.json`.

- **Threads that matter:**
  - `381433` — kingjones30's GLM-5.3 2× day-0 thread. Contains the **silent
    FP4 MoE corruption** finding (`--moe-backend marlin` fix — generalizes to
    DeepSeek-V4 0731) and the "shared memory broadcast" benign-boot note.
  - `381429` — tonyd2wild's GLM-5.3 2× thread.
  - `378824` — tonyd2wild's DSv4 lineage (patch conventions).
- **Search terms every review:** `glm 5.3`, `glm53`, `glm 5.3 flash dgx spark`,
  `glm nvfp4 sm121`, `glm flash marlin`, `moe-backend marlin sm121`, plus a
  sweep of the newest topics in category `721`/`723` (recipe-relevant info
  lands in threads whose titles don't contain "glm").
- **What to look for:** DSpark/DFlash2 for GLM-5.3 (forum projects 50–70
  tok/s), NVFP4-KV + zero-pad rope upstreaming, new image tags, regression
  reports matching this cluster's config, and the 353069 MTP gibberish thread.

### 4.2 GitHub
- **tonyd2wild/GLM-5.3-Flash-NVFP4-2x-DGX-Spark** — the upstream. Watch for:
  the launcher/README reconciliation, InstantTensor TP2 stabilization (v9→v10),
  new patches, the TP4 NVFP4-KV work. Its `probes/` (nan bisect, kernel probes)
  and `docs/` are the debugging kit for any day-0 model.
- **kingjones30, drowzeys, chishiki37, sfxnz, amasu** — the vLLM tuning lane.
  Especially drowzeys' NVFP4-KV + b12x + 1M-on-TP2 work (the zero-pad rope
  shim — a potential no-rebuild alternative to patch layer 1).
- **vllm-project/vllm:** PR **#53906** (GLM-5.3-Flash support, unmerged) —
  when it lands in a release, the patch ladder port is "mechanical" (guards
  fail loudly). Issue **#54062** (latest nightly fails even on B200 —
  `Glm5NextTextLinearAttention not supported`). Also relevant: the DSv4
  family PRs (#53896/#53899/#54070) if we ever consolidate.
- **sgl-project/sglang:** support PR **#36507**; TP>1 MTP bug **#36653**;
  #36550, #36596, #36597, #36669, #36711 (all open — re-check before any
  SGLang port).
- **eugr/spark-vllm-docker / styles01/sparkrun-recipes / brainchillz:** no
  GLM-5.3 sparkrun YAML exists — a future sparkrun port would be net-new
  (kingjones30 proves the eugr-framework + mods path; convert this compose).
- **Entrpi/glm-5.3-flash-exl3-2x-spark** — EXL3 4bpw + DFlash2 lane (not
  vLLM/NVFP4): one-shot installer, 33–72 tok/s c1, 900k+ context, vision.
  Different quant/engine stack, but useful for DFlash2 tuning ideas,
  long-context tricks, and cross-checking acceptance/throughput numbers.
- **Libertai/glm53-flash-vllm-gb10** — the LibertAI Labs repo *behind* the
  `LibertAIDAI/GLM-5.3-Flash-NVFP4` checkpoint we serve (the model card's
  "vLLM wrong output on sm_121" note is this work; see §3). Root causes
  documented there: (1) no MLA path for NoPE on sm_120/121 — fixed with a
  hand-written sparse-MLA CUDA kernel shipped as `vllm.general_plugins` entry
  points (no vLLM files patched; `VLLM_GLM53_CUDA_SPARSE_MLA=1`); (2)
  uninitialised `w13_input_scale` in `ModelOptNvFp4FusedMoE` zeroing every
  expert output — fixed by `kernel/glm53_sparse_mla/moe_fix.py`
  (`VLLM_GLM53_MOE_INPUT_SCALE=1.0`); filed upstream as vllm#54189. Their
  measured 2×GB10 lane: 24.2 tok/s c1 (MTP-3 + eager + fp8 KV), 88,790 KV
  tokens @ 64K ctx, `--max-num-seqs 2` — broadly consistent with our 23-28.
  Their transferable lesson: CUDA graphs buy ~1% on GB10 (bandwidth-bound) but
  ~500% on 4× RTX PRO 6000 (latency-bound) — never carry a config lever across
  boxes with different bottlenecks. Watch for kernel/backend releases, updates
  to the upstream issue, and any KV-context gains that beat our pin.
- **MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks** — sibling EXL3 2× Spark
  recipe; cross-check for the EXL3 lane and any shared drafter/tooling.
- **tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark** — tonyd2wild's
  NVFP4 + DFlash2 follow-up to the base NVFP4 repo (the DFlash2-on-NVFP4
  lineage closest to our `DFLASH2=1` lane). Watch for new patches, DFlash2
  fixes, KV-pin changes, and issue-report patterns that hit our config.

### 4.3 Images & checkpoints
- **Docker Hub `vllm/vllm-openai`** — `glm53-flash-*` tags. All currently date
  to 2026-08-26 (no newer tags); a rebuilt tag (after #53906 or fixes) would
  justify re-pinning and may make some patches obsolete. Re-check the digest
  `sha256:905c0293…` hasn't moved (digest-pinned in `patches/Dockerfile`).
- **Docker Hub `lmsysorg/sglang`** — `glm-5.3-flash*` tags (SGLang lane).
- **HF `LibertAIDAI/GLM-5.3-Flash-NVFP4`** — check `lastModified`/`siblings`
  for a newer revision; mirror any bump in `MODEL_REVISION` (compose `.env`)
  and re-verify the quant still loads.
- **HF `zai-org/GLM-5.3-Flash*`** — official FP8/BF16 (for reference/TP4).

### 4.4 DFlash2 for GLM-5.3 — landed and adopted (2026-08-28)
TP2-proven upstream (46.9 tok/s C1, 74.1% acceptance); adopted as the
`DFLASH2=1` opt-in lane (section 0). Watch for: issue #7 resolution
(reproduction gap), a commercial-friendly drafter or the DSpark port, and
GLM-5.2-style DFlash1 licensing. Forum projects 50-70 tok/s once settled.

---

## 5. Known improvements / fixes to track (with blockers)

1. **Measure this cluster's actual numbers** (partly done — README audit
   trail has first-boot numbers; MTP3 23-28 tok/s single, 51-62 agg C6).
   Still to record: prefill tok/s, TTFT, and the deep-decode gate (28–32K
   prompt, ≥100 decoded tokens — proves the persistent_topk overlay).
2. **KV upgrade to 5.5 GiB (672K tokens).** Default is 4.14 GiB (3/3 reliable,
   NFS-safe). 5.5 GiB is tonyd2wild's stress-verified local-weights record —
   our cluster has local weights on both ranks, so try `KV_CACHE_MEMORY=5905580032`
   after a stable boot, with the cache-flush ritual. Never above ~6 GiB/rank
   on TP2 (phantom backing / first-touch death).
3. **MTP depth A/B.** Default MTP3 (chishiki37's measured winner). A/B vs MTP4
   on this cluster and record; consider MTP5 only after reading forum 353069.
4. **The zero-pad rope shim (`VLLM_MLA_NOPE_PAD_ROPE=1`, kingjones30/drowzeys).**
   Enables `fp8_ds_mla`/`nvfp4_ds_mla` KV on the STOCK image (no rebuild) and
   is the route to 1M ctx on 2 nodes (drowzeys: NVFP4 KV + b12x = 1.22M-token
   pool). Evaluate as a simplification or capacity upgrade — but note the
   b12x decode penalty (~33% slower for NVFP4 KV) and that it's a different
   kernel path than the SM90-backend port we ship.
5. **`--language-model-only`** — documented knob; A/B whether skipping the
   multimodal processor stabilizes the head rank's memory (tonyd2wild didn't
   need it; kingjones30 did).
6. **InstantTensor loader (v9)** — re-test when upstream moves (multi-node
   direct-I/O). Loads drop 10 min → 40-100 s and it bypasses the page-cache
   KV wall, but it was TP2-unstable in all four test boots.
7. **Thinking default parity with the other recipes (DONE 2026-08-28).** The
   other recipes default `thinking=true` (+ reasoning_effort). Investigation
   on the live cluster found the original `THINKING=false` knob was a no-op:
   the pinned `chat_template.jinja` has **no `enable_thinking` gate** — the
   generation prompt always opens a ` ++)  block, the effort header is always
   emitted (undefined → max), and no-effort requests were observed to
   degenerate (800 tokens, empty reasoning and content). The recipe now pins
   `--default-chat-template-kwargs '{"reasoning_effort": "high"}'`
   (all-recipes parity: thinking on, high, temp 1.0 / top_p 0.95 via the
   checkpoint's own generation_config.json). Verified live: top-level
   OpenAI `reasoning_effort` overrides engage thinking (glm45 parser →
   `reasoning` field), ~330 completion tokens for a short high-effort answer
   vs 41 without. Remaining watch: thinking + structured tools on the vLLM
   day-0 stack (SGLang #36669 degeneration class) — re-run
   `tools/load_test_glm.py` with thinking if agents misbehave. GLM's
   `reasoning_effort` levels are low/high/max (template default max).
8. **Benchmark DFlash2 on this cluster** (adopted 2026-08-28, not yet
   measured here). Gate before trusting 46.9 tok/s: upstream issue #7 shows a
   v9-based overlay pair at 28-31 tok/s @ 0.35 acceptance. Check `/metrics`
   acceptance 0.6-0.8 and the Eagle3 aux-layer boot line. Confirm the DFlash2
   drafter revision fingerprint stays `8931dc522be0aa31…` (issue #7 asked the
   same of upstream).
9. **The NVFP4-KV + b12x lane (drowzeys)** as the path to 1M ctx on 2 nodes —
   the single most valuable capacity upgrade; needs the zero-pad shim.
10. **Re-pin / de-duplicate patches when vLLM #53906 lands.** The patch guards
   (`refusing to patch` on count mismatch) make re-pinning mechanical; keep
   the ladder-through-experiment-lane workflow (tonyd2wild's method).

---

## 6. Log checks after testing (what to look for on first boots)

On **both** head and worker logs (`docker logs glm53-nvfp4`):

### Good markers (recipe working as intended)
- `Loading model weights took ...` and **no** `pe_dim must be 64 for fp8_ds_mla`
  (proves the SM121 NoPE-MLA patch engaged — stock image dies at warmup).
- `flashinfer 0.6.18.dev20260819` (NaN fix present, not 0.6.17).
- `NCCL ... Version: 2.30.7` and `cutlass-dsl ... Version: 4.6.2` (re-pins held).
- `Initial free memory ... reserved N GiB` with N ≈ your KV pin + weights, and
  no `NV_ERR_NO_MEMORY` in `dmesg`/`journalctl -k` in the death window.
- `Mamba ... / KDA state cache` allocation lines; `Uvicorn running on
  http://0.0.0.0:4000` — ready.
- Deep-decode request (28-32K prompt) completes ≥100 tokens — persistent_topk
  overlay verified. `/health` → 200.

### Bad markers (investigate)
- **`pe_dim must be 64 for fp8_ds_mla`** in warmup → stock image (build failed
  silently or image drift). MUST NOT appear.
- **Serves but every reply is a repeated-token loop** (e.g. `locklock` or
  `!`×N), `/health` green → silent FP4 MoE corruption — check `--moe-backend
  marlin` actually applied (`--moe-backend` in the serve args; also check the
  boot line for which MoE backend engaged). This is the one that lies.
- **NaN logits / garbage on 64-256-row batches** → FlashInfer still 0.6.17
  (patch 2 missing).
- **`EngineDeadError`, `sample_tokens` timeout on a deep decode** → the
  persistent_topk crash (overlay missing / not applied). dmesg will be clean.
- **Rank dies 1-2 min after "Initial free memory ... reserved"** with no
  Python exception → phantom KV backing — lower `KV_CACHE_MEMORY` (6 GiB+
  fails on TP2).
- **`ncclCommInitRank: internal error`** → NCCL downgraded (2.29.7) — patch 3
  missing. **CuTeDSL `cute-to-nvvm` ICE at ~91% load** → cutlass-dsl mixed
  (patch 4 missing).
- **Worker "Connection reset by peer" + head hangs at "Init torch distributed
  begin"** → stale-head rendezvous — `docker compose down` BOTH nodes first,
  worker first.
- **Silent segfault in warmup when you touched max-num-batched-tokens** → it
  went below 2048 (index_topk invariant) — the compose refuses it, but check
  if you overrode via env.
- `No available shared memory broadcast block found in 60 seconds` → **benign**
  (FlashInfer autotuning, CPU ~150%, not a hang).
- NVRM failures surface in dmesg minutes after the real event — capture
  `docker logs` BEFORE any teardown.

### Measurements to record (for the README audit trail)
- Decode tok/s single-stream + aggregate at 1/2/4/8 (upstream: 21.8-28.3
  single, 39.3 C2, 64.9-73.4 C4/C8).
- Prefill tok/s (upstream ~1150 @ 200K, flat).
- TTFT warmed (upstream 0.20-0.70 s).
- KV pool from the boot line (expect 507K @ 4.14 GiB or 672K @ 5.5 GiB).
- Deep-decode gate (28-32K ctx, ≥100 tokens) and 3× concurrent ~20K prefills.
- Vision probe if `LANGUAGE_MODEL_ONLY=0`.
- Boot time + first vs warm boot.

---

## 7. Provenance of this research

- **HF report**: live HF API + safetensors index metadata, 2026-08-28.
- **Forum report**: Discourse JSON API, threads 381433 / 381429 + category
  sweeps (11K-word report with verbatim quotes), 2026-08-28.
- **Engine report**: GitHub API (vllm-project/vllm, sgl-project/sglang, all
  community 2× repos), Docker Hub API, forum, 2026-08-28.
- **Image verification**: `vllm/vllm-openai:glm53-flash-arm64-cu130` on Docker
  Hub (digest `sha256:905c0293…`, arm64 only, ENTRYPOINT `["vllm","serve"]`).
  `radixark/vllm-glm53-flash:*` returns 404 (not public).
- **Model verification**: `LibertAIDAI/GLM-5.3-Flash-NVFP4` (194.6 GB,
  revision `aa28e1f5…`), `zai-org/GLM-5.3-Flash` FP8 (328.3 GB) and `-BF16`
  (642.6 GB) all live on HF, 2026-08-28.
