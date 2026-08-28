# GLM-5.3-Flash — Research & Handoff Notes

**Status: DRAFT, not tested.** Everything below was gathered 2026-08-28
(model released 2026-08-26) from the NVIDIA Developer Forum, GitHub, Docker
Hub, and HuggingFace (sizes measured from safetensors index metadata, not
README claims). It is the context for `README.md` plus the watch-list for
future improvement passes. Treat every number as day-0/day-1 community data
until re-measured on this cluster.

Quick links: recipe `README.md` · primary upstream
[tonyd2wild/GLM-5.3-Flash-NVFP4-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-2x-DGX-Spark) ·
forum threads 381433 / 381429.

---

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
- `incoai/GLM-5.3-Flash-DFlash2` (0.4B block-diffusion drafter) is faster than
  MTP on GB300 but **CC BY-NC-ND (non-commercial) — do not use**; native MTP
  (BF16 head, in-checkpoint) is the license-free path.

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
mostly pay verify cost). MTP5 can gibberish (forum 353069). DSpark/DFlash2 for
GLM-5.3 does not exist yet (GLM-5.2 has it) — the biggest expected speed lever.

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

### 4.4 The big expected speedup: DSpark/DFlash2 for GLM-5.3
Not available yet (GLM-5.2 has it; `incoai/GLM-5.3-Flash-DFlash2` is SGLang +
non-commercial licensed). tonyd615 "update coming soon with DFLASH"; renek
targets 50–70 tok/s when DFlash2/DSpark lands. Watch the forum + tonyd2wild.

---

## 5. Known improvements / fixes to track (with blockers)

1. **Measure this cluster's actual numbers** (recipe is DRAFT). Record: boot
   time (15–25 min expected), KV pool from the boot line, decode tok/s
   (`stream:false` + `usage.completion_tokens`; streamed deltas under-report
   spec decode), prefill tok/s, TTFT, and the deep-decode gate (28–32K prompt,
   ≥100 decoded tokens — proves the persistent_topk overlay).
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
7. **Thinking default parity with the other recipes (WANTED, BLOCKED).** The
   other recipes default `thinking=true` (+ reasoning_effort). GLM-5.3's
   day-0 stack has thinking+tools issues (SGLang #36669 shows `!`
   degeneration under multi-tool agentic prompts; the GLM family has the same
   token-0 class as Qwen). When upstream fixes land (or a flag emerges that
   gives thinking + structured tools together), flip `THINKING=true` and
   re-run `tools/load_test_glm.py`. GLM's `reasoning_effort` levels are
   low/high/max (default max; `clear_thinking=true` for chat).
8. **The NVFP4-KV + b12x lane (drowzeys)** as the path to 1M ctx on 2 nodes —
   the single most valuable capacity upgrade; needs the zero-pad shim.
9. **Re-pin / de-duplicate patches when vLLM #53906 lands.** The patch guards
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
