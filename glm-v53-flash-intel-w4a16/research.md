# glm-v53-flash-intel-w4a16-v029 — research

## 2026-09-20 — recipe inception: stock vLLM v0.29.0 "one-shot" image

**Objective.** Serve `Intel/GLM-5.3-Flash-W4A16-AutoRound@5eee1846f0321058ed73745f9aa16f2aaf0fc0a0`
on the 2-node DGX Spark cluster from a self-contained image built FROM the
official stable `vllm/vllm-openai:v0.29.0` arm64 image — **no GPTQ surgery**
(`prepare-model.sh` retired for this lane), **no runtime bind-mount overlays**,
all SM121 patches baked at build. Image publishes to `ghcr.io/taoofshawn`.
Scope: **mtp3 lane only** (native MTP k=3 + PMU128). The dflash2pmu lane is
explicitly out of scope — it remains available on the legacy image
(`glm-v53-flash-intel-w4a16/`), which is untouched and stays as the rollback.

**Research receipts (verified 2026-09-20 against primary sources; VERIFIED =
fetched live):**

| fact | status | source |
|---|---|---|
| v0.29.0 is the latest stable vLLM (2026-09-09); official multi-arch arm64 image exists | VERIFIED | GitHub Releases API + Docker Hub API |
| Base pin: `vllm/vllm-openai:v0.29.0-aarch64@sha256:18372a7224938643461b846fb64c5c9d3d6e9727e82caf2dc3043e620c9d4d7a` (CUDA 13.0.3, torch 2.13.0, flashinfer 0.6.18.post1, NCCL 2.30.7, cutlass-dsl 4.6.2, Python 3.12, Ubuntu 24.04) | VERIFIED | Docker Hub API + `docker/versions.json` |
| GLM-5.3-Flash native upstream support (PR #53906, merged 2026-09-03) is in v0.29.0; official recipe `recipes.vllm.ai/zai-org/GLM-5.3-Flash` lists `min_vllm_version: 0.29.0` | VERIFIED | PR JSON + `vllm/models/glm5next/` + recipe YAML |
| `quant_method: "auto-round"` loads natively on v0.29.0 (`INCConfig.override_quantization_method` → `inc`), incl. `auto_round:auto_gptq` packing + `extra_config` BF16 exclusions → no surgery | VERIFIED (tag file fetched) | `vllm/model_executor/layers/quantization/inc/inc.py@v0.29.0` |
| `dflash` spec-decode method is upstream-native in v0.29.0 (#52816/#53797) — DFlash2 overlay unnecessary (irrelevant here: dflash lane out of scope) | VERIFIED | `vllm/v1/spec_decode/dflash.py@v0.29.0` |
| eugr sym-qzeros fix superseded upstream (is_sym guard, PR #45656, commit `058cc0a8`) | VERIFIED (tag file fetched) | `vllm/model_executor/layers/quantization/auto_gptq.py@v0.29.0` |
| PR #53388 `disable_eagle_block_drop` merged 2026-09-01 (`481839ad`) but NOT in v0.29.0 → backport needed for the mtp3 lane | VERIFIED (tag's `speculative.py` lacks the field) | PR JSON + tag file |
| GB10/SM121 sparse-MLA fixes still OPEN upstream (PRs #53969, #46055) → tonyd2wild SM121 fixes must be re-derived and baked | VERIFIED | commit history + open PRs |
| `sparse_attn_indexer_kpool.py` exists at the old path in v0.29.0 (moved on main only, 2026-09-16) → overlay path valid for this tag | VERIFIED | contents API @v0.29.0 |
| `--prefix-match-unit`, `--kv-cache-memory`, `--async-scheduling` present in v0.29.0 | VERIFIED (grep of tag files) | `vllm/engine/arg_utils.py@v0.29.0` |
| `--reasoning-parser glm45` valid (glm45/glm47 both map to `Glm47MoeParserReasoningAdapter`) | VERIFIED | `vllm/reasoning/__init__.py@v0.29.0` |
| `--prefix-cache-retention-interval` is now a CLI flag defaulting to 0 (#52216) | VERIFIED (release notes) | v0.29.0 release notes |
| tonyd2wild's `sm121-v11-dflash2` base = day-0 `vllm/vllm-openai:glm53-flash-arm64-cu130` (~Aug-15 tree, v0.1.dev20051+g487ecf187) + 9-stage pure-python patch chain (v1 sparse-MLA gate, v3 flashinfer nightly, v4 NCCL pin, v5 cutlass pin, v6 PDL gate, v7 indexer top-k, v8 fp8-KV tile cap) + DFlash2 overlay; nothing newer published by him | VERIFIED | his `docker/` tree @`050081dc` + repo list |
| Spark build hosts OK: aarch64, docker 29.2.1, buildx v0.31.1, 1.1 TB free (both nodes) | VERIFIED (ssh) | spark-0f0b / spark-6d14 |
| No report found of the Intel W4A16 AutoRound quant running on stock vLLM 0.29 on Spark (INC path verified in code, not on GB10 hardware) → bring-up Gate A is the proof point | research negative | forum 382041 + thread sweeps |

**Provenance of the patch series carried here:**

- `0001-vllm-53388-native-mtp-block-drop.patch` — upstream PR #53388 (commit
  `481839ad`), backported to v0.29.0. Provides `disable_eagle_block_drop`
  (mtp3 lane boots with `"disable_eagle_block_drop": true`).
- `0002-vllm-coordinator-partial-hits.patch` — hybrid KV-cache coordinator
  partial-hash-hit gating (groups participating in prefix caching only);
  originally vendored from the florianbrede-ayet dflash2pmu lane but part of
  the mtp3 patch series as well (`mtp3-pmu128/Dockerfile` carries 0002).
- `0003-vllm-scheduler-lcm-mamba-block-align.patch` — scheduler
  `_mamba_block_aligned_split` uses the scheduler LCM block size (PMU128
  shrinks hash granularity to 128).
- tonyd2wild SM121 fixes (v1/v6/v7/v8 of his chain) — carried only where
  v0.29.0 still lacks them (see `docs/PATCH-REBASE-NOTES.md` for per-fix
  verdicts).
- **Dropped:** DFlash2 overlay (out of scope + upstream-native), eugr qzeros
  mod (upstream), flashinfer nightly + NCCL/cutlass re-pins (official release
  image already ships 0.6.18.post1 / NCCL 2.30.7 / cutlass-dsl 4.6.2), v9
  instanttensor loader (experimental, rank deaths).

**Deployment constraints inherited from the legacy lane (validated values,
re-gated at bring-up on this new image):** GMU 0.85, MAX_LEN 1,048,576,
MAX_SEQS 6, MNBT 8192, BLOCK_SIZE 2304, KV pin 13.5 GB fp8 e4m3 → ~1.92M-token
pool, marlin MoE, temp 1.0 / top_p 0.95 / thinking ON at effort `high`,
worker-first start order, port 8000, GID auto-detect, offline serving.

## 2026-09-20 — bring-up of `glm53-intel-w4a16-v029:20260920` — VERDICT: FATAL at first boot; recipe premise unmet; rolled back to legacy mtp3 lane

Executed per `docs/TESTING-RUNBOOK.md` from a remote session. All §1 staging
passed, §5 worker-first launch executed; **worker AND leader both FATAL ~20 s
after container start** — before any weight load, GID detect fine
(`NCCL_IB_GID_INDEX=3` on node1), raw auto-round snapshot preflight OK.

### FATAL

```
NotImplementedError: Unsupported speculative method: 'mtp'
  at vllm/config/speculative.py:1331 (__post_init__ else-raise),
  via EngineArgs.create_speculative_config → SpeculativeConfig(**speculative_config)
```

Same traceback on both ranks (worker `run_headless`, leader APIServer).

### Root cause (verified against the built image, not the tag tree)

`SpeculativeConfig.__post_init__` on v0.29.0 classifies a self-drafting MTP
target **purely by the draft config's `hf_config.model_type ∈ MTPModelTypes`**;
for an explicit `method:"mtp"` with no drafter it self-drafts from the target
checkpoint (`self.model = target.model_weights or target.model`, quantization
aligned) and injects **no** model_type override. The Intel checkpoint (and the
official zai FP8 one — checked, `num_nextn_predict_layers: null` in BOTH, so
that field is not the discriminator) has `model_type: "glm5_next"`, which is
**not** a member of the image's `MTPModelTypes` Literal (`deepseek_mtp`,
`glm4_moe_mtp`, `qwen3_next_mtp`, … 25 members, no `glm5_next`) → the elif
chain falls to the final `else: raise`.

Deeper: **the image has no GLM-5.3 integration at all.** Broad grep
(`grep -rln -i glm5 vllm/`) finds only `models/deepseek_v32/nvidia/glm52_low_latency_gemm.py`
(a fused-GEMM kernel shipped for DSv3.2, not GLM-5.3 support): no
`vllm/models/glm5next/`, no registry entries (`Glm5NextForCausalLM` /
`Glm5NextForConditionalGeneration` / `Glm5NextMTPModel` absent from
`model_executor/models/registry.py`). `ModelConfig` probe in-image resolves the
checkpoint to `TransformersMultiModalMoEForCausalLM` → "has no vLLM
implementation, falling back to Transformers". In-image transformers 5.16.1
DOES ship `models/glm5_next/`, so a no-spec-decode boot through the
Transformers fallback is *possible* in principle — but the mtp3 lane,
`FLASHINFER_MLA_SPARSE_SM120` selection and every SM121 patch in the stack
assume the native model, so the recipe's premise (mtp3 + sparse-MLA on a stock
v0.29.0 base + 4 patches) is **unmet**: the patch stack baked 0001/0003/0011/0012
onto a base that never had the model.

**Research receipt correction:** the inception receipt's "PR #53906 GLM-5.3
native support in v0.29.0 — VERIFIED, `vllm/models/glm5next/`" is WRONG for the
v0.29.0 tag (2026-09-08). The native module exists on **main** only
(24-file tree: `common/{model,mtp,attention,kda,multimodal,sparse_indexer}.py`,
`nvidia/ops/{fused_eh_norm,kpool_compress}`, `third_party/kda` triton kernels,
amd/ variants) with registry entries at `registry.py:122/428/692` and the spec
glue in main's `speculative.py` (~:1045: `glm5_next` → `glm5_next_mtp`,
`n_predict = num_nextn_predict_layers`, `architectures: ["Glm5NextMTPModel"]`;
`glm5_next_mtp` added to `MTPModelTypes`). The official recipe's
`min_vllm_version: 0.29.0` evidently presumes a build carrying that main-branch
integration. v0.29.0 is still the newest stable release today (no 0.29.1/0.30),
so there is no stable base that carries GLM-5.3 natively.

### Corrections + observations recorded during bring-up

- **Missing `IMAGE` var (fixed pre-launch):** compose reads `image: ${IMAGE}`
  but no `.env` defined it → compose FATALs at `up`. Added
  `IMAGE=glm53-intel-w4a16-v029:20260920` on the branch (commit `d7a206c`),
  pulled on both node clones; `docker compose config` then resolved the
  image + container name correctly on both nodes. Mirror of the legacy
  recipe's convention.
- **FLAGS-AUDIT watch item #1 confirmed:** boot warns
  `Unknown vLLM environment variable detected: VLLM_EXECUTE_MODEL_TIMEOUT_S`
  — v0.29.0 does not read it. A future build should set
  `VLLM_ENGINE_READY_TIMEOUT_S=3600` instead (both rows in compose env).
- **Image distribution:** image was on node0 under both tags
  (`glm53-intel-w4a16-v029:20260920` == `ghcr.io/taoofshawn/vllm-glm53-intel-w4a16:v0.29.0-pmu128-mtp3`,
  ID `1e987123e58f`); node1 had NEITHER (ghcr push does not land on node
  dockerd). Measured paths node0→node1: workstation-relayed
  `docker save|gzip|ssh` = **11 MB/s** (mgmt enP7s7); ssh pipe over the RoCE
  rail collapses sustained (342 MB/s burst for 300 MB, ~MB/s sustained);
  **disk-staged scp over RoCE = 411 MB/s sustained** (22,261,490,688 B in
  51.7 s) + `docker load -i` 2m09s. For future ~20 GiB image transfers use
  `docker save -o` → scp → `docker load`; skip streaming pipes. Temp tars
  removed from both nodes after load.

### Rollback (executed same session)

v029 containers `down` on both nodes → node clones back to `main` (clean,
`git diff origin/main...glm53-w4a16-v029-stock -- glm-v53-flash-intel-w4a16/`
empty, so main serves the identical legacy config as pre-cutover) →
drop_caches ritual → legacy `glm-v53-flash-intel-w4a16` worker-first relaunch
on `glm53-intel-mtp3-pmu128:20260907` → health re-verified (see below).
model-name-proxy (:4000, `BACKEND_MODEL=glm-5.3-flash`) stayed up throughout;
its healthcheck flips unhealthy while the backend is down and recovers with it.

### Bench gate (§8)

NOT RUN — the new image cannot serve, so no A/B numbers exist and the legacy
lane's recorded numbers (c4 129.3–132.1 stable, 2026-09-19 bench) stand
uncontested.

### Next-build options (user decision; none executed here)

1. **Port glm5next from main onto v0.29.0**: copy the 24-file module + 3
   registry lines + the `speculative.py` translation block + `glm5_next_mtp`
   Literal member, then audit `common/{attention,kda}.py` imports against
   v0.29.0's attention/kv-cache APIs (real drift risk — the 0011 patch already
   tunes the SM120 sparse-MLA file that main's module expects unmodified).
   Must go through the installer + build-receipt flow, not a bind-mount hack.
2. **Rebuild on a nightly base** that carries glm5next natively — re-opens
   every SM121 question (flashinfer pin, kernel stack) the v0.29.0 pin was
   chosen to avoid.
3. **Wait for 0.30 stable** carrying #53906's glue natively.
4. Diagnostic-only (not the recipe goal): boot the RAW checkpoint with NO
   speculative-config through the Transformers fallback to prove the INC
   auto-round load path on GB10. Loses mtp3 + likely sparse-MLA; not a
   serving candidate.

## 2026-09-20 (later) — parent-agent verification of the FATAL + probe of the official `glm53-flash` day-0 tag: viable new base found

Independent verification of the bring-up record (both facts confirmed):

- `git ls-tree -r v0.29.0 | grep -c glm5next` = **0**; `grep glm5_next_mtp
  vllm/config/speculative.py@v0.29.0` = 0. The tag has NO GLM-5.3 support.
- Ancestry check in the scratch clone:
  `git merge-base --is-ancestor 98ed0856 (#53906) v0.29.0` → **NOT an
  ancestor**; same for `481839ad (#53388)`. The v0.29.0 tag lives on a release
  branch cut before the GLM-5.3 main merge — the "merged 09-03, tag 09-08"
  dates were misleading; the inception receipt's inference ("in v0.29.0 per
  merge date") was wrong. Agent-verified in-image finding stands.

**Probe of the official re-pushed tag** `vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:b0501f99fec5136f248f78d5850977a2ec32d55cd9a665f4a9ffef24cbdf7fe5`
(pushed 2026-09-09 06:16, i.e. AFTER the GLM-5.3 main merge; version string
`0.28.1rc1.dev580+g385dce36b` — a newer main snapshot than tonyd2wild's
`0.1.dev20051+g487ecf187` base), probed in-image on spark-0f0b:

| capability | present |
|---|---|
| `vllm/models/glm5next/` native module | YES (21 .py files) |
| registry: Glm5NextForConditionalGeneration / ForCausalLM / MTPModel | ALL YES |
| `speculative.py`: `glm5_next_mtp` | YES (native MTP for GLM-5.3) |
| `disable_eagle_block_drop` (#53388) | **ALREADY IN TREE** → patch 0001 droppable |
| INC auto-round override (`auto-round`) | YES → no surgery |
| auto_gptq `is_sym`/`use_zp` guard | YES → no eugr patch |
| `platforms/cuda.py::is_arch_support_pdl` | still `major >= 9` → PDL gate (0012) needed |
| `sparse_attn_indexer_kpool.py` | EXISTS (v7 overlay target alive) |
| CTA_TILE_KV in `*mla*sparse*` files | not found (v8 target TBD) |
| flashinfer 0.6.18 / NCCL 2.30.7 / cutlass-dsl 4.6.2 / torch 2.13.0+cu130 | shipped (v3/v4/v5 pins unnecessary) |

This is the image the official recipe prescribes ("use
`vllm/vllm-openai:glm53-flash` until support lands in the standard image") —
official docker-hub, digest-pinnable, arm64/CUDA-13.0, and it carries native
GLM-5.3 + MTP + auto-round. Recommended base for build round 2; the patch
stack must be re-derived against THIS tree with the same fail-closed
empirical-gate flow (0001 drops; 0003/0012 re-check; v1/v7/v8-equivalent
SM121 fixes re-evaluated; flags re-audited — this tree predates the
`--kv-cache-memory-bytes` rename question and needs its own check).


## 2026-09-21 — round-2 bring-up + validation (image `20260921-r9`, all gates green)

Executed `docs/TESTING-RUNBOOK.md` top-to-bottom on the 2-node cluster.
Image lineage: `20260920-r2` (`6b838e5a0ff8`, all r2 gates green in-build) →
four bring-up fixes → `20260921-r9` (`b7fee2d76a85`). ghcr staging: node1
pulled `ghcr.io/taoofshawn/vllm-glm53-intel-w4a16:glm53flash-pmu128-mtp3`
(= r2 ID; user-pushed) in ~2 min via daemon-side layer dedup; later rounds
transferred node0→node1 via `docker save` → scp over RoCE → `docker load`
(~4 min for ~29.8 GiB). Note: `ghcr ...:v0.29.0-pmu128-mtp3` (`1e987123`) is
the FATAL round-1 image — never serve it.

### Bring-up failures and fixes (each boot ~14 min; 5 rounds)

| boot | failure | root cause | fix |
|---|---|---|---|
| r2 (`20260920-r2`) | `ValueError: moe_backend='marlin' is not supported for unquantized MoE` | compose force-passed `--moe-backend marlin`; v0.29 applies it globally and the checkpoint is MIXED: 45 quantized W4A16 layers + the nextn layer's experts, which INCConfig resolved unquantized | compose: emit `--moe-backend` only when `MOE_BACKEND` is set (oracle auto-select; `.env` `MOE_BACKEND=` empty) |
| r2 | `KeyError: 'model.layers.45.mtp_block.mlp.experts.routed_experts.w2_qweight'` (draft load) | the nextn layer IS quantized in the checkpoint (`layers.45.*` qweight) but the draft FusedMoE prefix `model.layers.45.mlp.experts` misses `block_name_to_quantize` (runtime value `language_model.model.layers` — Glm5Next inherits GLM-4V's `model.language_model.*`→`language_model.model.*` mapper; the `mtp_block` nesting exists only in attribute/weight-name space) → draft experts built unquantized, quantized tensors homeless | patch **0015**: `INCConfigParser._resolve_raw` re-tests any missed name against the target-model naming candidates (`language_model.model.layers.*` / `model.language_model.layers.*`) |
| r9-partial (during KV/warmup) | `ValueError: MLA kv_data_type torch.uint8 is not supported` (flashinfer MLA plan allowlist) | vLLM fp8-MLA KV spec stores E4M3 payloads as raw uint8; flashinfer's planner allowlists logical dtypes only | patch **0016**: `_SM90State` maps uint8→`float8_e4m3fn` for plan (forward already `.view()`s) |
| r7 | `no kernel image is available for execution on the device` at SM90 warmup | base's flashinfer 0.6.18 release: fp8-MLA gate narrowed to `major != 9` (legacy lane's `0.6.18.dev20260819` allowed `(9,12)`), and fa3 (CUTLASS SM90a) carries no SM121 SASS | image: adopt the legacy image's `flashinfer 0.6.18.dev20260819` + `flashinfer_cubin` via `COPY --from=glm53-intel-mtp3-pmu128:20260907` (torch 2.13.0+cu130 ABI-identical, ckv_scale_arr API identical, 19 refs both sides; dev build carries the `(9,12)` gate natively → planned 0017 dropped) |
| r8 | same "no kernel image", now from a JIT op dir suffixed `_sm90` | v0.29's sm90 impl hard-codes `backend="fa3"`; legacy tree selected `("fa3" if major==9 else "fa2")` — on GB10 the validated path is fa2/trtllm-fmha (JIT compiles for 121a; legacy lane's JIT cache proves it) | patch **0018**: sm90 sparse-MLA wrapper selects fa2 on non-SM90 |

### Gate A — boot markers (leader log, r9)

- `GID auto-detect: NCCL_IB_GID_INDEX=3` (no error dump)
- `SpeculativeConfig(method='mtp', ..., num_spec_tokens=3)` — mtp3 lane
- preflight: `raw auto-round snapshot OK` — INC/auto-round load path, NO surgery
- `Using 'MARLIN' WNA16 MoE backend` + `Using MarlinExperts` (+ `MarlinLinearKernel for AutoGPTQLinearMethod`)
- `GPU KV cache size: 1,920,956 tokens` — **exact legacy-lane pool @ 13.5 GB pin** (MRV2 accounting did NOT shrink it)
- `Using FLASHINFER_MLA_SPARSE_SM90 attention backend` (patch 0013, running fa2 on SM121 via 0018)
- `Application startup complete`; cold boot ≈ 14 min (worker-first + 35 s stagger)

### Gate B — API sanity (leader)

`/health` 200; `/v1/models` → id `glm-5.3-flash`, root `Intel/GLM-5.3-Flash-W4A16-AutoRound`,
`max_model_len` 1048576; chat completion returns real text ("Hi!", 12 completion tokens).

### Gate C — PMU128 + spec decode + long-context indexer

- PMU128: 274-token prompt ×2 → pass2 `cached_tokens=256` = `floor(274/128)×128` exact; 31.5K-token
  prompt ×2 → pass2 `cached_tokens=31488` = exact floor. PASS.
- Spec decode: 5 agent-style code-ish prompts (200–300 tok each), all coherent, no
  `EngineDeadError`/`DistStoreError`; `/metrics` acceptance counters: accepted/draft = 811/1242
  = **0.653** (~2.6 accepted/step, inside the legacy lane's 2.4–3.9 band). PASS (bench quantifies decode).
- Indexer/long-context: **31,533-token** prompt decoded correctly twice — past the legacy kpool
  ~24K crash territory; pass1 prefill 21.3 s, pass2 1.8 s (PMU replay). PASS (patch 0014's topk
  gate + PR #53969 unified indexer survived real traffic).

### Notes / watch items

- flashinfer version in-image is now `0.6.18.dev20260819` (adopted from the legacy image) —
  the base's flashinfer release 0.6.18 is NOT Blackwell-native for fa2/fa3 MLA. This is a
  dependency divergence from the official base worth re-checking when flashinfer ships
  Blackwell-native MLA.
- KV pool gate: 1,920,956 tokens — no need for the `KV_CACHE_MEMORY=` profiler-sized fallback.
- Legacy lane torn down per runbook §3 (containers removed on both nodes); rollback = bring
  `glm-v53-flash-intel-w4a16/` compose back up (worker first), nothing about it was modified.

### §8 A/B bench — VERDICT: no regression (2026-09-21)

`bench_recipe.py` Tier 1 (after side fresh boot `20260921-v029-r9`, warm-up stage
mandatory; raw runs `~/benchmarks/20260921-v029-r9` on the head). Before sides:
the 2026-09-19 recorded legacy mtp3 @ 13.5 GB runs (`20260919-mtp3-before`
32 h-warm; `20260919-mtp3-before2` fresh-boot control). Prompts byte-identical
(script-baked constants), stream:false semantics via final-usage chunks.

| cell | legacy before median | v029-r9 after median | delta | verdict |
|---|---|---|---|---|
| c4_prose_short (primary) | 129.97 (129.29–130.65) | 124.78 (117.69–131.88) | −4.0% | NOISE (ranges overlap) |
| c1_prose_short | 39.68 | 39.07 | −1.5% | INFO |
| c1_code_short | 39.77 | 39.89 | +0.3% | INFO |
| c1_prose_medium | 37.57 | 38.06 | +1.3% | INFO |
| c1_prose_long | 35.79 | 36.67 | +2.5% | INFO |
| pmu_replay_long | cached 43904 = expected | cached 43904 = expected | — | PASS both |

- acceptance (temp-0): 0.9968/0.9976 → 0.9979 (Δ +0.001, NOISE). Note: the
  engine's SpecDecoding log line reads "Mean acceptance length: 4.00, per-position
  1.000/1.000/1.000" under the temp-0 bench — genuine greedy acceptance (the
  drafter is the target's MTP head), matching the legacy lane's 0.997+ band.
- KV pool: 1,920,956 tokens on BOTH sides (compare tool flags "DIFFERENT" only
  because it can't parse the legacy log format; values identical).
- MemAvailable 3.28/3.0 → 3.37 GiB (ok).
- Verdicts from both references (`mtp3-before` and `mtp3-before2`): **no
  regression detected (medians within noise bands)**.
- JIT watch: `_rejection_kernel`/`_resample_kernel` Triton compilations during
  the first bench round on the worker (one-off latency spike; benign per skill
  doctrine, but the warmup does not cover them in this tree).

### Round-2 stop state (2026-09-21)

Legacy lane restored to serving per runbook §9 default (user decides after
seeing numbers; PR not merged). Branch `glm53-w4a16-v029-stock` @ `1918c78`+
has everything (patches 0015/0016/0018, legacy-flashinfer Dockerfile adoption,
moe-backend conditional, runbook/gate/bench records). One-command path back to
the new lane: pull branch on both nodes, `docker compose down` both, ritual,
worker-first `up` (image `20260921-r9` already on both nodes).
