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
