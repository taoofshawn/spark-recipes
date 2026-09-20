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
