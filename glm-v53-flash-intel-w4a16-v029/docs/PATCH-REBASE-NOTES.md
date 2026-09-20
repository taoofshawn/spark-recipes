# Patch rebase notes — legacy mtp3-pmu128 series → vLLM v0.29.0

Rebase performed 2026-09-20 in a scratch clone of vllm-project/vllm at tag
`v0.29.0` (commit `98dff2a81d747d1dba01a47f939f48c3526d4206`, 2026-09-08) on
spark-0f0b (`/tmp/vllm-rebase`).

| legacy patch | verdict | evidence |
|---|---|---|
| `0001-vllm-53388-native-mtp-block-drop.patch` | **REGENERATED** — cherry-pick of upstream commit `481839ad9e5ebf87aecb54fa5c9d986bd5ea4b81` (PR #53388) applied cleanly to v0.29.0; exported as the `vllm/`-files-only diff (168 lines, 4 files). Adds `disable_eagle_block_drop` field + `use_eagle_block_drop()` to `vllm/config/speculative.py` (verified additive — absent from the tag) plus the trailing-block handling in `kv_cache_utils.py`, `single_type_kv_cache_manager.py`, `scheduler.py`. | `git cherry-pick -n` clean; diff inspected |
| `0002-vllm-53906-coordinator-partial-hits.patch` | **DROPPED — already upstream in v0.29.0.** `vllm/v1/core/kv_cache_coordinator.py` (tag tree, lines ~642-659) contains the evolved form of the same fix: the `unsupported_partial_hit_managers` check zips managers with groups and gates on `group.kv_cache_spec.prefix_cacheable` (v0.29.0 renamed the spec attribute from the legacy patch's `participates_in_prefix_caching`). `git apply --check` fails because the code is already there. | tag file inspected 2026-09-20 |
| `0003-vllm-scheduler-lcm-mamba-block-align.patch` | **KEPT VERBATIM** — `git apply --check` succeeds on v0.29.0 (single hunk, offset +32 lines). | `git apply --check -v` output |

Order in the new image's installer: `0001` → `0003` → `0011` → `0012`. Patch
0002's deletion does not weaken the series: the coordinator behavior it
provided is present in the v0.29.0 tree.

Note on numbering: the legacy `0002` slot stays empty (dropped); no renumbering
so the lineage stays traceable.

## tonyd2wild SM121 fixes (v1–v8) — verdicts against v0.29.0

The legacy fixes were written against the ~Aug-15 day-0 tree
(`v0.1.dev20051+g487ecf187`). v0.29.0 refactored the affected code, so several
become obsolete in their literal form. Per-fix verdicts (tree inspected
2026-09-20):

| legacy fix | verdict | evidence |
|---|---|---|
| v1 — sparse-MLA backend gate: add `FLASHINFER_MLA_SPARSE_SM90` to cap-12 candidates; FA2 off-Hopper; scope FlashInfer ≥0.6.18 gate to fp8-KV | **OBSOLETE / REPLACED.** v0.29.0's cap-12 MLA list is `[TRITON_MLA, FLASHINFER_MLA_SPARSE_SM120]`; the SM120 backend's `supports_combination` now gates on `index_topk` (GLM-5.3-Flash: 2048 ✓) instead of the old `pe_dim=64` requirement. The TRTLLM-class sparse backend (`FLASHINFER_MLA_SPARSE`, `major == 10`) rejects GLM outright (`qk_nope_head_dim=256` not in [128,192]) — adding it to the cap-12 list buys nothing. The real GB10 gap moved into the SM120 backend itself and is covered by `0011`. FlashInfer 0.6.18.post1 ships sm_121a JIT support, making the 0.6.17 FA2/FA3 NaN workaround moot. | `platforms/cuda.py`, `flashinfer_mla_sparse.py` @v0.29.0; Intel checkpoint config (`index_topk=2048`, `qk_nope_head_dim=256`, `qk_rope_head_dim=0`) |
| v6 — PDL gate `is_arch_support_pdl` → `major in (9,10)` | **PORTED as `0012-sm121-pdl-gate.patch`.** v0.29.0 still has `return major >= 9` (`platforms/cuda.py:718-724`). | tree inspected |
| v7 — indexer kpool top-k hardening (our full-file overlay) | **OBSOLETE — target file does not exist in v0.29.0.** There is no `sparse_attn_indexer_kpool.py` anywhere in the tag tree (correction to earlier research: the file's path history lives on *main*, not at v0.29.0); GLM's kpool indexer was integrated into the unified `vllm/model_executor/layers/sparse_attn_indexer.py` by PR #53906. Whether the topk-buffer init hardening is still needed is a **bring-up watch item** (Gate C: decode past ~24K ctx). | `git ls-tree -r v0.29.0` grep |
| v8 — fp8-KV `CTA_TILE_KV=32` cap (GB10 ~101 KB smem) | **OBSOLETE — no `CTA_TILE_KV` symbol exists anywhere in the v0.29.0 tree**; the SM120 impl delegates tile selection to flashinfer's SM120 sparse-MLA API. fp8-KV warmup on GB10 is a **bring-up watch item** (Gate A). | tree-wide grep |
| v3/v4/v5 — flashinfer nightly + NCCL/cutlass re-pins | **DROPPED.** The official v0.29.0 release image already ships flashinfer 0.6.18.post1 / NCCL 2.30.7 / cutlass-dsl 4.6.2. | `docker/versions.json` |
| v9 — instanttensor loader | **DROPPED.** Experimental; rank deaths on TP2 in tonyd2wild's own testing. | his CURRENT.md |

## New SM121 patches created for v0.29.0

| patch | source | content |
|---|---|---|
| `0011-sm121-sm120-nope-topk.patch` | upstream PR [#53969](https://github.com/vllm-project/vllm/pull/53969) (open, "[Bugfix] Support NoPE models on FLASHINFER_MLA_SPARSE_SM120 and validate effective topk buffer width", 2 files +69/−7) — `git apply --check` **passes** on v0.29.0 (hunk offset +1). This is the PR written explicitly for GLM-5.3-Flash on GB10/SM121: zero-pads the rope section for NoPE models (pe_dim==0) in the SM120 impl and validates the effective topk buffer width instead of the raw `index_topk`. | `/pull/53969.diff` fetched 2026-09-20 |
| `0012-sm121-pdl-gate.patch` | tonyd2wild sm121-v6 port (one-expression change in `vllm/platforms/cuda.py::is_arch_support_pdl`, comment added citing provenance) | produced on the v0.29.0 scratch tree |

Also noted but **not** carried: upstream PR #46055 (SM121 capability gates for
FLASHMLA/FLASHMLA_SPARSE backends) — irrelevant here because those backends are
not on the cap-12 MLA candidate list for this model.
