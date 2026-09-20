# Patch rebase notes — legacy mtp3-pmu128 series → vLLM v0.29.0

Rebase performed 2026-09-20 in a scratch clone of vllm-project/vllm at tag
`v0.29.0` (commit `98dff2a81d747d1dba01a47f939f48c3526d4206`, 2026-09-08) on
spark-0f0b (`/tmp/vllm-rebase`).

| legacy patch | verdict | evidence |
|---|---|---|
| `0001-vllm-53388-native-mtp-block-drop.patch` | **REGENERATED** — cherry-pick of upstream commit `481839ad9e5ebf87aecb54fa5c9d986bd5ea4b81` (PR #53388) applied cleanly to v0.29.0; exported as the `vllm/`-files-only diff (168 lines, 4 files). Adds `disable_eagle_block_drop` field + `use_eagle_block_drop()` to `vllm/config/speculative.py` (verified additive — absent from the tag) plus the trailing-block handling in `kv_cache_utils.py`, `single_type_kv_cache_manager.py`, `scheduler.py`. | `git cherry-pick -n` clean; diff inspected |
| `0002-vllm-53906-coordinator-partial-hits.patch` | **DROPPED — already upstream in v0.29.0.** `vllm/v1/core/kv_cache_coordinator.py` (tag tree, lines ~642-659) contains the evolved form of the same fix: the `unsupported_partial_hit_managers` check zips managers with groups and gates on `group.kv_cache_spec.prefix_cacheable` (v0.29.0 renamed the spec attribute from the legacy patch's `participates_in_prefix_caching`). `git apply --check` fails because the code is already there. | tag file inspected 2026-09-20 |
| `0003-vllm-scheduler-lcm-mamba-block-align.patch` | **KEPT VERBATIM** — `git apply --check` succeeds on v0.29.0 (single hunk, offset +32 lines). | `git apply --check -v` output |

Order in the new image's installer: `0001` → `0003` → (SM121 fixes, see
below). Patch 0002's deletion does not weaken the series: the coordinator
behavior it provided is present in the v0.29.0 tree.

Note on numbering: the legacy `0002` slot stays empty (dropped); no renumbering
so the lineage stays traceable.
