# Build receipt — glm53-intel-w4a16-v029:20260920-r2 (round 2)

Built 2026-09-20 on `spark-0f0b.shawndo.intra` (aarch64, Docker 29.2.1), build
context `glm-v53-flash-intel-w4a16-v029/image/` from branch
`glm53-w4a16-v029-stock` @ `a1b5c2b`.

## Round-2 base change (after the round-1 bring-up FATAL)

Round 1 (`20260920`, base `vllm/vllm-openai:v0.29.0-aarch64`) died at boot:
the v0.29.0 tag is cut from a release branch that predates the GLM-5.3-Flash
main merge (PR #53906) — no `glm5next` module, no registry entries, `mtp`
unclassifiable (`NotImplementedError`, see `research.md` bring-up record).
Round 2 rebuilds on the official model-recipe image the vLLM recipe itself
prescribes: `vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:b0501f99…`
(re-pushed 2026-09-09; main snapshot `0.28.1rc1.dev580+g385dce36b`), which
carries glm5next + native MTP + `disable_eagle_block_drop` + INC auto-round +
the is_sym qzeros guard **in-tree**. Patch 0001 (#53388 backport) is dropped;
the SM121 patch stack was re-derived against this tree.

| item | value |
|---|---|
| local tag | `glm53-intel-w4a16-v029:20260920-r2` |
| image ID | `sha256:6b838e5a0ff8d18df5d3ece7d196061f98a98f7c9aff59faeb7820a9cb75bbf6` |
| size | 22,172,877,505 bytes (~20.6 GiB) |
| base | `vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:b0501f99fec5136f248f78d5850977a2ec32d55cd9a665f4a9ffef24cbdf7fe5` |
| ghcr tag (user push) | `ghcr.io/taoofshawn/vllm-glm53-intel-w4a16:glm53flash-pmu128-mtp3` |
| build log | `/tmp/build-v029-r2.log` on spark-0f0b (transient) |

## Installer gate results (both passes inside the build — all OK)

```
0003  vllm/v1/core/sched/scheduler.py                 0f7e248a… → 25965dab…  (LCM/mamba align)
0011  vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py       98182222… → a7dbbbc7…  (effective topk width)
0011  vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py a0023f72… → cd1c9fb4…  (NoPE zero-pad)
0012  vllm/platforms/cuda.py                          e0345eb8… → b62c8ab7…  (PDL gate (9,10))
0013  vllm/platforms/cuda.py                          b62c8ab7… → b6a3b5a7…  (cap-12 + SM90 first)
0013  vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py  8b974fa9… → 4042e4f9…  (SM90 gate (9,12))
0014  vllm/model_executor/layers/sparse_attn_indexer_kpool.py       631ae1cc… → d105bbee…  (topk SM-count gate)
```

## In-image verification battery (post-build, exit 0)

```
vllm version: 0.28.1rc1.dev580+g385dce36b
native GLM-5.3 + MTP + disable_eagle_block_drop: OK
INC auto-round + is_sym qzeros guard: OK
baked patches 0003/0011/0012/0013/0014: OK
template + env wiring: OK
ALL ROUND-2 IN-IMAGE CHECKS PASSED
```

## Round-1 build (superseded, kept for the record)

`glm53-intel-w4a16-v029:20260920` (ID `1e987123…`, base `v0.29.0-aarch64`):
installer gates + in-image checks all passed, but the base lacked GLM-5.3
support — boot FATAL, rolled back (see `research.md`). Do not push it to ghcr.

## Not yet validated (bring-up gates — see TESTING-RUNBOOK.md)

- GPU-side: SM90 sparse-MLA selection on GB10 (0013 makes it the first
  cap-12 candidate — boot log should show `FLASHINFER_MLA_SPARSE_SM90`),
  INC auto-round weight load, marlin MoE, KV pool under the 13.5 GB pin,
  PMU128 behavior, spec-decode acceptance, decode past ~24K ctx (0014 under
  test), fp8-KV warmup on the SM90 path (the legacy v8 tile fix has no
  literal target in this tree — the SM90 impl carries its own
  `_WORKSPACE_BYTES`/fp8 handling).
