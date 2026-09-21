# Build receipt — glm53-intel-w4a16-v029:20260921-r9 (FINAL)

## Final validated image (round-2 bring-up lineage, 2026-09-21)

| item | value |
|---|---|
| final tag | `glm53-intel-w4a16-v029:20260921-r9` |
| image ID | `sha256:b7fee2d76a85…` (full ID in `docker images` on both nodes) |
| ghcr tag | `ghcr.io/taoofshawn/vllm-glm53-intel-w4a16:glm53flash-pmu128-mtp3` |
| on both nodes | YES (`r2`→`r9` lineage present; `r3`–`r8` are intermediate build tags, safe to delete locally) |

The `r2`→`r9` lineage (5 boot rounds, each fix root-caused and recorded in
`research.md`): r3–r5 = 0015 (INC mtp_block quant resolve, three iterations
against the Glm5Next weight-name mapping), r6 = 0016 (SM90 fp8-KV plan dtype),
r7 = 0017 (fp8-MLA gate — superseded), r8 = legacy flashinfer
`0.6.18.dev20260819` adoption via `COPY --from` the legacy image (the base's
release flashinfer is not Blackwell-native for fa2/fa3 MLA), r9 = 0018 (sm90
sparse-MLA selects fa2 on non-SM90). The Dockerfile change for flashinfer
adoption means **builds require `glm53-intel-mtp3-pmu128:20260907` present on
the build host** (it is, on both nodes).

All bring-up gates + the §8 A/B bench passed on r9 (see `research.md`):
backend `FLASHINFER_MLA_SPARSE_SM90` (fa2), marlin MoE via the oracle
(compose emits `--moe-backend` only when set — the checkpoint is mixed
W4A16/BF16), KV pool 1,920,956 tokens (identical to legacy), PMU128 exact,
31.5K-token decode past the legacy kpool crash depth, acceptance in-band,
bench no-regression vs both 2026-09-19 legacy references.

## Round-2 base change (after the round-1 bring-up FATAL)

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
| local tag | `glm53-intel-w4a16-v029:20260920-r2` (superseded by r3–r9 lineage, see above) |
| image ID | `sha256:6b838e5a0ff8d18df5d3ece7d196061f98a98f7c9aff59faeb7820a9cb75bbf6` |
| size | 22,172,877,505 bytes (~20.6 GiB) |
| base | `vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:b0501f99fec5136f248f78d5850977a2ec32d55cd9a665f4a9ffef24cbdf7fe5` |
| build log | `/tmp/build-v029-r2.log` on spark-0f0b (transient) |

## Installer gate results at r2 (all OK; r3–r9 additions gated the same way)

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
support — boot FATAL, rolled back (see `research.md`). **Never serve or push
it**; `ghcr …:v0.29.0-pmu128-mtp3` (`1e987123`) is this dead image.

The stale `ghcr.io/taoofshawn/vllm-glm53-intel-w4a16:v0.29.0-pmu128-mtp3`
round-1 tag (if ever pushed) and the superseded `glm53flash-pmu128-mtp3` r2
digest should be removed on ghcr once r9 is pushed under the same tag
(over-writing a tag leaves the old digest untagged — delete it via the GitHub
package UI or `crane delete ghcr.io/taoofshawn/vllm-glm53-intel-w4a16@sha256:<digest>`).

## Bring-up validation status

VALIDATED on r9 (2026-09-21): all runbook gates green and the §8 A/B bench
showed no regression — see `research.md` for the full record.
