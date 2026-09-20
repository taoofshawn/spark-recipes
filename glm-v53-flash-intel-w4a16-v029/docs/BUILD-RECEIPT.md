# Build receipt — glm53-intel-w4a16-v029:20260920

Built 2026-09-20 on `spark-0f0b.shawndo.intra` (aarch64, Docker 29.2.1), build
context `glm-v53-flash-intel-w4a16-v029/image/` from branch
`glm53-w4a16-v029-stock` (commit `a625748` + compose commit `35b0c94`).

| item | value |
|---|---|
| local tag | `glm53-intel-w4a16-v029:20260920` |
| image ID | `sha256:1e987123e58f992177721dc0ddfcb3991321ad84c04d03cdd7f42c27d286b048` |
| size | 22,053,245,216 bytes (~20.5 GiB uncompressed; ~10 GiB compressed on push) |
| base | `vllm/vllm-openai:v0.29.0-aarch64@sha256:18372a7224938643461b846fb64c5c9d3d6e9727e82caf2dc3043e620c9d4d7a` |
| ghcr tag (user push) | `ghcr.io/taoofshawn/vllm-glm53-intel-w4a16:v0.29.0-pmu128-mtp3` |
| build log | `/tmp/build-v029.log` on spark-0f0b (transient) |

## Installer gate results (both passes inside the build)

Fail-closed preflight + apply + independent `--verify-only` second pass — all
10 changed files at their expected final SHA-256:

```
vllm/config/speculative.py                 4e3d0a9b…  (0001: disable_eagle_block_drop)
vllm/distributed/.../mooncake/store/worker.py   e03fcf29…  (0001)
vllm/distributed/.../offloading/scheduler.py    cd43cf5f…  (0001)
vllm/v1/core/kv_cache_utils.py             f587863b…  (0001)
vllm/v1/core/sched/scheduler.py            a14442fb…  (0001 + 0003 LCM align)
vllm/v1/core/single_type_kv_cache_manager.py    14f5e759…  (0001)
vllm/v1/simple_kv_offload/manager.py       6877ac8f…  (0001)
vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py        3f42aabc…  (0011: effective topk width)
vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py  cd1c9fb4…  (0011: NoPE zero-pad)
vllm/platforms/cuda.py                     3d0914a5…  (0012: PDL gate (9,10))
```

## In-image verification battery (post-build, exit 0)

```
vllm version: 0.29.0
0001 disable_eagle_block_drop: OK
INC auto-round override: OK
0011 SM120 NoPE zero-pad + effective topk width: OK
0012 PDL gate (9,10): OK
0003 scheduler mamba-align present: OK
baked chat template sha256: OK   (918b0d8c…, byte-identical to the legacy lane's)
SM121 env wiring: OK             (TORCH_CUDA_ARCH_LIST / FLASHINFER_CUDA_ARCH_LIST = 12.1a)
ALL IN-IMAGE CHECKS PASSED
```

## Not yet validated (deliberately deferred to bring-up — see TESTING-RUNBOOK.md)

- Anything GPU-side: SM121 sparse-MLA boot (PR #53969 path), INC auto-round
  weight load on GB10, marlin MoE with the INC-delegated GPTQ config, KV pool
  size under the 13.5 GB pin with MRV2 accounting, PMU128 cached-token
  behavior, spec-decode acceptance, decode past ~24K ctx (former kpool-overlay
  territory), fp8-KV warmup (former v8-fix territory).
- The GID auto-detect / RoCE fabric path (unchanged compose block, but new
  base NCCL 2.30.7).
