# glm-v53-flash-intel-w4a16-v029 — GLM-5.3-Flash Intel W4A16 (2× DGX Spark)

Serve [Intel/GLM-5.3-Flash-W4A16-AutoRound](https://huggingface.co/Intel/GLM-5.3-Flash-W4A16-AutoRound)
(GLM-5.3-Flash INT4 W4A16) on the 2-node DGX Spark cluster with native MTP3
speculative decoding + PMU128 prefix caching at 1M context.

## Image

| | |
|---|---|
| ghcr (canonical) | `ghcr.io/taoofshawn/vllm-glm53-intel-w4a16:glm53flash-pmu128-mtp3` |
| local build tag | `glm53-intel-w4a16-v029:20260921-r9` (ID `b7fee2d76a85`) |
| base | `vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:b0501f99…` — the official model-recipe image, digest-pinned; the only image carrying GLM-5.3-Flash native model + MTP (v0.29.0 stable does NOT, it predates the #53906 merge) |
| vLLM | `0.28.1rc1.dev580+g385dce36b`, CUDA 13.0 aarch64, torch 2.13.0+cu130 |
| flashinfer | `0.6.18.dev20260819` — copied from the `glm53-intel-mtp3-pmu128:20260907` image, which must be present on the build host (the base's release flashinfer is not Blackwell-native for fa2/fa3 MLA) |

vLLM's `INCConfig` loads the Intel `quant_method: "auto-round"` checkpoint
natively (packing `auto_round:auto_gptq` + 679 `extra_config` BF16 exclusions
honored): the raw snapshot `5eee1846…` is served directly and `/v1/models
.root` reports the HF id.

The SM121 patch stack is baked at build by a fail-closed, SHA-gated installer
(`image/apply_runtime_patches.py` — exact before/after SHA-256 per file,
syntax-checked, two verification passes):

| patch | what it fixes |
|---|---|
| 0003 | scheduler `_mamba_block_aligned_split` uses scheduler LCM block size (PMU128 core) |
| 0011 | SM120 sparse-MLA: NoPE zero-pad + effective-topk-width validation ([vllm PR #53969](https://github.com/vllm-project/vllm/pull/53969), open) |
| 0012 | `is_arch_support_pdl` → `major in (9,10)` (SM12x PDL NaN lottery) |
| 0013 | NoPE-capable `FLASHINFER_MLA_SPARSE_SM90` backend available on cap-12 |
| 0014 | kpool indexer top-k gated on ≥78 SMs (decode crash past ~24K ctx on GB10's 48 SMs) |
| 0015 | `INCConfigParser` re-resolves nextn/MTP-layer expert names against the target-model naming (Glm5Next `model.language_model.*` → `language_model.model.*`) |
| 0016 | SM90 fp8-KV plan dtype: map uint8 fp8-MLA storage → `float8_e4m3fn` for flashinfer's planner |
| 0018 | SM90 sparse-MLA selects fa2 on non-SM90 (fa3/CUTLASS has no SM121 SASS) |

Full patch provenance and the validation record live in `research.md`.

## Deploy

### 0) Weights (one-time, BOTH nodes)
```bash
hf download Intel/GLM-5.3-Flash-W4A16-AutoRound \
    --revision 5eee1846f0321058ed73745f9aa16f2aaf0fc0a0
```

### 1) Image (BOTH nodes)
```bash
docker pull ghcr.io/taoofshawn/vllm-glm53-intel-w4a16:glm53flash-pmu128-mtp3
```

or build locally on each node (aarch64; the `glm53-intel-mtp3-pmu128:20260907`
image must be present for the flashinfer copy):
```bash
cd glm-v53-flash-intel-w4a16-v029/image
docker build -t glm53-intel-w4a16-v029:mycopy .
```

Transfers between nodes: `docker save -o` → `scp` over RoCE (~411 MB/s) →
`docker load` (streaming pipes measure ~11 MB/s — avoid).

### 2) Launch — worker (rank 1) FIRST, leader ~35 s later

```bash
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
docker compose --env-file .env --env-file .env.node1 up -d   # worker
docker compose --env-file .env --env-file .env.node0 up -d   # leader ~35 s later
```

Cold boot ≈14 min. `docker compose down` on BOTH nodes between relaunches.
Wrong order hangs rendezvous (`DistStoreError: 1/2 clients`).

### 3) Verify (leader)

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health   # 200
curl -s http://127.0.0.1:8000/v1/models | jq -r '.data[0]|(.id,.root,.max_model_len)'
# glm-5.3-flash / Intel/GLM-5.3-Flash-W4A16-AutoRound / 1048576
```

## Boot markers (leader log)

```bash
docker logs glm53-intel-w4a16-v029 2>&1 | grep -F "GID auto-detect"          # NCCL_IB_GID_INDEX=N
docker logs glm53-intel-w4a16-v029 2>&1 | grep -F "SpeculativeConfig"        # method='mtp', num_speculative_tokens: 3
docker logs glm53-intel-w4a16-v029 2>&1 | grep -F "raw auto-round snapshot"  # preflight OK
docker logs glm53-intel-w4a16-v029 2>&1 | grep -F "MARLIN"                   # 'MARLIN' WNA16 MoE backend (oracle-selected)
docker logs glm53-intel-w4a16-v029 2>&1 | grep -F "GPU KV cache size"        # 1,920,956 tokens @ 1M ctx (13.5 GB pin)
docker logs glm53-intel-w4a16-v029 2>&1 | grep -F "FLASHINFER_MLA_SPARSE_SM90"  # the NoPE sparse-MLA backend (fa2)
```

PMU128 check: repeat a >128-token prompt with
`"stream_options":{"include_usage":true}`; the second response's
`usage.prompt_tokens_details.cached_tokens` ≈ prompt_tokens floored to a
128-multiple.

## Serving profile

GMU 0.85, MAX_LEN 1,048,576, MAX_SEQS 6, MNBT 8192, BLOCK_SIZE 2304, KV pin
13.5 GB fp8_e4m3 → 1,920,956-token pool, temp 1.0 / top_p 0.95 / thinking ON
at effort `high`, `MOE_BACKEND=` empty (oracle per-layer selection — the
checkpoint is mixed W4A16/BF16; forcing `marlin` globally FATALs on the BF16
MTP-layer experts).

## Known issues & watch items

- This image is mtp3-only (no external DFlash2 drafter lane).
- flashinfer `0.6.18.dev20260819` is a dev build; re-check when flashinfer
  ships Blackwell-native fa2/fa3 MLA in a release.
- Single ~310K+ token prompts have hung hosts in upstream tests; the measured
  fleet norm is 30–40K cached agent prompts.
- Vision+text concurrency: a pair of simultaneous image+text requests crashed
  a sibling profile fatally (forum 381350) — test before relying on it.
- Driver: cluster is on 580.173.02 (don't upgrade blindly).

## References

- Changelog / validation history: `research.md` in this directory
- Base image & official recipe: [recipes.vllm.ai/zai-org/GLM-5.3-Flash](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash)
- SM121 patch provenance: [tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark) docker/ chain
- Upstream SM121 fix still open: [vllm PR #53969](https://github.com/vllm-project/vllm/pull/53969)
