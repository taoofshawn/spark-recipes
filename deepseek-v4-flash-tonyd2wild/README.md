# deepseek-v4-flash-tonyd2wild - 1M token context (NVFP4 DS-MLA KV)

A second recipe for running DeepSeek-V4-Flash-0731 on a 2-node DGX Spark cluster,
built from [tonyd2wild's DSpark stack](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark)
instead of the aiden image. It exists to **head-to-head test** whether this image
and configuration is better than the aiden one — same model revision, same
hardware, but a different runtime:

| | aiden `deepseek-v4-flash-aiden` | this recipe |
|---|---|---|
| Image | prebuilt `aidendle94/sparkrun-vllm-ds4-gb10@3b4d2b5f…` | local build `vllm-dspark-runtime:dspark-nvfp4-stage-c` |
| Model runner | v2 (`VLLM_USE_V2_MODEL_RUNNER=1`) | v1 (`--distributed-executor-backend mp`) |
| KV cache | `fp8` | `nvfp4_ds_mla` |
| DSpark spec tokens | 4 | **5** (`MTP_NUM_TOKENS=5`) |
| `max_num_seqs` | 16 | **6** (measured-best for 1M) |
| `max_num_batched_tokens` | 16384 | 8192 |
| `max_cudagraph_capture_size` | 256 | `seqs×(k+1)` = 36 |
| `gpu_memory_utilization` | 0.83 | **0.78** (0.80 "boots-then-dies" on this stack) |
| sampling default | temp/top_p overrides | `--generation-config vllm`, no override |
| thinking default | on / max | `false` (parameterizable via `THINKING`) |

> ⚠️ **Do not copy flags across images.** The aiden image rejects
> `nvfp4_ds_mla` + `--distributed-executor-backend mp`; this image rejects
> aiden's `VLLM_USE_V2_MODEL_RUNNER=1` and
> `--attention-backend FLASHINFER_MLA_SPARSE_DSV4`. Keep each recipe's native
> backend wiring.

---

## Prep (both nodes) — 1) Ensure the 0731 model is cached

```bash
cd ~/spark-recipes/deepseek-v4-flash-tonyd2wild/
HF_MODEL=$(grep "MODEL_PATH:" docker-compose.yml | awk '{print $NF}')
HF_REVISION=$(grep "MODEL_REVISION:" docker-compose.yml | awk '{print $NF}')
hf download $HF_MODEL --revision $HF_REVISION
```
`HF_CACHE` in `.env` already points at the same 0731 snapshot the aiden recipe
uses, so if you already ran that recipe there is nothing to download.

## Prep — 2) Build the image (CPU build, no GPU needed)

There is **no prebuilt public image** — the final image is a scripted 4-stage
overlay build on top of a public base
(`ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready`). Build on the head node
with the vendored upstream scripts:

```bash
cd upstream/
docker pull ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready
./build-dspark-vllm-runtime.sh        # builds mia-raf-pr1 → nvfp4-a → nvfp4-b → dspark-nvfp4-stage-c
```
`build-dspark-vllm-runtime.sh` rsyncs to the worker and rebuilds there too by
default (`WORKER_BUILD=1`); or build on one node and `docker save`/`load` to the
other. Confirm the tag exists on **both** nodes:

```bash
docker images vllm-dspark-runtime:dspark-nvfp4-stage-c
```

### Verify Patch 4 is in the image (0731 shared-expert fix)

Without it, 0731 decode roughly halves (acceptance ~26%). A fresh build already
contains it, but confirm:

```bash
docker run --rm --entrypoint grep vllm-dspark-runtime:dspark-nvfp4-stage-c \
  -n "shared_experts.gate_up_proj" \
  /opt/env/lib/python3.12/site-packages/vllm/v1/spec_decode/dspark.py
```
Expected: two lines, `.shared_experts.w1` and `.shared_experts.w3`.

### Missing module?

This upstream is known to occasionally reference a module that is **not in this
repo** but exists in **another repo by the same author** (`tonyd2wild`). If the
build or first run errors on a missing path, search the author's other public
repos (e.g. his DSpark serving-stack / MiMo repos) before going deeper.

---

## Run

```bash
# Node 1 (follower): start first
docker compose --env-file .env --env-file .env.node1 up -d

# Node 0 (leader): start about 30s after
docker compose --env-file .env --env-file .env.node0 up -d
```

API serves at `http://HEAD_NODE_IP:8000/v1` (served model
`deepseek-v4-flash` — same as the aiden recipe so existing router/client wiring works unchanged).

## files

| File/dir | Purpose |
|---|---|
| `.env` | Shared config — must be customized |
| `.env.node0` / `.env.node1` | Per-node overrides |
| `docker-compose.yml` | compose (aiden-style, tonyd2wild config) |
| `upstream/` | **Vendored upstream repo** (build scripts, patches, docs) — see `upstream/VENDORED-AT.md` |

## Key knobs

| var | default | what it controls |
|---|---|---|
| `MTP_NUM_TOKENS` | 5 | DSpark `num_speculative_tokens` (k=5 validated; k=3 ≈ −24%) |
| `MAX_NUM_SEQS` | 6 | concurrency cap (6 is measured-best at 1M; 12 is riskier) |
| `GPU_MEM` | 0.78 | keep ≤0.78 on this stack |
| `MAX_MODEL_LEN` | 1048576 | 1M is the true YaRN ceiling |
| `THINKING` | false | server `thinking` default; your pi client drives effort client-side anyway |
| `PORT` | 8000 | serve port |

`MAX_NUM_BATCHED_TOKENS=8192` and `max-cudagraph-capture-size=seqs×(k+1)` are
derived per the upstream's validated profile — don't touch without re-measuring.

## Finding interface names for .env

### Ethernet ports (control plane → `ETH_IF`, `ETH_IF2`)
```bash
ip addr show | grep -E '^[0-9]+: en'
```
### RoCE ports (data plane → `IB_PORTS`)
```bash
ibdev2netdev     # or: rdma link show
```
Typical mapping: `enp1s0f0np0` ↔ `rocep1s0f0`, `enP2p1s0f0np0` ↔ `roceP2p1s0f0`.
`GLOO_SOCKET_IFNAME`/`TP_SOCKET_IFNAME` use `ETH_IF`; the RoCE-v2 IPv4 GID index
is auto-detected at boot (override via `NCCL_IB_GID_INDEX` if needed).

## Benchmarking caveats (from upstream — avoid misleading numbers)

- **Use `stream: false`** and read `usage.completion_tokens` — under spec-decoding,
  streamed deltas measure *steps/s*, not tok/s (up to ~4× under-report).
- **Warm the engine** — fresh boot is ~30% slow until a few hundred tokens of real
  traffic, and the warm state **decays after idle**. Never benchmark right after
  boot or after a quiet period.
- KV pool is per-boot (varies ~15% boot-to-boot); the 1.5M pool figure is not a
  fixed property.

## Attribution / upstream

Built from **tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark**,
pinned at commit `d728faee9f5a8d5ebafe7bc44bca6c5d8d0d192f` (2026-07-31), fully
vendored under `upstream/` (see `upstream/VENDORED-AT.md` for refresh steps and
license). Patches: 1/2/2b (DSpark concurrency), 3 (k=5 garble fix), 4 (0731
shared-expert loader) — all baked into the image at build time.
