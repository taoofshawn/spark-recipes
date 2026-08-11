# MiMo-V2.5 + DFlash + FP8 KV — docker-compose for 2x DGX Spark

> Docker compose startup for **MiMo-V2.5** (NVFP4 4-bit) with Xiaomi **DFlash** speculative decoding
> and an **FP8 KV cache** — vLLM TP=2 over Ray across two GB10 Sparks.
>
> Refactored from [tonyd2wild/MiMo-V2.5-DFlash-FP8-KV-2x-DGX-Spark](https://github.com/tonyd2wild/MiMo-V2.5-DFlash-FP8-KV-2x-DGX-Spark)
> into a self-contained docker-compose package following the
> [deepseek-v4-flash-aiden](../deepseek-v4-flash-aiden) style.
>
> **Start node 1 (follower) first, then node 0 (leader) ~30s later.**

## Quick start

### Prerequisites (both nodes)

All dependencies are now self-contained in this repo. The overlay image is built
from a public base (`ghcr.io/tonyd2wild/mimo-v2.5-tp2-1m-nvfp4kv:20260620`).

Clone this repo to both nodes, then:

```bash
cd spark-recipes/mimo-v25-dflash-tonyd2wild

# 1. Build the overlay image (pulls public base + DFlash model/proposer + eagle3 wiring)
docker compose build

# 2. Download the target model (171G — on both nodes)
#    Pinned revision: a147dd04d6cf861e43b2d783dcde23b53ab7ee68
hf download lukealonso/MiMo-V2.5-NVFP4 --revision a147dd04d6cf861e43b2d783dcde23b53ab7ee68

# 3. Download the drafter — dflash/ subdir only (2.8G — on both nodes)
#    Pinned revision: 1f58446181abcaa01030fdbde835fbd38ae9a2b1
hf download XiaomiMiMo/MiMo-V2.5-DFlash --revision 1f58446181abcaa01030fdbde835fbd38ae9a2b1 --include "dflash/*"
```

> **Caution:** Do NOT download the full XiaomiMiMo/MiMo-V2.5-DFlash repo — it
> carries a 311GB fp8 copy of the target you don't need. The `--include "dflash/*"`
> flag is mandatory. Both revisions are HF commit-style hex hashes from the original
> DEFAULT-CONFIG — they pin the exact model snapshots verified in the go-live deploy.

### Run

```bash
# Node 1 (follower): start first
docker compose --env-file .env --env-file .env.node1 up -d

# Node 0 (leader): start about 30s after
docker compose --env-file .env --env-file .env.node0 up -d
```

### Verify

```bash
curl http://192.168.0.170:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"MiMo-V2.5-NVFP4-DFlash-FP8KV","messages":[{"role":"user","content":"Reply exactly: OK DFLASH LIVE"}],"max_tokens":16,"temperature":0}'
```

Expected boot evidence in the leader's logs: `GPU KV cache size: 980,748 tokens`.

## Files

| File / dir | Purpose |
|---|---|
| `.env` | Shared config — interfaces, network |
| `.env.node0` | Per-node overrides for node 0 (leader) |
| `.env.node1` | Per-node overrides for node 1 (follower) |
| `docker-compose.yml` | Compose file with full startup script |
| `patches/` | Runtime engine patches (applied inside container at startup) |
| `patches/Dockerfile` | Build the overlay image from public ghcr.io base + DFlash model/proposer + eagle3 wiring |
| `mods/` | 6 base mods (from the companion repo) — applied at container startup |

## What the compose does

The `docker-compose.yml` `command` block automates the entire DEFAULT-CONFIG startup:

1. **Auto-detects RoCE-v2 GID index** (same logic as deepseek recipe — survives reboots)
2. **Applies 6 base mods** from `mods/` (required — registers MimoV2Config, etc.)
3. **Applies 7 engine patches** from `patches/` (idempotent — safe to re-run)
4. **Starts Ray** — head (node 0) or worker (node 1), with 1 GiB object-store cap
5. **Resolves HF snapshot paths** from the pinned revision hashes — validates exact expected paths in HuggingFace cache
6. **Launches vLLM serve** with the go-live config:
   - 500K max context, FP8 KV, block-size 16
   - DFlash with 7 speculative tokens
   - `triton_attn` backend (GB10 is FA2-only)
   - `--enforce-eager` (CUDA graphs are neutral-to-negative with custom kernels)
   - `NCCL_PROTO=LL` + `NCCL_MAX_NCHANNELS=2` (danielgbates small-message latency tuning)
   - Prefix caching + chunked prefill
   - Auto tool choice with `mimo` parser
   - No repetition penalty, temperature 0, top_p 0.95

## Customization

### Finding your interface names

```bash
# Ethernet ports (control plane)
ip addr show | grep -E '^[0-9]+: en'

# RoCE/InfiniBand ports (data plane)
ibdev2netdev
# or
rdma link show
```

Update `ETH_IF`, `ETH_IF2`, and `IB_PORTS` in `.env` to match your hardware. Typical mapping:

| Ethernet | RoCE |
|---|---|
| `enp1s0f0np0` | `rocep1s0f0` |
| `enP2p1s0f0np0` | `roceP2p1s0f0` |

### Override model launch parameters

Pass env vars to `docker compose up`:

```bash
MAX_MODEL_LEN=131072 MAX_NUM_SEQS=4 \
  docker compose --env-file .env --env-file .env.node0 up -d
```

Or set them in `.env.node0` / `.env.node1`.

## Node layout (from DEFAULT-CONFIG)

| role | node | ROCE_IP | HF cache |
|---|---|---|---|
| Ray head (rank0) | Node 0 | 192.168.0.170 | `~/.cache/huggingface` |
| Ray worker (rank1) | Node 1 | 192.168.0.171 | `~/.cache/huggingface` |

Direct-cabled RoCE on `enp1s0f0np0` / `rocep1s0f0`, `192.168.0.0/24`.

## Docker Compose Reference

The [discussion thread](https://forums.developer.nvidia.com/t/deepseek-v4-flash-aiden-recipe-from-reddit-1m-token-session-operational-cuda-12-1-tailored-for-dgx-spark-gb10/372268)
for the deepseek compose style this is based on.
