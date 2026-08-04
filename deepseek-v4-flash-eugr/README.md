# deepseek-v4-flash-eugr

DeepSeek-V4-Flash-0731 served on a **2-node DGX Spark cluster** using **eugr's
`spark-vllm-docker` `b12x` branch** (B12X-optimized vLLM / SparkInfer stack).

This is [bernisse's solution](https://forums.developer.nvidia.com/t/instructions-for-running-deepseek-v4-flash-with-dspark-using-eugrs-repo/376220/18)
from the DSpark forum: eugr's official b12x recipe for `DeepSeek-V4-Flash-0731`,
plus a **reasoning-effort fix mod** (real low/high/max prompt prefixes) and the
**instanttensor hybrid draft loader** mod.

> Built on the DSpark **speculative decoding** (`--speculative-config` method `dspark`)
> with **B12X** `--moe-backend b12x --linear-backend b12x --attention-backend B12X_MLA_SPARSE`.

---

## What's in this directory

```
deepseek-v4-flash-eugr/
├── README.md                                        # this file
├── deepseek-v4-flash-eugr.yaml                      # the recipe (eugr b12x run-recipe format)
└── mods/
    ├── dsv4-reasoning-effort-fix/run.sh             # reasoning-effort fix (bernisse, committed here)
    └── instanttensor-hybrid-draft-loader/           # hybrid loader (vendored from eugr b12x)
        ├── README.md
        ├── patch_model_loader.py
        └── run.sh
```

The recipe (`deepseek-v4-flash-eugr.yaml`) references these mods by relative path
and must be run from inside a checkout of eugr's `spark-vllm-docker` repo on the
**b12x** branch (that repo ships the required `run-recipe.py`, `launch-cluster.sh`
and `build-and-copy.sh`).

---

## Prerequisites (both nodes)

- Two DGX Sparks cabled over RoCE (ConnectX) + control-plane Ethernet.
- Passwordless SSH from the **head node (node0 / leader)** to the worker (node1).
- The model ready in the HF cache **on both nodes** (avoids re-download):

```bash
# head and worker both
ls ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731  # ~156GB, 96 shards
```

---

## Build the b12x container (on the head node)

1. Check out eugr's `spark-vllm-docker` and switch to the `b12x` branch:

```bash
git clone https://github.com/eugr/spark-vllm-docker.git
cd spark-vllm-docker
git checkout b12x
```

2. Build the B12X image and distribute it across the cluster:

```bash
./build-and-copy.sh -c --exp-b12x
```

   - `--exp-b12x` produces the image tagged `vllm-node-b12x`.
   - `-c` copies the freshly built image to the worker nodes over the fast
     InfiniBand link.
   - ⚠️ `--exp-b12x` does **not** support `--use-wheels`; it rebuilds vLLM
     wheels from source, so the first build takes ~20–40 minutes.

   Rebuild anytime the repo or mods change:

```bash
./build-and-copy.sh -c --exp-b12x --force-build
```

---

## Install the recipe + mods

Copy this directory's recipe and mods into the checked-out `b12x` branch.
They must sit under the repo so `run-recipe.py` / `launch-cluster.sh` can resolve
the relative `mods/...` paths:

```bash
# from repo root of spark-vllm-docker (b12x branch)
cp deepseek-v4-flash-eugr.yaml recipes/deepseek-v4-flash-eugr.yaml
cp -r deepseek-v4-flash-eugr/mods .
```

The `dsv4-reasoning-effort-fix` mod is a `run.sh` (executable) that patches
`vllm/tokenizers/deepseek_v4_encoding.py` + `deepseek_v4.py` inside the running
container to give real distinct `low` / `high` / `max` reasoning prefixes.

---

## Launch on the 2-node cluster

From `spark-vllm-docker` (b12x) on the head node:

```bash
python3 run-recipe.py recipes/deepseek-v4-flash-eugr.yaml
```

or with the wrapper:

```bash
./run-recipe.sh recipes/deepseek-v4-flash-eugr.yaml
```

`run-recipe.py` will:
1. Apply the recipe's `mods` (instanttensor hybrid loader + reasoning-effort fix).
2. Launch vLLM on both nodes using the cluster in `.env` (`./.env` / autodiscovery).

To force a build and run in one shot:

```bash
./run-recipe.sh recipes/deepseek-v4-flash-eugr.yaml --setup --force-build
```

### Low-level launch (equivalent, manual)

If you prefer to drive `launch-cluster.sh` directly:

```bash
./launch-cluster.sh \
  --apply-mod mods/instanttensor-hybrid-draft-loader \
  --apply-mod mods/dsv4-reasoning-effort-fix \
  -e VLLM_USE_AOT_COMPILE=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  -e VLLM_USE_MEGA_AOT_ARTIFACT=-1 -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_B12X_WO_PROJECTION=1 -e VLLM_USE_B12X_MHC=1 \
  -e VLLM_USE_B12X_FP8_GEMM=1 -e VLLM_USE_B12X_MOE=1 \
  -e VLLM_USE_B12X_SPARSE_INDEXER=1 -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e B12X_MLA_SM120_UNIFIED=1 -e B12X_MOE_FORCE_A8=1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  exec vllm serve deepseek-ai/DeepSeek-V4-Flash-0731 \
    --port 8000 --host 0.0.0.0 --trust-remote-code \
    --served-model-name deepseek-v4-flash \
    --tensor-parallel-size 2 --kv-cache-dtype fp8 --block-size 256 \
    --max-model-len auto --max-num-seqs 6 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.85 --enable-prefix-caching \
    --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice --reasoning-parser deepseek_v4 \
    --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' \
    --default-chat-template-kwargs.thinking=true \
    --default-chat-template-kwargs.reasoning_effort=high \
    --override-generation-config '{"temperature":1.0,"top_p":0.95}' \
    --load-format instanttensor --moe-backend b12x --linear-backend b12x \
    --attention-backend B12X_MLA_SPARSE --max-cudagraph-capture-size 64 \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
    --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","attention_backend":"B12X_MLA_SPARSE"}'
```

---

## Recipe summary (from `deepseek-v4-flash-eugr.yaml`)

| knob | value |
|---|---|
| model | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| served name | `deepseek-v4-flash` |
| container | `vllm-node-b12x` (build arg `--exp-b12x`) |
| TP | 2 (across 2 nodes) |
| KV cache | fp8, block 256 |
| `max_model_len` | `auto` (full context; reduce `max_num_seqs` to 6 to keep it) |
| `max_num_seqs` | 6 |
| `max_num_batched_tokens` | 8192 |
| `gpu_memory_utilization` | 0.85 |
| spec tokens | 5 (DSpark) |
| load format | `instanttensor` |
| backend | B12X MoE / linear / `B12X_MLA_SPARSE` attention |
| port | 8000 |

**Env vars** (set in the recipe):

```
CUTE_DSL_ARCH=sm_121a  VLLM_USE_AOT_COMPILE=1  VLLM_USE_BREAKABLE_CUDAGRAPH=0
VLLM_USE_MEGA_AOT_ARTIFACT=-1  VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1
VLLM_USE_FLASHINFER_SAMPLER=1  VLLM_USE_B12X_WO_PROJECTION=1
VLLM_USE_B12X_MHC=1  VLLM_USE_B12X_FP8_GEMM=1  VLLM_USE_B12X_MOE=1
VLLM_USE_B12X_SPARSE_INDEXER=1  VLLM_USE_V2_MODEL_RUNNER=1
B12X_MLA_SM120_UNIFIED=1  B12X_MOE_FORCE_A8=1
HF_HUB_OFFLINE=1  TRANSFORMERS_OFFLINE=1
```

---

## Verify it's serving

```bash
curl -sS http://127.0.0.1:8000/v1/models | jq .
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash",
       "messages":[{"role":"user","content":"Say hi"}],
       "thinking":true,"reasoning_effort":"high"}' \
  | jq '.choices[0].message'
```

---

## Graceful shutdown / stop

The cluster is a set of `vllm_node` containers — one on the **head** node and one
on the **worker** node. eugr's `launch-cluster.sh` provides a built-in `stop`
action that sends `docker stop` (SIGTERM → graceful exit → SIGKILL on timeout)
to the container on **both** nodes, using the nodes + container name from `.env`.

On the head node, from the eugr `b12x` checkout:

```bash
cd ~/.cache/sparkrun/eugr-spark-vllm-docker   # or wherever you checked out b12x
./launch-cluster.sh stop
```

This is the graceful way to tear the cluster down. Verify it stopped:

```bash
# head
./launch-cluster.sh status
# or manually both nodes
docker ps --filter name=vllm_node
```

> `status` also works to check health before/after.

**Manual equivalent** (if you only ever tweak the recipe by hand):

```bash
# on head
docker stop vllm_node
# on worker (from the head node)
ssh sdrew@<worker-ip> docker stop vllm_node
```

There's no requirement to `docker rm` — the next `run-recipe.py` launch recreates
the container. The GPU is released once the container stops.

> ⚠️ Only restart after both containers are down. A clean `launch-cluster.sh stop`
> followed by re-running the recipe is the intended restart flow.

---

## Troubleshooting

- **`Check failed: num_tokens > 64` / DSpark draft errors** — remove the
  `--speculative-config` or lower `num_speculative_tokens`.
- **Container not on worker** — `launch-cluster.sh` aborts if the image isn't in
  sync. Sync with `./build-and-copy.sh --no-build -t vllm-node-b12x --copy-to <worker>`.
- **Long safetensors load / hangs** — the recipe uses `--load-format instanttensor`
  to avoid this. If you prefer safetensors, use
  `--load-format safetensors --safetensors-load-strategy lazy`.
- **Stale b12x image** — the shipped `vllm-node-b12x` may predate the 0731 model /
  b12x support. Rebuild: `./build-and-copy.sh -c --exp-b12x` (add `--rebuild-vllm`
  or `--rebuild-flashinfer` to force source rebuilds).
- **Model not found** — ensure the HF cache is present on **both** nodes.

## Reference

- [DSpark forum thread](https://forums.developer.nvidia.com/t/instructions-for-running-deepseek-v4-flash-with-dspark-using-eugrs-repo/376220)
  (post 18 = bernisse's solution)
- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) (branch `b12x`)
