# deepseek-v4-flash-aiden-sparkrun

Serve **DeepSeek-V4-Flash-0731** at **1M-token context** on a 2-node DGX Spark
cluster, run and managed entirely through `sparkrun`.

- **Model:** `deepseek-ai/DeepSeek-V4-Flash-0731`
- **Context:** 1,048,576 tokens (sparse attention + fp8 KV)
- **Nodes:** 2 (tensor parallel), leader + worker
- **Served as:** `deepseek-v4-flash` on port `4000`

## Prerequisites

On **both** nodes:

- The recipe repo is checked out on this branch (`deepseek-v4-flash-aiden-sparkrun`):
  ```bash
  cd ~/code/spark-recipes
  git fetch origin
  git checkout deepseek-v4-flash-aiden-sparkrun
  git pull --ff-only origin deepseek-v4-flash-aiden-sparkrun
  ```
- The container image is present (or pullable): `aidendle94/sparkrun-vllm-ds4-gb10`
- The model `DeepSeek-V4-Flash-0731` is in the local HuggingFace cache
  (`~/.cache/huggingface`), since serving runs offline.
- **GPUs are free.** This recipe uses all GPUs on both nodes, so nothing else
  (any other model container) can be running at the same time.

## Configure a cluster (only once)

Sparkrun needs a cluster of the two nodes. If one isn't set up yet:

```bash
sparkrun cluster create spark \
  -H spark-0f0b.shawndo.intra,spark-6d14.shawndo.intra
sparkrun cluster set-default spark
```

> Hosts don't have to be spelled out on every run — sparkrun uses the default
> cluster. To override, pass `-H HOST1,HOST2` or `--cluster NAME`.

## Run

On the **leader node** (`spark-0f0b`), from the repo root:

```bash
cd ~/code/spark-recipes

# 1) Check the recipe is valid and see what would happen (no containers started)
sparkrun recipe validate deepseek-v4-flash-aiden-sparkrun/deepseek-v4-flash-aiden-sparkrun.yaml
sparkrun run deepseek-v4-flash-aiden-sparkrun/deepseek-v4-flash-aiden-sparkrun.yaml -n

# 2) Launch (runs in the background)
sparkrun run deepseek-v4-flash-aiden-sparkrun/deepseek-v4-flash-aiden-sparkrun.yaml
```

Boot takes roughly **7–8 minutes** (loading the ~155 GiB model, AOT compilation,
and warmup). You can watch progress with `sparkrun logs`.

## Verify it's up

```bash
sparkrun status
sparkrun logs <id>   # follow startup

# Health + model metadata once booted:
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:4000/health   # expect 200
curl -s http://127.0.0.1:4000/v1/models | python3 -m json.tool           # see below
```

Look for:

```json
"id": "deepseek-v4-flash",
"max_model_len": 1048576
```

A `max_model_len` of `1048576` is the 1M-context confirmation. The engine logs
also print `GPU KV cache size` and `Maximum concurrency for 1,048,576 tokens per
request`.

## Talk to it

```bash
curl -s http://127.0.0.1:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash",
       "messages":[{"role":"user","content":"Say hi"}],
       "max_tokens":64,"thinking":true,"reasoning_effort":"low"}' \
  | jq '.choices[0].message'
```

Reasoning is surfaced in the assistant message's `reasoning` field (see below).

## Reasoning field

This model build emits chain-of-thought in the assistant message's
**`reasoning`** field (the native field name for this vLLM image); `content`
holds the final answer. Clients that read `reasoning` (the current standard)
work directly against the server on port `4000`.

```bash
curl -s http://127.0.0.1:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash",
       "messages":[{"role":"user","content":"Say hi"}],
       "max_tokens":64,"thinking":true,"reasoning_effort":"low"}' \
  | jq '.choices[0].message'   # reasoning is populated
```

## Stop / status

```bash
sparkrun status
sparkrun stop <id>
```

## Files

```
deepseek-v4-flash-aiden-sparkrun/
├── deepseek-v4-flash-aiden-sparkrun.yaml   # sparkrun recipe (image, flags, env, executor)
├── README.md
└── mods/
    └── aiden-encoder-overlay/
        ├── run.sh                  # loads the two overlays before vllm serve
        ├── encoding_dsv4.py        # reasoning-effort prompts + tool-arg repair
        └── deepseek_v4_wrapper.py  # low/high/max reasoning routing
```

The `mods/aiden-encoder-overlay` files are applied inside each container before
the server starts; they fix reasoning-effort levels and tool-call argument
handling. They ship with the recipe, so nothing extra to install.

## Notes / tuning

- **Batching (agent profile):** `max-num-batched-tokens 2048` + `max-num-seqs 6`.
  Smaller prefill chunks give much better decode fairness and lower output jitter
  under concurrent agent traffic (at a modest prefill-throughput cost). The
  earlier 16384/16 was tuned more toward high-concurrency batching.
- **GPU memory utilization stays at `0.83`** (the recipe's
  `gpu_memory_utilization`). Do **not** lower it expecting stability — weights
  (~80 GiB/node) dominate, so GMU is a KV-cache lever here, and dropping it could
  shrink the KV pool below what's needed for 1M. **However:** if the server
  crashes or dies under traffic while you are testing this build, the first knob
  to try is **lowering `gpu_memory_utilization` (e.g. to `0.78`)** and restarting.
  That is the fallback documented in the upstream agent-serving thread.
- **Port:** `4000` (default). Override with `sparkrun run ... --port 8080`.
- **Reasoning effort** defaults to `low`; `thinking` defaults to `true`.
- **Interface caveat:** the multi-node master address comes from sparkrun's host
  detection. If a restart ever fails to rendezvous across the two nodes, set the
  per-node `VLLM_HOST_IP` (leader `192.168.0.170`, worker `192.168.0.171`).
- **Cold starts recompile:** only the HuggingFace cache is persisted as a volume,
  so AOT/JIT artifacts rebuild on each cold start (that's part of the ~7–8 min
  boot). Correctness is unaffected.
