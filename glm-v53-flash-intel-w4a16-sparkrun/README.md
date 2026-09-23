# glm-v53-flash-intel-w4a16-sparkrun

Serve **GLM-5.3-Flash** (Intel W4A16 AutoRound quant) on a 2-node DGX Spark
cluster at **1M-token context** with **native MTP3** speculative decoding and
**PMU128** fine-grained prefix caching — launched and managed through
[`sparkrun`](https://sparkrun.dev).

This is the sparkrun-native port of the
[`glm-v53-flash-intel-w4a16`](../glm-v53-flash-intel-w4a16) docker-compose
recipe. vLLM's `INCConfig` loads the Intel `quant_method: "auto-round"`
checkpoint natively, so the raw HF snapshot is served directly — no weight
preparation step.

- **Model:** `Intel/GLM-5.3-Flash-W4A16-AutoRound` (320B total / 18B active,
  INT4 W4A16 AutoRound, loaded natively by `INCConfig`)
- **Served as:** `glm-5.3-flash` on port `8000` (the model-name proxy on
  :4000 serves it as `spark-llm`)
- **Nodes:** 2 (tensor parallel 2 — one GPU per node)
- **Image:** `ghcr.io/taoofshawn/vllm-glm53-intel-w4a16:glm53flash-pmu128-mtp3`
  (public, digest-pinned in the recipe) — the official model-recipe base
  (`vllm/vllm-openai:glm53-flash-arm64-cu130`, vLLM
  `0.28.1rc1.dev580+g385dce36b`, CUDA 13.0 aarch64) plus the SM121 patch
  stack and the fixed chat template **baked in**, so the recipe needs no mods
  and no rebuild. Pulled automatically; sparkrun distributes it to both nodes.

## Prerequisites

- Two DGX Sparks connected (management network + the RoCE crossover cable)
- [sparkrun](https://sparkrun.dev/getting-started/installation/) installed on the **leader** node
- This repo cloned on the **leader** node
- Docker working on both nodes.
- ~60 GB free per node for the model weights, and enough container-disk for
  the image (~30 GB).

> **GPU contention:** this recipe uses all reserved GPUs on both nodes. Stop
> any currently-running model (`sparkrun status`, then stop it) before
> launching.

## One-time setup

### 1) Download the model (BOTH nodes)

Install [huggingface cli](https://huggingface.co/docs/huggingface_hub/main/en/installation#install-the-hugging-face-cli)

```bash
hf download Intel/GLM-5.3-Flash-W4A16-AutoRound \
    --revision 5eee1846f0321058ed73745f9aa16f2aaf0fc0a0
```

Run on **both** nodes — tensor parallel loads ranks from local disk. The
recipe serves the HF repo ID + revision offline, so `/v1/models` reports
`root: Intel/GLM-5.3-Flash-W4A16-AutoRound`.

## Run

On the leader, from the repo root:

```bash

# 1) validate the recipe and preview what would happen (starts nothing):
sparkrun recipe validate glm-v53-flash-intel-w4a16-sparkrun/glm-v53-flash-intel-w4a16-sparkrun.yaml
sparkrun run glm-v53-flash-intel-w4a16-sparkrun/glm-v53-flash-intel-w4a16-sparkrun.yaml -n

# 2) launch:
sparkrun run glm-v53-flash-intel-w4a16-sparkrun/glm-v53-flash-intel-w4a16-sparkrun.yaml
```

sparkrun handles the start order itself (head first, workers after the
rendezvous port opens) and auto-detects the RoCE networking per node —
including the NCCL GID index — so there is no "worker first" ritual like the
compose recipes.

Cold boot ≈ 14 minutes (weight load + engine init; JIT caches persist under
the HF cache so later boots are faster). Never benchmark right after boot.

## Verify it's up

```bash
sparkrun status
sparkrun logs <id>          # follow startup

# health + model metadata once booted (on the leader):
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health   # expect 200
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

Look for:

```json
"id": "glm-5.3-flash",
"root": "Intel/GLM-5.3-Flash-W4A16-AutoRound",
"max_model_len": 1048576
```

Useful boot-log markers (`sparkrun logs`):

- `GID auto-detect` / RoCE v2 GID lines from sparkrun's IB detection —
  networking wired
- `SpeculativeConfig(method='mtp', ... num_speculative_tokens: 3 ...)` —
  MTP3 active
- `Using 'MARLIN' WNA16 MoE backend` — oracle-selected MoE backend (do NOT
  force `--moe-backend marlin`; it FATALs on the BF16 nextn-layer experts)
- `GPU KV cache size` → ~1.92M tokens (13.5 GB fp8 pin at 1M ctx)
- `Using FLASHINFER_MLA_SPARSE_SM90 attention backend` — the sparse-MLA
  backend (fa2 on SM121)

## Talk to it

Direct (port 8000):

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash",
       "messages":[{"role":"user","content":"Say hi in one word"}],
       "max_tokens":32}' \
  | jq -r '.choices[0].message.content'
```

Thinking is ON at reasoning effort `high` by default (server-side pins:
temp 1.0 / top_p 0.95); clients can override per request via
`chat_template_kwargs` or the OpenAI `reasoning_effort` field. Tool calling
uses the `glm47` parser. PMU128 check: send the same >128-token prompt twice
with `"stream_options":{"include_usage":true}` — the second response's
`usage.prompt_tokens_details.cached_tokens` should be ≈ the prompt length
rounded down to a 128 multiple.

## Stop / status

```bash
sparkrun status
sparkrun stop <id>
```

## Files

```
glm-v53-flash-intel-w4a16-sparkrun/
├── glm-v53-flash-intel-w4a16-sparkrun.yaml   # the sparkrun recipe
├── README.md
└── research.md                               # provenance + decisions
```

## Tuning

Each row below is a one-line change to the recipe's `defaults` (then relaunch):

| You want | Set | Effect |
|---|---|---|
| smaller KV pool | `kv_cache_memory_bytes: 9663676416` | 9 GiB pin → ~1.34M-token pool |
| faster boots (staging) | `max_model_len: 262144` + `kv_cache_memory_bytes: 3221225472` | 3 GiB pin |
| profiler-sized KV | delete `kv_cache_memory_bytes` | profiler sizes the pool |
| disable CUDA graphs | add `--enforce-eager` to `command` | open A/B; neutral/better per one source |

Don't raise the KV pin above 13.5 GB without re-verifying CUDA-graph capture
and host headroom through a ~950K prefill — the parent recipe measured a
14.0 GB bump at ~12% c4 aggregate decode cost on this stack.

## Gotchas

- **KV pin trap:** `--kv-cache-memory-bytes` skips the profiler's activation
  check — too high kills the engine on the first long prompt.
- **Do NOT force `--moe-backend marlin`:** the checkpoint is MIXED (45
  quantized W4A16 layers + the BF16 nextn-layer experts); a global `marlin`
  FATALs ("not supported for unquantized MoE"). Leave the backend unset —
  vLLM's oracle auto-selects per layer (MARLIN for the W4A16 experts).
- **Block size stays 2304** (4608/1152 break prefix hits). The PMU128
  `--prefix-match-unit 128` is what gives fine-grained caching.
- **Single prompts ≳ ~310K tokens** have hung hosts in upstream tests; 1M
  `max_model_len` ≠ 1M single-request read today.
- **Vision + text concurrency** crashed a sibling profile; test the combo
  before relying on it.
- **Cold JIT caches:** torchinductor/tilelang caches persist under the HF
  cache mount; if you wipe it, the next boot recompiles (slower).
- **Driver:** cluster is validated on NVIDIA 580.173.02 — don't upgrade
  blindly.
