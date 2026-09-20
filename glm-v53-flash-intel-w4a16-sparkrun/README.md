# glm-v53-flash-intel-w4a16-sparkrun

Serve **GLM-5.3-Flash** (Intel W4A16 AutoRound quant) on a 2-node DGX Spark
cluster at **1M-token context** with **native MTP3** speculative decoding and
**PMU128** fine-grained prefix caching — launched and managed through
[`sparkrun`](https://sparkrun.dev).

This is the sparkrun-native port of the
[`glm-v53-flash-intel-w4a16`](../glm-v53-flash-intel-w4a16) docker-compose
recipe, **mtp3 profile** (the parent recipe's default and, per its
2026-09-18 on-cluster A/B, its measured-best lane). If you want the DFlash2
alternative lane, use the parent compose recipe.

- **Model:** `Intel/GLM-5.3-Flash-W4A16-AutoRound` (320B total / 18B active,
  INT4 group-128 GPTQ after a one-time "surgery" step — see below)
- **Served as:** `glm-5.3-flash` on port `8000` (the model-name proxy on
  :4000 serves it as `spark-llm`)
- **Nodes:** 2 (tensor parallel 2 — one GPU per node)
- **Image:** `ghcr.io/taoofshawn/vllm-glm53-intel-mtp3-pmu128` (public,
  digest-pinned in the recipe) — the digest-pinned tonyd2wild base plus the
  MTP3/PMU128 patch series and the fixed chat template **baked in**, so the
  recipe needs no mods and no rebuild. Pulled automatically; sparkrun
  distributes it to both nodes.

## Prerequisites

- Two DGX Sparks connected (management network + the RoCE crossover cable)
- [sparkrun](https://sparkrun.dev/getting-started/installation/) installed on the **leader** node
- This repo cloned on the **leader** node
- Docker working on both nodes.
- ~60 GB free per node for the model weights, and enough container-disk for
  the image (~31 GB base).

> **GPU contention:** this recipe uses all reserved GPUs on both nodes. Stop
> any currently-running model (`sparkrun status`, then stop it) before
> launching.

## One-time setup

### 1) Download the model + run the GPTQ surgery (BOTH nodes)

The checkpoint ships as `auto-round` quantization, which no GB10 vLLM build
loads; a small surgery script materializes a servable `gptq-surgery` revision
inside the model's own HF-cache entry (hardlinks only — no extra disk).

Install [huggingface cli](https://huggingface.co/docs/huggingface_hub/main/en/installation#install-the-hugging-face-cli)

```bash
hf download Intel/GLM-5.3-Flash-W4A16-AutoRound \
    --revision 5eee1846f0321058ed73745f9aa16f2aaf0fc0a0

# b) the surgery (idempotent; writes the synthetic "gptq-surgery" revision):
cd glm-v53-flash-intel-w4a16-sparkrun
./prepare-model.sh
```

Run **both** steps on **both** nodes — tensor parallel loads ranks from local
disk. The recipe serves the HF repo ID + the `gptq-surgery` revision offline,
so `/v1/models` reports `root: Intel/GLM-5.3-Flash-W4A16-AutoRound`.

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

Boot takes roughly **8–10 minutes** (~84 GiB weight load + engine init; JIT
caches persist under the HF cache so later boots are faster). Never benchmark
right after boot.

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
- `GPU KV cache size` → ~1.92M tokens (13.5 GB fp8 pin at 1M ctx)
- `Setting attention block size to 4608` — expected (fp8 KV auto-bump from
  block 2304), not an error

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
├── prepare-model.sh                          # GPTQ surgery (run on both nodes)
├── README.md
└── research.md                               # provenance + decisions
```

## Tuning

Each row below is a one-line change to the recipe's `defaults` (then relaunch):

| You want | Set | Effect |
|---|---|---|
| smaller KV pool | `kv_cache_memory: 9663676416` | 9 GiB pin → ~1.34M-token pool |
| faster boots (staging) | `max_model_len: 262144` + `kv_cache_memory: 3221225472` | 3 GiB pin |
| profiler-sized KV (safe fallback) | delete `kv_cache_memory` | profiler sizes the pool |
| disable CUDA graphs | add `--enforce-eager` to `command` | open A/B; neutral/better per one source |

Don't raise the KV pin above 13.5 GB without re-verifying CUDA-graph capture
and host headroom through a ~950K prefill — the parent recipe measured
upstream's 14.0 GB bump at ~12% c4 aggregate decode cost on this stack.

## Gotchas

- **KV pin trap:** `--kv-cache-memory` skips the profiler's activation check —
  too high kills the engine on the first long prompt.
- **`--moe-backend marlin` is mandatory:** `flashinfer_cutlass` won't boot
  W4A16; `triton` boots at ~half speed.
- **Block size stays 2304** (4608/1152 break prefix hits). The PMU128
  `--prefix-match-unit 128` is what gives fine-grained caching.
- **Single prompts ≳ ~310K tokens** have hung hosts in upstream tests; 1M
  `max_model_len` ≠ 1M single-request read today.
- **Vision + text concurrency** crashed a sibling profile; test the combo
  before relying on it.
- **Cold JIT caches:** torchinductor/tilelang caches persist under the HF
  cache mount; if you wipe it, the next boot recompiles (slower).
- **Surgery after re-sync:** if the model files are ever re-downloaded/re-synced
  over the HF cache entry, re-run `./prepare-model.sh` (it is idempotent and
  takes seconds).
- **Driver:** cluster is validated on NVIDIA 580.173.02 — don't upgrade
  blindly.
