# deepseek-v4-flash-aiden-sparkrun

DeepSeek-V4-Flash-0731, **1M context**, on 2x DGX Spark, served ENTIRELY through
`sparkrun` — running the proven **aiden 3.75 image** directly (no rebuild).

This supersedes the old `deepseek-v4-flash-sparkrun` recipe, which told sparkrun
to run the **eugr source build** (`vllm-node-b12x --exp-b12x`). That build lacked
aiden's `FLASHINFER_MLA_SPARSE_DSV4` sparse-attention backend, so it could only
reach ~340k context. The aiden **prebuilt** image already contains that backend
(and is itself literally a sparkrun image, `aidendle94/sparkrun-vllm-ds4-gb10`),
so the clean fix is: **point sparkrun at the aiden image and stop rebuilding.**

## Why this works (research conclusion)

| Concern | Answer |
|---|---|
| "sparkrun expects image built a certain way" | Only when using `mods:` + no `builder`, or `build_args:`. We **pin `builder: docker-pull`** so `mods:` never routes to the eugr source build. |
| Image build needed? | **No.** `container:` is the aiden image (docker.io pullable). `docker-pull` distributes it as-is. |
| Encoder / reasoning-effort / tool-arg fix | Applied as a **mod** (builder-agnostic `pre_exec`): copies `encoding_dsv4.py` + `deepseek_v4_wrapper.py` over the image's `/opt/venv/.../vllm/tokenizers/` before `vllm serve` — the sparkrun equivalent of aiden's bind-mount overlay. |
| 1M context | Same aiden flags: `--attention-backend FLASHINFER_MLA_SPARSE_DSV4 --kv-cache-dtype fp8 --max-model-len 1048576`. sparkrun's VRAM estimator will warn "max_model_len exceeds KV budget" — that estimate is non-sparse and **ignored**; aiden's sparse KV is what makes 1M fit. |
| Root | **Baked into the recipe** — `executor_config` sets `privileged: true` + `security_opt/cap_add/user: null`, which outranks sparkrun's rootless defaults, so the container runs as root **with no `--rootful` flag needed**. |
| Cluster flags | sparkrun's `vllm-distributed` runtime appends `--nnodes 2 --node-rank --master-addr --master-port 25000` (+`--headless` on the worker) automatically — do NOT put them in the command. |
| HF model | sparkrun mounts host `~/.cache/huggingface` → `/cache/huggingface`; recipe sets `HF_HOME=/cache/huggingface`. 0731 model is already cached on both nodes (offline). |
| RoCE/NCCL | Set explicitly from aiden's proven `.env` (recipe env always wins) so the launch never depends on sparkrun's IB auto-detect. |

## Run (on the leader node)

Hosts do **not** need to be passed: sparkrun uses the saved **default cluster
`spark`** (`spark-0f0b` head, `spark-6d14` worker). The earlier launch just
showed `-H 192.168.0.170,192.168.0.171` explicitly — that was for clarity, not a
requirement. You can omit `-H` (default cluster), or use `--cluster spark`, or
pass `-H`/`--hosts-file` to override. Rootful is baked into `executor_config`,
so no `--rootful` flag is needed either.

```bash
cd ~/code/spark-recipes            # on node0 (192.168.0.170)

# 0) IMPORTANT: aiden/tonyd2 share the 2 GPUs per node with this recipe.
#    Stop the currently-running aiden container first (sparkrun uses its own
#    container names, but the GPUs can only host one recipe at a time).
#    e.g. on BOTH nodes: docker compose -f deepseek-v4-flash-aiden/docker-compose.yml down

# 1) Validate / inspect (no containers started)
sparkrun recipe validate deepseek-v4-flash-aiden-sparkrun/deepseek-v4-flash-aiden-sparkrun.yaml
sparkrun run deepseek-v4-flash-aiden-sparkrun/deepseek-v4-flash-aiden-sparkrun.yaml -n

# 2) Launch (detaches by default)
sparkrun run deepseek-v4-flash-aiden-sparkrun/deepseek-v4-flash-aiden-sparkrun.yaml
```

Boot takes ~7–8 min (155 GiB model + AOT compile + warmup) — same as aiden.

## Verify

```bash
sparkrun logs deepseek-v4-flash-aiden-sparkrun
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health          # 200
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool                # max_model_len=1048576
```
The `/v1/models` `max_model_len: 1048576` is the 1M reachability proof (logs show
`GPU KV cache size` / `Maximum concurrency for 1,048,576 tokens per request`).

## Stop / status

```bash
sparkrun status
sparkrun stop deepseek-v4-flash-aiden-sparkrun
```

## Files

```
deepseek-v4-flash-aiden-sparkrun/
├── deepseek-v4-flash-aiden-sparkrun.yaml  # sparkrun recipe
├── README.md
└── mods/
    └── aiden-encoder-overlay/
        ├── run.sh                  # installs both overlays into /opt/venv vLLM
        ├── encoding_dsv4.py        # official 0731 encoder + tool-arg repairs
        └── deepseek_v4_wrapper.py  # corrected low/high/max reasoning routing
```

The two overlay `.py` files are byte-for-byte the ones used by the
`deepseek-v4-flash-aiden` recipe (hardened versions from
`aiden-encoder-toolarg-fix`).

## Phase-2 knobs / caveats (if live test needs adjustment)

- **VLLM_HOST_IP** — not set (it is per-node). If the 2-node rendezvous picks the
  wrong interface, add it per node (node0 `192.168.0.170`, node1 `192.168.0.171`).
- **AOT/jit persistence** — sparkrun only persists the HF cache volume
  (`/cache/huggingface`), not aiden's extra `/cache` (jit/tilelang) volume, so each
  cold start recompiles AOT (~the boot time). Correctness unaffected; can add a
  volume later if restarts are frequent.
- **Start order** — sparkrun launches its own ranked containers (`_node_0`,
  `_node_1`) and runs the serve steps (head then worker) — no manual node0/node1
  sequencing needed.
