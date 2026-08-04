# deepseek-v4-flash-aiden - 1m token context

This is a docker compose file for starting vLLM with DeepSeek-V4-Flash-0731 on a 2-node DGX Spark cluster.  
The repository should be cloned to both nodes and started on the second node first. 

## Prep (both nodes)
```bash
cd ~/spark-recipes/deepseek-v4-flash-aiden/
docker pull $(grep image docker-compose.yml | awk '{print $NF}')
HF_MODEL=$(grep "MODEL_PATH:" docker-compose.yml | awk '{print $NF}')
HF_REVISION=$(grep "MODEL_REVISION:" docker-compose.yml | awk '{print $NF}')
hf download $HF_MODEL --revision $HF_REVISION
```

## Run
```bash
# Node 1 (follower): start first
docker compose --env-file .env --env-file .env.node1 up -d

# Node 0 (leader): start about 30s after
docker compose --env-file .env --env-file .env.node0 up -d
```

## Reasoning-effort + tool-call fix (official native encoder)

The image's native `deepseek_v4` tokenizer collapses reasoning levels: its
bundled (pre-0731) encoder has only one effort prefix (mislabeled `max`,
actually the `high` text) and injects either that or nothing — so requests sent
as `low`/`high`/`max` do **not** get distinct prompts. It also lacks multi-turn
tool-result ordering.

**This branch replaces the bundled encoder with the model's official 0731
encoder** (`encoding/encoding_dsv4.py`) via vLLM's `DSPARK_ENCODING_FILE` hook,
which installs it into vLLM before import on both ranks. We keep the native
`deepseek_v4` tokenizer (we do **not** use `--tokenizer-mode hf`), so tool-call /
reasoning parsing stays on the native path. The official encoder has correct,
distinct effort prompts — `low` adds nothing, `high` = "Absolute maximum…",
`max` = "Beyond maximum…" — and implements multi-turn tool-result sorting.

Why not the jinja approach? An earlier attempt mounted a reverse-engineered
`chat_template.jinja` and switched to `--tokenizer-mode hf`. It produced
distinct effort levels but broke tool-call JSON at high/max (the template omits
multi-turn tool-result ordering and re-renders large contexts). The native
encoder is the forum-recommended path and matches this recipe's native parsers.

What changed (already in this recipe):

```yaml
volumes:
  - ./encoding_dsv4.py:/opt/deepseek/encoding_dsv4.py:ro
environment:
  DSPARK_ENCODING_FILE: /opt/deepseek/encoding_dsv4.py
```
```bash
--tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 \
--enable-auto-tool-choice \
--generation-config vllm \
--default-chat-template-kwargs.thinking=true \
--default-chat-template-kwargs.reasoning_effort=low \
```

`encoding_dsv4.py` is committed in this recipe directory (SHA-256
`abc0d26120250dda0ae077dc64aa28836026e61e970854aaeb792445e6a0dde6`), pinned
from `deepseek-ai/DeepSeek-V4-Flash-0731` path `encoding/encoding_dsv4.py` at
revision `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` (the same revision pinned
in this recipe's `MODEL_REVISION`).

**Verify it is applied** (after restart, adjust model/port as needed):

```bash
for effort in low high max; do
  curl -sS http://127.0.0.1:8000/v1/chat/completions/render \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"deepseek-v4-flash\",\"messages\":[{\"role\":\"user\",\"content\":\"Test\"}],\"add_generation_prompt\":true,\"chat_template_kwargs\":{\"thinking\":true,\"reasoning_effort\":\"$effort\"}}" |
  jq -r "[\"$effort\", (.token_ids | length)] | @tsv"
done
# expect three distinct lengths: low short, high ≠ max (both long) = encoder active
```
Then sanity-check a two-turn tool call (assistant tool call → tool result → user)
to confirm tool-result ordering is preserved (the high/max tool-JSON failure
mode from the jinja attempt is gone).

**Rolling back** (e.g. when a future aiden image ships a correct built-in
encoder), restore the original bits:

1. Remove `DSPARK_ENCODING_FILE` and the `./encoding_dsv4.py:...` volume line.
2. `--default-chat-template-kwargs.reasoning_effort=low` → `=max`
   (`thinking=true` stays).

`encoding_dsv4.py` can stay in the repo (it is inert once unmounted).
The one-step revert is `git checkout main -- deepseek-v4-flash-aiden/` — this
branch's only runtime diff vs main is this fix. This is the exact original state:

```bash
--tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 \
--enable-auto-tool-choice \
--generation-config vllm \
--default-chat-template-kwargs.temperature=$${TEMPERATURE} \
--default-chat-template-kwargs.top_p=$${TOP_P} \
--default-chat-template-kwargs.thinking=true --default-chat-template-kwargs.reasoning_effort=max \
```

## Reference
The [discussion thread](https://forums.developer.nvidia.com/t/deepseek-v4-flash-aiden-recipe-from-reddit-1m-token-session-operational-cuda-12-1-tailored-for-dgx-spark-gb10/372268) for this configuration

## files

| File | Purpose |
|---|---|
| `.env` | Shared config — must be customized |
| `.env.node0` | Per-node overrides for node 0 (leader) |
| `.env.node1` | Per-node overrides for node 1 (follower) |
| `docker-compose.yml` | compose file |


## Finding interface names for .env

### **Ethernet ports** (used for control-plane traffic): These will be used for `ETH_IF` and `ETH_IF2`

```bash
# List all Ethernet interfaces — pick the ones that are "BROADCAST" with link up:
ip addr show | grep -E '^[0-9]+: en'
```
These ports will change depending on if you used the left or right QSFP ports to connect the sparks to each other.

### **RoCE/InfiniBand ports** (used for NCCL data-plane traffic): These will be used for `IB_PORTS` (comma separated)

```bash
# Show RoCE interface names and link state:
ibdev2netdev

# Alternative: list RDMA devices directly:
rdma link show
```

### On DGX Spark each Ethernet port has a matching RoCE port on the same physical port.
Typical mapping:

| Ethernet (`ip addr`) | RoCE (`rdma link show`) |
|---|---|
| `enp1s0f0np0` | `rocep1s0f0` |
| `enP2p1s0f0np0` | `roceP2p1s0f0` |

Update the three variables in `.env` to match your system:

| .env variable | Typical value | What it controls |
|---|---|---|
| `ETH_IF` | `enp1s0f0np0` | GLOO, MPI, TP control-plane traffic |
| `ETH_IF2` | `enP2p1s0f0np0` | Second Ethernet port (for multi-interface NCCL socket) |
| `IB_PORTS` | `rocep1s0f0,roceP2p1s0f0` | NCCL InfiniBand data-plane traffic |
