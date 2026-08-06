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
encoder** (`encoding/encoding_dsv4.py`) by bind-mounting it **directly over** the
image's bundled `vllm/tokenizers/deepseek_v4_encoding.py`, so vllm loads it at
import time. We keep the native `deepseek_v4` tokenizer (we do **not** use
`--tokenizer-mode hf`), so tool-call / reasoning parsing stays on the native
path. The official encoder has correct, distinct effort prompts — `low` adds
nothing, `high` = "Absolute maximum…", `max` = "Beyond maximum…" — the bundled
pre-0731 encoder collapses these (low/high → no prefix, max → the *high* text).

> **Note on `DSPARK_ENCODING_FILE`:** the MiaAI-Lab doc recommends setting this
> env var to point the runtime at `encoding/encoding_dsv4.py`. That hook is a
> feature of *their* launcher — verified it is referenced **nowhere** in the
> aiden image, so it is inert here. That is why this recipe uses a direct file
> overlay instead. (If a future aiden image starts honoring it, either mechanism
> works.)

Why not the jinja approach? An earlier attempt mounted a reverse-engineered
`chat_template.jinja` and switched to `--tokenizer-mode hf`. It produced
distinct effort levels but broke tool-call JSON at high/max (the template omits
multi-turn tool-result ordering and re-renders large contexts). The native
encoder is the forum-recommended path and matches this recipe's native parsers.

What changed (already in this recipe):

```yaml
volumes:
  # overlay the image's bundled encoder (fixes low/high/max effort prompts)
  - ./encoding_dsv4.py:/opt/venv/lib/python3.12/site-packages/vllm/tokenizers/deepseek_v4_encoding.py:ro
  # overlay the tokenizer wrapper (fixes pre-0731 low->high collapse + adds off)
  - ./deepseek_v4_wrapper.py:/opt/venv/lib/python3.12/site-packages/vllm/tokenizers/deepseek_v4.py:ro
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
in this recipe's `MODEL_REVISION`). `deepseek_v4_wrapper.py` is the image's own
`tokenizers/deepseek_v4.py` (2026-06-26 eldritch build) with only the
`reasoning_effort` mapping corrected.

> **Current image: `production-3.75`** (vLLM `0.11.2`), pinned by digest
> `sha256:3b4d2b5f…`. This recipe was briefly moved to `production-3.8` (vLLM
> `0.21.1`) then reverted — see "Upgrading to production-3.8" below for why
> and what to watch for. 3.75 and 3.7 are functionally identical (same vLLM
> 0.11.2, pre-0731 encoder/wrapper, `/opt/venv`, `FLASHINFER_MLA_SPARSE_DSV4`)
> and both still need these two overlays.

> **Why the wrapper overlay?** The official encoder fixes the effort **prompts**,
> but the image's wrapper still maps every non-`none`/non-`max` effort to `high`
> (pre-0731 bug) — so `low` rendered the same as `high`. The overlay restores
> `low`→`low`, `high`→`high`, `max`/`xhigh`→`max`, and `none`/`off`→chat (thinking off).

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

1. Remove the `./encoding_dsv4.py` and `./deepseek_v4_wrapper.py` volume overlay
   lines.
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

## Upgrading to production-3.8 (vLLM 0.21) — notes

`production-3.8` upgrades this recipe from vLLM `0.11.2` → `0.21.1rc1`. We tried
it and reverted to 3.75 because of a DSpark regression (below). Those 3.8-only
changes live in this repo's commit range `9326714..56997d8` and are listed here
so a future update can re-apply them:

1. **Image**: `@sha256:3b4d2b5f…` → `@sha256:50b139fb…` (`production-3.8`).
2. **vllm moved `/opt/venv` → `/opt/env`**: overlay mounts and the launcher must
   use `/opt/env/...`. The CLI is at `/opt/env/bin/vllm` and the login shell
   resets PATH, so launch with `exec /opt/env/bin/vllm serve ...` (or export the
   image's PATH).
3. **HF_HOME changed to `/cache/huggingface`** in the 3.8 image env, but this
   recipe mounts the cache at `/root/.cache/huggingface` — set
   `HF_HOME: /root/.cache/huggingface` or the offline model is "not found".
4. **Attention backend renamed**: `FLASHINFER_MLA_SPARSE_DSV4` → `B12X_MLA_SPARSE`.
5. **`VLLM_USE_V2_MODEL_RUNNER` must be `"0"`** (incompatible with the `dspark`
   speculative method in 0.21).
6. Harmless 0.21 warnings: `VLLM_PCIE_ALLREDUCE_BACKEND`,
   `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` are unknown env vars.

**Why we reverted (important):** 3.8 has a DSpark regression —
`dspark_proposer.py::_trim_rejected_target_context` now requires **uniform
per-request effective context lengths after rejection trimming**. Under a batch
of heterogeneous-length requests (which pi's mixed reasoning + tool workload
produces constantly), it raises `ValueError: DSpark currently requires uniform
effective per-request target context lengths ...` and kills the whole engine
(`EngineDeadError`). This check does **not** exist in vLLM 0.11 (3.75/3.7) —
verified absent by grep — so 3.75 is stable for this workload. The tonyd2wild /
eugr fork has the same constraint.

To make 3.8 usable you would either (a) disable DSpark (`--speculative-config`),
or (b) patch `dspark_proposer.py` to degrade gracefully on non-uniform lengths.

**Multi-node start order:** start the **leader (rank 0) first** so its TCP store
on `master_addr:25000` is up before the follower connects. Starting the follower
first caused `DistStoreError: 1/2 clients` / `Connection reset by peer` on a
restart.

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
