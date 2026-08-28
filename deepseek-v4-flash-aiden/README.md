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
  curl -sS http://127.0.0.1:4000/v1/chat/completions/render \
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
verified absent by grep — so 3.75 is stable for this workload. The tonyd2wild
fork has the same constraint.

To make 3.8 usable you would either (a) disable DSpark (`--speculative-config`),
or (b) patch `dspark_proposer.py` to degrade gracefully on non-uniform lengths.

**Multi-node start order:** start the **leader (rank 0) first** so its TCP store
on `master_addr:25000` is up before the follower connects. Starting the follower
first caused `DistStoreError: 1/2 clients` / `Connection reset by peer` on a
restart.

## Recipe updates (2026-08-15) — aiden 3.75 unchanged, config refreshed

Reviewed against the tonyd2wild recipe and the aiden/DS4F-DSpark forum threads
(372268, 374846, 376220, 378824, 378890). No image change — `production-3.75`
already carries every *engine-level* fix (DSpark shared-expert loader mapping,
Patch-3 cold-start garble guard, stop-string suppression; see below). Config
tunables refreshed on evidence:

- **Agent-serving profile:** `max-num-seqs 6` / `max-num-batched-tokens 2048`
  (was 16 / 16384). Massively larger KV pool + decode fairness under concurrent
  agent traffic (validated on the sparkrun port; SvangenStudios 378890; forum
  372268 #678 "2048 for better agentic workout"). If draft acceptance ever drops
  under load, raise the batch toward 8192–16384 (acceptance-optimal per #687).
- **GPU memory utilization stays at `0.83`** (deliberate). The aiden 3.75 forum
  validated 0.87–0.90 as *optional headroom* (0.90 → 3.1M-token KV pool #677;
  0.85→0.90 = +46% KV #682), but it buys no per-session speed and the operator
  prefers 0.83 after seeing degradation on long sessions with high settings. If
  more long-context concurrency is ever needed, raising to 0.88–0.90 is a
  one-line, community-validated lever. Do **not** drop toward 0.78 for
  "stability" — that is NVFP4/tonyd2wild-specific advice.
- **Known "slow after a long session" symptom (not swap — checked):** if it
  recurs, troubleshoot in this order before touching GMU: (1) GB10 lockstep
  power-state trap — a node stuck at ~22 W / ~2.0 GHz throttles the TP=2 pair
  (`nvidia-smi -q -d POWER|CLOCK`; fix `nvidia-smi -lgc <max>` on BOTH nodes, or
  a full power cycle); (2) KV starvation under concurrency — engine metrics show
  near-0 gen tok/s with high KV usage and spec acceptance collapse (notably
  positions 2-4); (3) driver regressions (580.159.03 measured slower on GB10);
  (4) "cutout mode" one-die-underclock glitch → unplug ≥1 min. Restarting the
  engine recovers KV-starvation/degradation cases that GMU alone cannot fix.
- **`reasoning_effort` default `high`** (was low). #520 A/B: +7 tool tests passed
  for +5.5% wall time / +5.8% tokens. `max` measured no better and adds a safety
  regression (#639) — keep `max` per-request only. Our encoder overlay makes the
  three levels actually correct (pre-fix, high silently ran as low).
- **CUDA-graph steady-state capture:** explicit `cudagraph_capture_sizes` now
  include 36 (= 6 seqs × (k+1), active at k=5) and 30 (k=4 fallback). The
  auto-generated list omitted them — a missed hot shape truncates graph replay
  at concurrency (bakeoff + PR#5: ~+9–14% at c4–c6).
- **DSpark spec config** now explicitly sets `"moe_backend":"b12x"` (draft must
  use the native B12X MoE oracle, not flashinfer_b12x — srivatsa1 378824).
  k was **4** with `draft_sample_method:"probabilistic"` on 08-15 (#513
  "3=balanced, 5=code-heavy"; #687 best at 4); updated 08-16 to **k=5 + greedy
  draft** (see "Recipe updates (2026-08-16)" below). Drafter block is 5 (k ≤ 5
  or a multiple of 5); the 36 shape is already in the capture list.
- **`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`** — stop reserving full graph
  memory in the profiler; KV gets the budget (verified honored in this build).
  Plus explicit parity envs (`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`,
  `VLLM_DSPARK_CONFIDENCE_SCHEDULER=off`, `_THRESHOLD=0.0`,
  `VLLM_DSPARK_FUSED_MARKOV_ARGMAX=0` — all already the build's defaults; kept
  explicit for diagnosing from the container env).
- **`--reasoning-config`** with explicit `" thinking"` / `" response"` start/end
  markers (tonyd2wild + 372268 #119/#184) so thinking never leaks into content.

## Recipe updates (2026-08-16) — k=5 + greedy draft

Follow-up to the 08-15 refresh after comparing against `0rand/DeepSeek-v4-DSpark-Aidendle94-GB10-ServingStack`
(the same `production-3.75` image + official 0731 GA checkpoint, validated on a
live 2-node cluster). No image change; no temperature / top_p / GMU / serving-profile
changes — defaults stay: temp 1.0 / top_p 0.95 / GMU 0.83 / agent profile 6 & 2048.

- **Spec tokens 4 → 5, draft `probabilistic` → `greedy`.** The 0rand GA stack
  validated k=5 with `draft_sample_method:"greedy"` on the official 0731
  checkpoint (tool-eval-bench 93/100; steadier, higher acceptance pre-GA vs GA
  shift noted in their README). Drafter block = 5, so k=5 is the natural max
  (k ≤ 5 or a multiple of 5). The steady-state decode shape becomes
  6 × (5+1) = **36**, which is already in `cudagraph_capture_sizes`.
- **Revert path:** every prior choice is one flag back — `SPEC_TOKENS: 4` and
  `draft_sample_method:"probabilistic"` (the aiden-forum default, 372268
  #513/#687). Revert only if a live A/B shows acceptance or answer-quality
  regression.

**Engine-level fixes already in 3.75 (verified by inspection):**
`stacked_params_mapping` includes the shared-expert `gate_up_proj` w1/w3 rows
(the +69% decode fix; tonyd615 378824) — present; scheduler `update_draft_token_ids`
guards `is_prefill_chunk` (Patch 3 cold-start garble) — present; detokenizer
`VLLM_SUPPRESS_STOPS_IN_REASONING` — present. The PR #17 streaming
`tool_calls: []` leak does **not** reproduce on 0.11.2 (verified live: 0/30
deltas) — no port needed.

> **Stay on `production-3.75`.** `production-3.8` upstream has a DSpark
> uniform-length regression (EngineDeadError under heterogeneous agent batches);
> the author himself says "there's not suppose to be a 3.8" (#662). This recipe
> pins the 3.75 digest.

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
