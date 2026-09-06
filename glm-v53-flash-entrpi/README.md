# glm-v53-flash-entrpi — GLM-5.3-Flash EXL3 4bpw (2× DGX Spark, Entrpi lane)

**Adoption** of [Entrpi/glm-5.3-flash-exl3-2x-spark](https://github.com/Entrpi/glm-5.3-flash-exl3-2x-spark)
(v2.3-tier1, see `research.md` watching notes) into this repo's docker-compose
conventions. This is a **different runtime lane** from
[`glm-v53-flash-miaai`](../glm-v53-flash-miaai): Entrpi's own community
`ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark` image built on the
local-inference-lab vLLM fork lineage (b12x runtime, breakable CUDA graphs,
kpool sparse MLA, ring draft-KV, EXL3 fused MoE) — the "permanent fix" lane
the miaai recipe's research.md has been tracking. It carries its own fixes
in the image; **no boot-time hotfixes, no overlay mounts**.

Forum discussion: [GLM-5.3-Flash main thread](https://forums.developer.nvidia.com/t/glm-5-3-flash-320b-total-parameters-18b-active/381350)
(many users report Entrpi's recipe as the smoothest EXL3 path; see the
"Known issues" section for the caveats people actually hit).

## What it serves

| | |
|---|---|
| Model | GLM-5.3-Flash, 320B total / 18B active, EXL3/TR3 4bpw (`brandonmusic/GLM-5.3-Flash-tr3-4bpw`, = `Mia-AiLab/...` mirror bytes) |
| Drafter | DFlash2 **MXFP8** (`local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8`, rev `62f758c0…`) — 1.20 GiB; ring draft-KV |
| Served name | `glm-5.3-flash` | Port: **4000** (repo convention; Entrpi default is 8000) |
| Context | **524,288** default per-request; pool **1,287,194 tokens** @ fp8_ds_mla, KV budget 14.4 GB, GMU 0.85 |
| Speed (measured by Entrpi) | ~30 tok/s prose, 71 structured, TTFT ~0.4s; 4-way aggregate ~47 tok/s |
| Quality (measured by Entrpi) | math_500 91%, GPQA 70%, 133k-retrieval 10/10, spec-decode lossless up to argmax ties |
| Modalities | text, images, tool calling (glm47 parser), optional reasoning (off by default) |

## Choosing a configuration (Entrpi's validated rows)

All are `.env` edits (one line per knob); the defaults = their validated
production config:

| You mostly want | Set | You get | You give up |
|---|---|---|---|
| **Long documents, a few users** (default) | (nothing) | 1,287,194-token pool, 524k requests, fastest single-stream | — |
| **Snappy chat while others upload** | `MIXED_PREFILL_DECODE_WEIGHT=1.0 MIXED_PREFILL_CAP=512` | chats keep 85–91% speed during a cold 100k upload, ~0.6s pauses | document reading −20–35% while chats active |
| **1M-token requests** | `MAX_LEN=1048576 KV_DTYPE=nvfp4_ds_mla VLLM_NVFP4_MLA_DYNAMIC_SCALE=1 MNBT=4096` | 2,144,814-token pool; requests up to 1M, two at once | ~15 min to read a full 1M prompt; small KV quality step (math 88 vs 91) |
| **Many concurrent short sessions** (agents) | `MAX_LEN=131072 MAX_SEQS=12 MNBT=4096 SPEC=none MIXED_PREFILL_DECODE_WEIGHT=1.0 MIXED_PREFILL_CAP=512` | 12–16 streams at 88–103 tok/s aggregate; +44% pool (1,858,451) with spec off | per-stream speed; 131k cap |
| **Unquantized KV** | `MAX_LEN=131072 KV_DTYPE= ATTN_BACKEND= SKIP_MM_PROFILING=0 MAX_SEQS=6` | 520,470-token pool, no 8-bit KV | the long banks |
| **Drafter fallback** | `MTP=4` | model's built-in MTP head; drafter not loaded | ~20% slower decode |

## Deploy

Pre-launch ritual (per node — GB10 unified memory swap-wedges instead of
OOMing; Entrpi's launcher does this before every start):
```bash
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches   # hot page cache at load time wedges the box
```

```bash
# 0) one-time provisioning (both nodes)
# 0a) drafter (1.2 GiB) — sparks have no hf CLI; use a venv:
python3 -m venv /tmp/hfvenv && /tmp/hfvenv/bin/pip install -q huggingface_hub
/tmp/hfvenv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8',
                  revision='62f758c0a0e19b9cb76fc098c911b8ed76daff5b',
                  local_dir='/home/sdrew/models/glm53-dflash2-mxfp8')"
# 0b) weights — the EXL3 bytes are already cached as the Mia-AiLab HF
#     snapshot (brandonmusic mirror, same 120 shards). Materialize the flat
#     dir the loader wants with hardlinks (zero extra space):
SN=/home/sdrew/.cache/huggingface/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw/snapshots/25a44fdbf16862a46b7cc9921142c6c81350af2f
mkdir -p /home/sdrew/models/glm53-exl3
cd "$SN" && for f in *; do [ -f "$f" ] && ln "$(readlink -f "$f")" /home/sdrew/models/glm53-exl3/"$f"; done

# 1) worker (rank 1) first, then leader ~35 s later:
docker compose --env-file .env --env-file .env.node1 up -d
docker compose --env-file .env --env-file .env.node0 up -d

# 2) verify (leader):
curl -s http://127.0.0.1:4000/v1/models    # "id":"glm-5.3-flash"
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:4000/health   # 200

# 3) optional warm-up: Entrpi ships scripts/glm53-warmup.sh in the kit repo
#    (compiles hot shapes so the first real request doesn't pay ~7s)
```

Memory preflight: Entrpi validates on lightly loaded headless boxes (< 6 GB
system memory in use before launch). `free -m` before starting; on a busier
box, lower `KV_CACHE_MEMORY` (~90k pool tokens per GB) — do not raise the
budget without re-measuring floors (a 13.4e9 pin on v2.1 dropped the floor
to 2.26 GiB).

## Boot markers (leader log)

```bash
docker logs glm53-exl3-entrpi 2>&1 | grep -F "GID auto-detect"
#   -> "GID auto-detect: NCCL_IB_GID_INDEX=N" — GID found (not the error dump)

docker logs glm53-exl3-entrpi 2>&1 | grep -F "GPU KV cache size"
#   -> ~1,287,194 tokens / 2.53 banks @524k default = healthy
#      (pool scales with the config row; the README table has the numbers)

docker logs glm53-exl3-entrpi 2>&1 | grep -F "Setting attention block size"
#   -> "Setting attention block size to 4608" (fp8 KV auto-bump from 2304)
```

## Known issues (from the forum thread + kit docs)

- **install.sh traps (N/A here).** The kit's `install.sh` "tries to be too
  smart" — caching copies, restoring stale `~/.glm53-serve.env` on failed
  starts, leaving you perplexed (0rand, post 259). This port **does not use
  install.sh**: the compose file is the single source of config. The
  launcher's `MEM_USED_MAX_GB` preflight, drop-caches ritual, and GID
  auto-detect are preserved in the compose command block.
- **Runtime reliability (still applies).** One reporter ran Entrpi's recipe
  >24h: mostly fine, but saw one random stop and one case of degenerate
  thinking looping (klement, post 262). Worth monitoring reasoning loops;
  the `MTP=4` fallback row above exists for drafter misbehavior.
- **Rebuild lane.** Entrpi's own image is community-built from the
  local-inference-lab fork. We run the public image digest-pinned, no local
  rebuild, no overlays. If you need a kernel-level fix, use the kit's
  hotfix hooks (`~/glm53-hotfix*` overlays) — the launcher binds them over
  the image; our compose does not (documented, not vendored).
- **One NIC vs two.** Entrpi validates with a single rail interface per
  box; this cluster's convention (all recipes here) uses both CX7 ports via
  `IB_PORTS`. Our compose sets `NCCL_IB_HCA` to both. If you see NCCL
  errors with the dual-HCA set, drop `roceP2p1s0f0` from `IB_PORTS` to match
  Entrpi's validated single-HCA topology.
- **Driver generation (09-05 watch, issue #4/#114).** NVIDIA 610.43.02
  costs ~4 GiB more unified memory and is nondeterministic at boot
  (0/4 launches); 580.173.02 is stable. This cluster is on 580.173.02 —
  don't "upgrade" blindly; re-check `nvidia-smi` if a DGX update lands.
- **Vision+text concurrency (09-05 watch, forum 381350/273).** A pair of
  simultaneous requests — one text, one image — crashed the EXL3 engine
  fatally (`CUDA_ERROR_NOT_PERMITTED` at DeepGEMM mhc_pre_tilelang) on a
  config dump identical to ours (exl3 + dflash2 k=7 + fp8_ds_mla +
  instanttensor). Test concurrent vision+text before relying on it.
- **Benchmark discipline (09-05 watch, forum 382099).** Under spec
  decode, long-prefill TTFT can alternate ~2× after mixed workloads
  (engine-state-dependent; restart clears). GPU util% lies on stall-bound
  steps — power draw is the honest signal. Always restart between config
  A/B runs and read cumulative `vllm:prompt_tokens_total`.

## Switching profiles (the short version)

`.env` edit → redeploy (both nodes: `docker compose --env-file .env
--env-file .env.node{1,0} down`; worker up, leader ~35 s later). Full
table + tradeoffs in the "Choosing a configuration" section above and
`research.md`. The three rows you'll actually use:

- **Default (as shipped):** 524k requests, 1.29M pool, ~30 tok/s prose,
  MAX_SEQS=4.
- **Agentic (multi-session/subagents — the pattern that timed out on the
  other lane):** add `MAX_LEN=131072 MAX_SEQS=12 MNBT=4096 SPEC=none
  MIXED_PREFILL_DECODE_WEIGHT=1.0 MIXED_PREFILL_CAP=512` → 12–16 streams
  at 88–103 agg tok/s, pool 1,858,451.
- **1M requests:** `MAX_LEN=1048576 KV_DTYPE=nvfp4_ds_mla
  VLLM_NVFP4_MLA_DYNAMIC_SCALE=1 MNBT=4096` → 2,144,814 pool, two 1M
  requests at once, but ~15 min cold 1M read and a small KV quality step.

Knobs are read by the compose command block and forwarded to `vllm
serve` exactly as Entrpi's launcher does; every knob is one `.env`
line.

## Deviations from Entrpi's kit (this repo's conventions)

| | Entrpi kit | ours | why |
|---|---|---|---|
| port | 8000 | **4000** | repo/omp convention |
| mechanism | install.sh + `~/.glm53-serve.env` + launcher scripts | docker-compose (`.env`/`.env.node0/1`) | repo convention |
| GID | launcher auto-detect (sysfs) | same, in compose command block | same method |
| NICs | one per box | both (`IB_PORTS`) | this cluster's validated set |
| drafter | MXFP8 (their default) | same | — |

## When to use this vs `glm-v53-flash-miaai`

Both serve the same model bytes and the same served name on :4000 (only one
recipe runs at a time). Differences:

- **This lane**: self-consistent image with the fixes baked in (builder/
  detector/accounting in sync — no `hotfix_kv_check_glm5.py`), ring draft-KV
  (32% larger pool at the same budget than the MiaAI drafter slot-share
  design per their equivalence table), fine-grained prefix reuse (512-token
  hash vs page-aligned), measured 91 math_500 / 70 GPQA, ~30 tok/s prose.
- **miaai lane**: hotfix-maintained (two fail-closed boot patches: KV
  accounting + kpool tail), 1M-native profile in `.env` by default, and the
  long-prefill-threshold/mixed-prefill-gate tuning from the 09-04 pass.
- Practical rule of thumb: if you want a proven, measured, low-maintenance
  EXL3 server → **entrpi**. If you need 1M-context out of the box or want the
  hotfix-audited config → **miaai**. Both are one `.env` switch from each
  other.

## References

- `research.md` in this directory — maintenance notes, profile system,
  upstream watchlist (incl. the 09-05 forum risks).
- [Entrpi/glm-5.3-flash-exl3-2x-spark](https://github.com/Entrpi/glm-5.3-flash-exl3-2x-spark)
  — README (validated config table + measured performance), `docs/FINDINGS.md`
  (fork provenance + what does NOT help; read before speculative tuning),
  `docs/COMPARISON.md` (vs the parallel MiaAI recipe).
- Forum: [GLM-5.3-Flash 320B thread](https://forums.developer.nvidia.com/t/glm-5-3-flash-320b-total-parameters-18b-active/381350)
- Weights: [brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)
  (also cached as the `Mia-AiLab` mirror, byte-identical)
- Drafter: [local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8)
  (CC BY-NC-ND 4.0 — research/eval; downloaded from source, not redistributed)

## GPU contention

Serves on all 2 GPUs per node. Tear down any other model container (DS4
vision, miaai GLM, etc.) on BOTH nodes before starting. One recipe at a time.
