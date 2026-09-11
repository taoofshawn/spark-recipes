# glm-v53-flash-intel-w4a16 — GLM-5.3-Flash Intel W4A16 AutoRound (2× DGX Spark)

Serve [Intel/GLM-5.3-Flash-W4A16-AutoRound](https://huggingface.co/Intel/GLM-5.3-Flash-W4A16-AutoRound)
(GLM-5.3-Flash quantized to INT4 W4A16) on a 2-node DGX Spark cluster with
speculative decoding and fine-grained prefix caching, at 1M-token context.

- **Model**: GLM-5.3-Flash, 320B total / 18B active, INT4 W4A16 group-128 sym
  GPTQ (Intel AutoRound quant; attention/router/shared-expert/vision/norms
  stay BF16) — requires a one-time surgery step (below), ~82 GiB/rank served.
- **Default serving profile**: `dflash2pmu` — external **DFlash2 k=7** drafter
  + **PMU128** prefix matching (`--prefix-match-unit 128`), patches baked into
  the serving image. KV pool ~1.87M tokens at 1M context (13.5 GB pin, fp8 KV).
- **Port** 8000 (repo convention); served name `glm-5.3-flash`; the
  model-name proxy serves clients `spark-llm` on :4000.
- **Modalities**: text, images/video, tool calling (`glm47` parser), thinking
  ON at effort `high` (explicit server-side pins: temp 1.0 / top_p 0.95 /
  thinking true / reasoning_effort high via `--generation-config vllm` +
  `--override-generation-config` + `--default-chat-template-kwargs`, so
  checkpoint/template defaults can't silently drift them).

## Serving profiles

The recipe ships one **default serving profile** plus two alternatives. All
serve the same model on the same port; a profile = spec-decoding method +
prefix-matching granularity + the image that carries the matching patches.
Switch profiles by setting `LANE` + `IMAGE` (+ the profile's tuning rows)
in `.env`.

| | **dflash2pmu (default)** | dflash2 | mtp3 |
|---|---|---|---|
| Spec decoding | external DFlash2 k=7 drafter, `disable_eagle_block_drop=true` | external DFlash2 k=7 drafter | native MTP3 (`num_speculative_tokens=3`, `disable_eagle_block_drop=true`) — no drafter checkpoint |
| Prefix matching | PMU128: `--prefix-match-unit 128`, retention 0, coordinator/SWA patches baked into the image | block granularity 2304 + a boot-time hybrid-APC patch | PMU128 (same baked-patch family, without the SWA fine-hits patch) |
| Drafter checkpoint | `incoai/GLM-5.3-Flash-DFlash2` @ `bf582e4e…` (CC BY-NC-ND-4.0) | same drafter @ `dc77ff1c…` | none |
| KV pin → pool | 13.5 GB → ~1.81–1.87M tokens | 12.52 GB → 1.75M tokens | 13.5 GB → 1.92M tokens |
| Max seqs | 6 | 8 | 6 |
| Image to build | `dflash2-pmu128/Dockerfile` → `glm53-intel-dflash2-pmu128:20260908` | none (uses the digest-pinned base image directly) | `mtp3-pmu128/Dockerfile` → `glm53-intel-mtp3-pmu128:20260907` |
| When to pick it | agent-style workloads: fine-grained prefix reuse across long repeated instruction blocks + strong draft acceptance | matching upstream measurements made on the block-2304 stack | no drafter download wanted; reasoning-heavy serving |

Why the default profile exists: on the base image, `dflash` spec decoding
plus block-2304 prefix caching loses prefix hits (the drafter's SWA group
zeroes the hybrid min) and only a boot-time patch fixes it. The dflash2pmu
profile bakes a patch series into the image instead (#53388 block-drop,
#53906 coordinator partial hits, scheduler LCM/mamba block alignment, and an
SWA fine-hits patch derived from draft PR #54397), which both fixes the
hybrid min and drops prefix granularity from 2304 tokens to 128 — so
repeated agent prefixes replay at ~128-token precision instead of ~2304.
Because the patches are baked in, the boot-time hybrid-APC patch is skipped
in this profile (the compose command block gates on `LANE`); the SM121
indexer overlay is still mounted in every profile. `prepare-model.sh`
surgery is identical across profiles.

The mtp3 profile replaces the drafter with native MTP3 (built into this
vLLM via the same #53388 patch series): no drafter checkpoint to download,
and per its author it wins reasoning-heavy workloads.

All serving images build from the same digest-pinned base
(`ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2@sha256:4def0ef6…`).
The patch installers are SHA-gated and fail-closed — a build either produces
the exact patched runtime or fails. See `dflash2-pmu128/README.md` /
`mtp3-pmu128/README.md` for build + validation details.

## Deploy

### 0) One-time provisioning (BOTH nodes)

```bash
python3 -m venv /tmp/hfvenv && /tmp/hfvenv/bin/pip install -q -U huggingface_hub

# a) weights — lands in the default HF hub cache (snapshot rev pin):
/tmp/hfvenv/bin/hf download Intel/GLM-5.3-Flash-W4A16-AutoRound \
    --revision 5eee1846f0321058ed73745f9aa16f2aaf0fc0a0

# b) drafter (2.3 GiB, same cache) — the dflash2pmu pin:
/tmp/hfvenv/bin/hf download incoai/GLM-5.3-Flash-DFlash2 \
    --revision bf582e4eacc1810f76656d1811693ff6c6737d2a

# c) the GPTQ surgery (builds $MODEL_HOST_PATH from the snapshot):
./prepare-model.sh            # run on BOTH nodes
```

`prepare-model.sh` is required because the model does not load as shipped:
`auto-round` is not in the GB10 forks' `QUANTIZATION_METHODS`, but the
tensors are plain GPTQ (`auto_round:auto_gptq`, sym, group-128). The script
materializes a serving dir from the HF snapshot (hardlinks, zero extra
space) and swaps `quantization_config` in `config.json`:

```json
{"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": true,
 "desc_act": false, "lm_head": false, "true_sequential": true,
 "dynamic": {"-:<regex>": {} /* = every extra_config BF16 exclusion, 679 rules */}}
```

`-:` rules tell vLLM's AutoGPTQConfig to leave matched modules unquantized
(BF16 load), which is exactly what auto-round's `extra_config` meant
(verified against `vllm/model_executor/layers/quantization/utils/gptq_utils.py`
`get_dynamic_override`). Idempotent; fail-closed.

### 1) Build the serving image (BOTH nodes)

```bash
# base image, digest-pinned (~31 GiB):
docker pull ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6

# default profile image (the patches bake in at build time; SHA-gated):
cd dflash2-pmu128 && docker build -t glm53-intel-dflash2-pmu128:20260908 . && cd ..
```

### 2) Launch — worker (rank 1) FIRST, leader ~35 s later

Pre-launch ritual per node (GB10 unified memory swap-wedges instead of
OOMing; heavy weight-load IO grows the page cache and starves the NVRM
allocator):
```bash
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
```
Optional but measured (+5–7% decode): `./tools/clocks.sh` on each node.

```bash
docker compose --env-file .env --env-file .env.node1 up -d   # worker
docker compose --env-file .env --env-file .env.node0 up -d   # leader
```

The wrong start order hangs the rendezvous (`DistStoreError: 1/2 clients`,
`Connection reset by peer`); always `docker compose down` on BOTH nodes
between relaunches. Cold boot ~8–10 min (weight load + engine init; JIT
caches persist). Never benchmark right after boot.

### 3) Verify (leader)

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health   # 200 — /v1/models returns 200 even with a dead engine
curl -s http://127.0.0.1:8000/v1/models        # "id":"glm-5.3-flash", max_model_len 1048576
```

End-to-end through the model-name proxy (clients use `spark-llm`):
```bash
curl -s http://spark.shawndo.intra:4000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"spark-llm","messages":[{"role":"user","content":"Say hi in one word"}],"max_tokens":32}' | jq -r '.choices[0].message.content'
```

## Boot markers (leader log)

```bash
docker logs glm53-intel-w4a16 2>&1 | grep -F "GID auto-detect"
#   -> "GID auto-detect: NCCL_IB_GID_INDEX=N" (not the error dump)
docker logs glm53-intel-w4a16 2>&1 | grep -F "DFlash2DraftModel"
#   -> "Resolved architecture: DFlash2DraftModel" = drafter wired (dflash2pmu profile)
docker logs glm53-intel-w4a16 2>&1 | grep -F "GPU KV cache size"
#   -> ~1.87M tokens @ 1M ctx (13.5 GB pin); the 12.52 GB row gives ~1.75M
docker logs glm53-intel-w4a16 2>&1 | grep -F "Setting attention block size"
#   -> 4608 (fp8 KV auto-bump from block-size 2304) — expected, NOT 2304
docker logs glm53-intel-w4a16 2>&1 | grep -F "Model loading took"
#   -> ~84 GiB, ~5 min
```

PMU check (dflash2pmu profile): send the same >128-token prompt twice with
`"stream_options": {"include_usage": true}`; both responses should report
`usage.prompt_tokens_details.cached_tokens` ≈ prompt_tokens rounded down to
a 128 multiple (the residue, ~1–130 tokens, is what gets computed). Prompts
shorter than 128 tokens report `cached_tokens=0` — expected; they never fill
a PMU unit. Prompts whose length is an exact multiple of 4608 hit a known
scheduler carve-out and compute the full prompt (0 cache hits) — unfixed
upstream, by design.

## Switching profiles

All profile switching is `.env` edits (then `docker compose down` both
nodes, and `up` worker-first again). Default is dflash2pmu; to switch:

```bash
# mtp3 profile:
LANE=mtp3
IMAGE=glm53-intel-mtp3-pmu128:20260907     # built from mtp3-pmu128/Dockerfile
KV_CACHE_MEMORY=13500000000
MAX_SEQS=6
PMU=1

# legacy dflash2 profile (uses the base image directly, no profile-image build):
LANE=dflash2
IMAGE=ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2@sha256:4def0ef6…
DFLASH_REVISION=dc77ff1c99eeb2df044ee3d4f0094eb033fee410
KV_CACHE_MEMORY=12520000000
MAX_SEQS=8
```

Tuning rows that apply on any profile (each is one `.env` line):

| You want | Set | Effect |
|---|---|---|
| more KV pool, fewer slots | `MAX_SEQS=6 KV_CACHE_MEMORY=9663676416` | 9 GiB pin → 1.34M pool |
| faster boots (staging) | `MAX_LEN=262144 KV_CACHE_MEMORY=3221225472 MAX_SEQS=6` | 3 GiB pin |
| profiler-sized KV (safe fallback) | unset `KV_CACHE_MEMORY` | profiler sizes the pool |
| disable CUDA graphs | `EAGER=1` | `--enforce-eager` |

## Known issues & gotchas

- **KV pin trap.** `--kv-cache-memory` skips the profiler's activation check;
  too high = the first long prompt kills the engine; a fat boot also let
  `CUDA graph pool memory: -2.73 GiB` credit graph memory straight to KV and
  push the host into swap (−20–40% timing). The 13.5 GB default (dflash2pmu,
  with `MAX_SEQS=6`) and the 12.52 GB row are the validated values — don't
  raise either without re-measuring host headroom through a ~950K prefill.
- **DFlash2 needs exactly `num_speculative_tokens=7`** — any other count
  wedges the boot.
- **`--moe-backend marlin` mandatory.** `flashinfer_cutlass` does not boot
  W4A16 (`ValueError ... WNA16 MoE`); `triton` boots but loses ~2× everywhere.
  `humming`/`flashinfer_trtllm` are incompatible with this checkpoint/image.
- **block-size 2304 (legacy dflash2 profile)**: 4608 or 1152 break prefix
  hits; the engine's internal attention block auto-bumps to 4608 with fp8 KV
  — that `Setting attention block size to 4608` line is healthy. The pmu128
  profiles replace block granularity with `--prefix-match-unit 128`.
- **SM121 indexer overlay stays mounted in every profile**: the image has
  the indexer top-k fix baked, but the full-file overlay
  (`patches/sparse_attn_indexer_kpool.py`) is the runtime crash fix for
  decode past ~24K context (`EngineDeadError`). Keep it.
- **AutoGPTQ symmetric-qzeros watch item (NOT an issue on this pin).** Intel's
  sym GPTQ stores unused qzeros filled with the `0x77777777` sentinel. Newer
  vLLM (≥ 2026-09-05 upstream) needs
  `patch_vllm_autogptq_symmetric_moe_qzeros.py` (eugr/spark-vllm-docker
  `fix-autogptq-sym-qzeros` mod) for this; the pinned base image's
  `AutoGPTQMoEMethod` threads `may_have_zp=not is_sym` / `use_zp` (verified
  in the image's `auto_gptq.py` at `487ecf187`). **If the base image is
  bumped to a newer vLLM, adopt the eugr mod first.**
- **CUDA graphs vs eager (open A/B, both measured).** Graphs on is the
  shipped default; `--enforce-eager` was measured neutral/better by one
  source. Restart between A/B runs; read `vllm:prompt_tokens_total` and
  power draw, not GPU util%.
- **Vision+text concurrency (watch).** A pair of simultaneous image+text
  requests crashed a sibling profile fatally (`CUDA_ERROR_NOT_PERMITTED`,
  forum 381350). Test the combo before relying on it.
- **1M request caveat.** Single prompts ≳ ~310K tokens have hung the host in
  upstream tests; the measured fleet was 30–40K cached agent prompts. 1M
  `max_model_len` ≠ 1M single-request read today — validate with your own
  length ladder.
- **`GLM53_INDEXER_WORKSPACE=rightsize`** is another recipe's env knob; on
  this image it is a no-op unless the fork reads it (reported to give ~5 GB
  back at 1M ctx — kept, harmless otherwise).
- **Driver generation (repo-wide watch).** NVIDIA 610.43.02 costs ~4 GiB more
  unified memory and is boot-nondeterministic; this cluster is on
  580.173.02 — don't upgrade blindly.
- **GPU contention.** Serves on all 2 GPUs per node. Tear down any other
  model container on BOTH nodes before starting; one recipe at a time.

## References

Provenance and comparisons: [forum 382041](https://forums.developer.nvidia.com/t/intel-glm-5-3-flash-w4a16-autoround/382041)
(@miken, post 5) for the original serving profile; the measured A/B harness is
[rodman80/glm-5.3-flash-w4a16-2x-DGX-Sparks](https://github.com/rodman80/glm-5.3-flash-w4a16-2x-DGX-Sparks);
the serving profiles come from [florianbrede-ayet's recipes](https://forums.developer.nvidia.com/t/glm-5-3-flash-intel-autoquant-w4a16-tp2-mtp3-concurrent-agentic-use/382632).
Maintenance history and provenance notes live in `research.md`.

- Forum: [382041 — Intel GLM-5.3-Flash-W4A16-AutoRound](https://forums.developer.nvidia.com/t/intel-glm-5-3-flash-w4a16-autoround/382041)
  (@miken post 5 = the measured A/B + surgery + receipts; post 1 = model link)
- Weights: [Intel/GLM-5.3-Flash-W4A16-AutoRound](https://huggingface.co/Intel/GLM-5.3-Flash-W4A16-AutoRound)
  (AutoRound 0.15, `auto_round:auto_gptq`, sym g128, MIT)
- dflash2pmu profile (default): [florianbrede-ayet/spark-recipes/tp2_glm53flash_autoround_dflash2_k7_pmu128](https://github.com/florianbrede-ayet/spark-recipes/tree/main/tp2_glm53flash_autoround_dflash2_k7_pmu128)
  ([forum 382632](https://forums.developer.nvidia.com/t/glm-5-3-flash-intel-autoquant-w4a16-tp2-mtp3-concurrent-agentic-use/382632);
  vendored verbatim in `dflash2-pmu128/` — #53388/#53906/LCM + SWA fine-hits
  patches baked, DFlash2 drafter pin `bf582e4e…`, validate.sh gate included)
- mtp3 profile: [florianbrede-ayet/spark-recipes/tp2_glm53flash_autoround_mtp3_pmu128](https://github.com/florianbrede-ayet/spark-recipes/tree/main/tp2_glm53flash_autoround_mtp3_pmu128)
  (same forum 382632; vendored verbatim in `mtp3-pmu128/` —
  #53388/#53906/LCM patches + PMU128; upstream vLLM PRs:
  [vllm-project/vllm#53388](https://github.com/vllm-project/vllm/pull/53388),
  [#53906](https://github.com/vllm-project/vllm/pull/53906))
- Recipe/harness: [rodman80/glm-5.3-flash-w4a16-2x-DGX-Sparks](https://github.com/rodman80/glm-5.3-flash-w4a16-2x-DGX-Sparks)
  (A/B harness + `benchmarks/{RESULTS,COMPARISON}.md`; sibling quant
  [canada-quant/glm-5.3-w4a16-mtp](https://huggingface.co/canada-quant/glm-5.3-w4a16-mtp))
- Image: [tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark)
  (`docker/dflash2-overlay` build; `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2`)
- Drafter: [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
  (CC BY-NC-ND 4.0 — research/eval)
- Legacy APC patch source (dflash2 profile only): [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
  (`overlay/patch_hybrid_prefix_hit.py`; ported by rodman80, anchors verified
  against this image)
- Sym-qzeros fix (newer vLLM only): [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)
  `docker/patch_vllm_autogptq_symmetric_moe_qzeros.py`
- `research.md` in this directory — provenance, maintenance history, surgery
  mechanics, watchlist.
