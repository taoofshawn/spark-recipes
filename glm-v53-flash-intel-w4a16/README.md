# glm-v53-flash-intel-w4a16 — GLM-5.3-Flash Intel W4A16 AutoRound (2× DGX Spark)

**Adoption** of the W4A16-AutoRound lane from
[forum 382041](https://forums.developer.nvidia.com/t/intel-glm-5-3-flash-w4a16-autoround/382041)
(@miken, post 5) into this repo's docker-compose conventions: serve
[`Intel/GLM-5.3-Flash-W4A16-AutoRound`](https://huggingface.co/Intel/GLM-5.3-Flash-W4A16-AutoRound)
(INT4 W4A16 + BF16 everywhere else) on the **Tony sm121-v11** image with
**DFlash2 k=7**, **fp8 KV**, **1M context**, TP=2. Thin wrapper — the recipe
that produced @miken's reported numbers is the thread post itself plus the
pieces it names (see References); the closest public repo for the harness and
A/B tables is
[rodman80/glm-5.3-flash-w4a16-2x-DGX-Sparks](https://github.com/rodman80/glm-5.3-flash-w4a16-2x-DGX-Sparks)
(same image, same patches, sibling `canada-quant` W4A16-MTP quant).

**Purpose:** this lane exists to answer "what is the Intel W4A16 quant capable
of vs the EXL3 ([`glm-v53-flash-miaai`](../glm-v53-flash-miaai) /
[`glm-v53-flash-entrpi`](../glm-v53-flash-entrpi)) and NVFP4
([`glm-v53-flash`](../glm-v53-flash)) variations". @miken's head-to-head and
rodman80's A/B harness are the reference protocol.

## What it serves

| | |
|---|---|
| Model | GLM-5.3-Flash, 320B total / 18B active, **INT4 W4A16 group-128 sym GPTQ** (Intel AutoRound quant; attention/router/shared-expert/vision/norms stay BF16) — post-surgery copy, **~82 GiB/rank** |
| Surgery | `prepare-model.sh`: auto-round config → stock GPTQ config with 679 `dynamic` skip rules (the model does **not** load as shipped) |
| Drafter | DFlash2 **k=7** (`incoai/GLM-5.3-Flash-DFlash2`, rev `dc77ff1c…`; CC **BY-NC-ND-4.0**) |
| Served name | `glm-5.3-flash` | Port: **8000** (repo convention) |
| Context | **1,048,576** default; KV pool **1,752,785 tokens** @ fp8 KV, pin 12.52 GB, GMU 0.85 (miken's receipt: `Maximum concurrency ... 1.67x`) |
| Speed (miken, 2× Spark TP2) | cold prefill **1,405 tok/s**; decode prose 23–25 / code **52** / structured **39** tok/s; 8-way wave **26–75** tok/s/stream; DFlash2 code acceptance **0.68–0.71** |
| Speed (rodman80, same image) | C1 31.8 tok/s (TTFT 0.35 s), C4 69.2 / C6 **81.1** tok/s aggregate, acceptance 0.418, boot ~8 min, prefill 1.3–1.6k tok/s flat to 300K |
| Quality (miken / Intel card) | tool-eval hardmode **90/100 max**; needle-exact @419K/836K/947K, 0 U+FFFD in CJK, vision passes; GSM8K 0.9712 / MMLU 0.8620 (99.84% of BF16) |
| Modalities | text, images/video, tool calling (`glm47` parser), thinking ON at effort `high` (explicit server-side pins) |
| Sampling (explicit) | **temp 1.0 / top_p 0.95 / thinking true / reasoning_effort high** — `--generation-config vllm` + `--override-generation-config` + `--default-chat-template-kwargs`, so checkpoint/template defaults can't silently drift them |

## Why this quant (miken's verdicts, post 5)

> Intel W4A16-AutoRound vs Local Inference Lab NVFP4 vs MiaAI EXL3, 2× Spark TP2, same protocol

| | Intel W4A16 | LIL NVFP4 (`378ca545`) | MiaAI EXL3 kit (`c190db1`) |
|---|---|---|---|
| cold prefill | 1,405 tok/s | 1,460–1,500 | 913 |
| decode prose / code / structured c=1 | 23–25 / **52** / **39** | 26–30 / 31 / 31 | 24 / 37 / 28 |
| per-stream decode, 8-way wave | **26–75 tok/s** | 23–43 | 27–73 |
| APC: cached 32K turn | **0.7 s (99% hits)** | 3.7–4.5 s (85%) | 3.6–4.5 s |
| c8 wave wall (8 real requests) | **33–40 s** | 54–62 s | not completed |
| speculative | DFlash2 k=7, 0.68–0.71 on code | MTP-2, 0.61–0.63 | DFlash2 k=7 |
| KV pool @1M ctx | **1.75M** (pinned) | 0.74–0.9M | 0.74M @ 8 slots |
| tool-eval hardmode s42 | 89 low / **90 max** | 89 | 92 |
| weights/rank · boot | 82 GiB · 10 min | 92 GiB · 19 min | 82 GiB · 10 min |

EXL3 keeps the quality crown (92, KLD 0.0246) — miken traded it for prefill,
pool and slots. rodman80's independent A/B adds the concurrency story:
single-stream is a **tie (~30 tok/s, acceptance ~0.42)** across NVFP4/EXL3/W4A16;
W4A16 keeps scaling where the others plateau (C4 69.2 / C6 81.1 vs ~42/51 and
~44/48) — the W4A16 advantage is **concurrency and KV pool**, not single-stream.

## The surgery (required — the model does not load as shipped)

`auto-round` is not in the GB10 forks' `QUANTIZATION_METHODS`, but the tensors
are plain GPTQ (`auto_round:auto_gptq`, sym, group-128). `prepare-model.sh`
materializes a serving dir from the HF snapshot (hardlinks, zero extra space)
and swaps `quantization_config` in `config.json`:

```json
{"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": true,
 "desc_act": false, "lm_head": false, "true_sequential": true,
 "dynamic": {"-:<regex>": {} /* = every extra_config BF16 exclusion, 679 rules */}}
```

`-:` rules tell vLLM's AutoGPTQConfig to leave matched modules unquantized
(BF16 load), which is exactly what auto-round's `extra_config` meant (verified
against `vllm/model_executor/layers/quantization/utils/gptq_utils.py`
`get_dynamic_override`). Run on BOTH nodes; idempotent; fail-closed.

## Profile rows

All are `.env` edits (one line each); defaults = miken's row:

| You want | Set | You get |
|---|---|---|
| **miken's validated row (default)** | (nothing) | 1M ctx, 8 seqs, 1.75M-token pool @ 12.52 GB pin, graphs on, DFlash2 k=7 |
| **rodman80's validated row** | `MAX_SEQS=6 KV_CACHE_MEMORY=9663676416 EAGER=1` | same ctx; C6 81.1 measured; 9 GiB pin → 1.34M pool |
| **262K staging (faster boots)** | `MAX_LEN=262144 KV_CACHE_MEMORY=3221225472 MAX_SEQS=6` | 3 GiB pin |
| **Profiler-sized KV (safe fallback)** | `KV_CACHE_MEMORY=` | pool sized by the profiler (no bypass of the activation check) |

## Deploy

Pre-launch ritual (per node — GB10 unified memory swap-wedges instead of OOMing):
```bash
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
```
Optional but measured (+5–7% decode): `./tools/clocks.sh` on each node.

```bash
# 0) one-time provisioning (BOTH nodes). No hf CLI on the Sparks: use a venv.
python3 -m venv /tmp/hfvenv && /tmp/hfvenv/bin/pip install -q -U huggingface_hub

# 0a) weights — lands in the default HF hub cache (snapshot rev pin):
/tmp/hfvenv/bin/hf download Intel/GLM-5.3-Flash-W4A16-AutoRound \
    --revision 5eee1846f0321058ed73745f9aa16f2aaf0fc0a0
# 0b) drafter (2.3 GiB, same cache):
/tmp/hfvenv/bin/hf download incoai/GLM-5.3-Flash-DFlash2 \
    --revision dc77ff1c99eeb2df044ee3d4f0094eb033fee410
# 0c) the GPTQ surgery (builds $MODEL_HOST_PATH from the snapshot):
./prepare-model.sh            # run on BOTH nodes

# 1) pull the image (both nodes; ~31 GiB):
docker pull ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6

# 2) worker (rank 1) FIRST, leader ~35 s later:
docker compose --env-file .env --env-file .env.node1 up -d
docker compose --env-file .env --env-file .env.node0 up -d

# 3) verify (leader):
curl -s http://127.0.0.1:8000/v1/models        # "id":"glm-5.3-flash", max_model_len 1048576
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health   # 200
```

Cold boot ~8–10 min (weight load + engine init; JIT caches persist, `init
engine` 87 s on a warm JIT cache). Never benchmark right after boot.

## Boot markers (leader log)

```bash
docker logs glm53-intel-w4a16 2>&1 | grep -F "GID auto-detect"
#   -> "GID auto-detect: NCCL_IB_GID_INDEX=N" (not the error dump)
docker logs glm53-intel-w4a16 2>&1 | grep -F "hybrid APC groups"
#   -> "hybrid APC groups: [...]; eagle_group_ids=[...]" = APC patch engaged
docker logs glm53-intel-w4a16 2>&1 | grep -F "GPU KV cache size"
#   -> 1,752,785 tokens @ 1M ctx = miken's receipt; ~1.34M = 9 GiB row
docker logs glm53-intel-w4a16 2>&1 | grep -F "Setting attention block size"
#   -> 4608 (fp8 KV auto-bump from block-size 2304) — expected, NOT 2304
docker logs glm53-intel-w4a16 2>&1 | grep -F "Model loading took"
#   -> ~84 GiB, ~5 min
```

## Known issues & gotchas

- **APC patch is required for prefix-cache hits.** `dflash` is `use_eagle()`,
  so stock vLLM flags *every* group EAGLE and the drafter's SWA group zeroes
  the hybrid min → **0 hits** (miken: 27,648/32,613 → 32,256/32,613 after).
  `patches/patch_hybrid_prefix_hit.py` fixes both (fail-closed; aborts boot if
  the image's coordinator anchors drift).
- **KV pin lottery + trap.** Pinning skips the profiler's activation check; too
  high = first long prompt kills the engine; the fat boot also let
  `CUDA graph pool memory: -2.73 GiB` credit graph memory straight to KV and
  pushed the host into swap (−20–40% timing). miken's 12.52 GB is validated
  to keep ≥4.3 GB host headroom through a 950K prefill — don't raise it without
  re-measuring floors.
- **DFlash2 needs exactly `num_speculative_tokens=7`** — any other count wedges
  the boot.
- **`--moe-backend marlin` mandatory.** `flashinfer_cutlass` does not boot
  W4A16 (`ValueError ... WNA16 MoE`); `triton` boots but loses ~2× everywhere
  (rodman80 A/B). `humming`/`flashinfer_trtllm` are incompatible with this
  checkpoint/image.
- **block-size 2304**: 4608 or 1152 break prefix hits (rodman80 A/B); the
  engine's internal attention block auto-bumps to 4608 with fp8 KV — that
  `Setting attention block size to 4608` line is healthy.
- **APC + DFlash2 drafter + indexer**: the image has the indexer top-k fix
  baked (patch_v7), but we still mount the SM121 full-file indexer overlay
  (`patches/sparse_attn_indexer_kpool.py`) — the runtime crash fix for
  decode past ~24K context (`EngineDeadError`). Keep both.
- **AutoGPTQ symmetric-qzeros watch item (NOT an issue on this pin).** Intel's
  sym GPTQ stores unused qzeros filled with the `0x77777777` sentinel. Newer
  vLLM (≥ 2026-09-05 upstream) needed `patch_vllm_autogptq_symmetric_moe_qzeros.py`
  (eugr/spark-vllm-docker `fix-autogptq-sym-qzeros` mod) for this; the pinned
  sm121-v11 image's `AutoGPTQMoEMethod` threads
  `may_have_zp=not is_sym` / `use_zp` (verified in the image's
  `auto_gptq.py` at `487ecf187`) and miken's quality receipts confirm. **If
  the image is bumped to a newer vLLM, adopt the eugr mod first.**
- **CUDA graphs vs eager (open A/B, both measured).** miken's receipts:
  graphs on (`graphs 1..64`), KV pin 12.52 GB. rodman80: `--enforce-eager`
  neutral/better and their validated row. Default here = miken (graphs on);
  `EAGER=1` = rodman80's row. Restart between A/B runs; read
  `vllm:prompt_tokens_total` and power draw, not GPU util% (benchmark
  discipline from the repo's other lanes).
- **Vision+text concurrency (watch).** A pair of simultaneous image+text
  requests crashed the EXL3 lane fatally (`CUDA_ERROR_NOT_PERMITTED`, forum
  381350/273) on a config dump similar to ours. Test the combo before relying
  on it.
- **1M request caveat.** Single prompts ≳ ~310K tokens hung the host in
  rodman80's tests; miken's fleet was 30–40K cached agent prompts. 1M
  `max_model_len` ≠ 1M single-request read now — validate with your own length
  ladder.
- **`GLM53_INDEXER_WORKSPACE=rightsize`** is MiaAI's code; on this image it is
  a no-op unless the fork reads it (miken reports 5 GB back at 1M ctx — keep
  it, harmless otherwise).
- **Driver generation (repo-wide watch).** NVIDIA 610.43.02 costs ~4 GiB more
  unified memory and boot-nondeterministic; this cluster is on 580.173.02 —
  don't upgrade blindly.

## Deviations from the sources

| | miken post / rodman80 repo | here | why |
|---|---|---|---|
| port | 8000 | **8000** | repo convention; model-name proxy serves :4000 |
| mechanism | `start.sh`/`launch-*.sh` orchestrators | docker-compose (`.env`/`.env.node0/1`) | repo convention |
| GID | hardcoded `3` (rodman80) | sysfs auto-detect in the command block | GID renumbers across reboots |
| NICs | one rail per box (rodman80) | both (`IB_PORTS`) | this cluster's validated set |
| image | `radixark/…` (alias) digest-non-pinned | `ghcr.io/tonyd2wild/…` digest-pinned `4def0ef6…` | alias is identical; pinning is required |
| model | `canada-quant/glm-5.3-w4a16-mtp` (rodman80) | **`Intel/GLM-5.3-Flash-W4A16-AutoRound` + surgery** | user goal: the Intel quant from the thread |
| KV pin | 9 GiB (rodman80) / 12.52 GB (miken) | **12.52 GB** (miken's receipt) | miken = the thread's target numbers; 9 GiB row documented |
| seqs / graphs | 6 + eager (rodman80) / 8 + graphs (miken) | **8 + graphs** | same |
| template | `--chat-template` vendored mm file (rodman80) | **same** — vendored `patches/chat_template_mm.jinja` (rodman80's, validated on this image; honors `enable_thinking` + `reasoning_effort`) | the Intel repo's shipped template ignores `enable_thinking` — vendoring makes the THINKING toggle explicit (parity with entrpi) |

## When to use this vs the other glm recipes

Same weights-class, same served name on :8000 (one recipe at a time):

- **intel-w4a16 (this)**: the W4A16 trade — highest KV pool/concurrency per
  GiB, ~82 GiB/rank, best prefill of the 4bpw lanes at 1M ctx; quality ~0.99×
  of BF16 (card) with EXL3 still ahead on hard tool evals (92 vs 90).
- **glm-v53-flash-miaai / -entrpi (EXL3)**: quality crown (KLD 0.0246,
  tool-eval 92), different drafter geometry; entrpi is the lowest-maintenance
  EXL3 lane, miaai the 1M-native hotfix lane.
- **glm-v53-flash (NVFP4)**: W4A4 (weights+activations), 262K ceiling on this
  cluster, MTP3 default; the reference vLLM patch stack.

## References

- Forum: [382041 — Intel GLM-5.3-Flash-W4A16-AutoRound](https://forums.developer.nvidia.com/t/intel-glm-5-3-flash-w4a16-autoround/382041)
  (@miken post 5 = the measured A/B + surgery + receipts; post 1 = model link)
- Weights: [Intel/GLM-5.3-Flash-W4A16-AutoRound](https://huggingface.co/Intel/GLM-5.3-Flash-W4A16-AutoRound)
  (AutoRound 0.15, `auto_round:auto_gptq`, sym g128, MIT)
- Recipe/harness: [rodman80/glm-5.3-flash-w4a16-2x-DGX-Sparks](https://github.com/rodman80/glm-5.3-flash-w4a16-2x-DGX-Sparks)
  (A/B harness + `benchmarks/{RESULTS,COMPARISON}.md`; sibling quant
  [canada-quant/glm-5.3-w4a16-mtp](https://huggingface.co/canada-quant/glm-5.3-w4a16-mtp))
- Image: [tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark)
  (`docker/dflash2-overlay` build; `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2`)
- Drafter: [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
  (CC BY-NC-ND 4.0 — research/eval)
- APC patch source: [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
  (`overlay/patch_hybrid_prefix_hit.py`; ported by rodman80, anchors verified
  against this image)
- Sym-qzeros fix (newer vLLM only): [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)
  `docker/patch_vllm_autogptq_symmetric_moe_qzeros.py`
- `research.md` in this directory — provenance, surgery mechanics, watchlist.

## GPU contention

Serves on all 2 GPUs per node. Tear down any other model container (DS4 vision,
EXL3 GLM, etc.) on BOTH nodes before starting. One recipe at a time.
