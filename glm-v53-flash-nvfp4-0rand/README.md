# glm-v53-flash-nvfp4-0rand — GLM-5.3-Flash NVIDIA NVFP4 + DFlash2 (2× DGX Spark)

Serve [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4)
(GLM-5.3-Flash quantized by NVIDIA to uniform NVFP4) on a 2-node DGX Spark cluster
with DFlash2 speculative decoding, at 900K-token context. Ported from
[0rand/glm-5.3-flash-nvidia-nvfp4-dflash-2x-dgx-sparks](https://github.com/0rand/glm-5.3-flash-nvidia-nvfp4-dflash-2x-dgx-sparks)
+ [forum 382939](https://forums.developer.nvidia.com/t/lets-optimize-nvidia-glm-5-3-flash-nvfp4-for-2x-dgx-spark/382939)
into this repo's docker-compose house style.

- **Model**: GLM-5.3-Flash (320B total / 18B active), **NVIDIA official NVFP4 =
  W4A4** (weights AND activations 4-bit, static scales). ~190 GiB on disk
  (~90.6 GiB/rank measured, forum post 39). **Ships no MTP heads** (0 `mtp`/
  `nextn` tensors in 147,661 per the upstream README) → the external DFlash2
  drafter is mandatory; this checkpoint cannot run native MTP.
- **Stack**: `pilcothink/vllm_spark_glm53:0.28` (vLLM 0.28.1rc1.dev475+g6fbb00b18)
  — supplies the **b12x MoE/linear backends** and the sparse-MLA attention
  default. DFlash2 k=5, CUDA graphs ≤16, async scheduling, `--kv-cache-dtype
  fp8`, block size 256, `--mamba-cache-mode align`.
- **Default profile**: 0rand's **launch-verified 900K** (900,096 = 3,516×256):
  GMU 0.88, explicit 9 GiB KV pin → ~1.08M-token pool (1.20× at full context),
  seqs 4, batch 1024, estimator off.
- **Port** 8000 (repo convention); served name `glm-5.3-flash`; the model-name
  proxy serves clients `spark-llm` on :4000.
- **Modalities**: text, tool calling (`glm47` parser) + reasoning (`glm45`),
  thinking ON at effort `high` (explicit server-side pins). Vision is
  **unvalidated** in this recipe: `--skip-mm-profiling` is set and the W4A4
  checkpoint is memory-tight under concurrency (forum posts 17/39).

## Upstream receipts (author-reported; different hosts/protocols — do not mix)

| profile | ctx | KV pool | quality (TEB hardmode) | speed | source |
|---|---|---|---|---|---|
| **900K (shipped default)** | 900,096 | 1,080,115 (1.20×) | **95/100** (167/176) — *independent replication* | boot ~980 s; pp1024/tg1024 c1 32.0 t/s; 40.3/28.8/33.7 t/s @ depth 0/2K/8K | 0rand README (launch) + forum post 41 (quality) |
| 700K (prior, quality-verified) | 700,160 | 892,139 (1.27×) | **94/100** (166/176), Hard Mode 38/38, e2e 996 s | spec-bench: structured 49.7 / code 43.0 / filler 33.5 eff t/s | 0rand README + docs/FINDINGS.md |
| 524K (jetspark's concurrency row) | 524,288 | 668,803 (6.5 GiB pin) | 90 (nvidia W4A4, k=7) | decode 19.6 t/s; **only 2 parallel clients fit** at batch 8192 | forum posts 17 / 39 |

Note: 0rand explicitly did **not** run hardmode on his 900K profile; the 95/100
at 900K is paxren2020's independent replication (post 41, same engine build).

## Deploy

### 0) One-time provisioning (BOTH nodes)

```bash
python3 -m venv /tmp/hfvenv && /tmp/hfvenv/bin/pip install -q -U huggingface_hub

# a) weights (~190 GiB — lands in the default HF hub cache, revision-pinned):
/tmp/hfvenv/bin/hf download nvidia/GLM-5.3-Flash-NVFP4 \
    --revision 09b04e5e74bca08ca8549fc736d4cdd8624bfde3

# b) DFlash2 drafter (2.3 GiB, same cache; CC BY-NC-ND-4.0 — non-commercial):
/tmp/hfvenv/bin/hf download incoai/GLM-5.3-Flash-DFlash2 \
    --revision bf582e4eacc1810f76656d1811693ff6c6737d2a
```

**No surgery step** — unlike the Intel W4A16 recipe, the NVFP4 checkpoint loads
natively on the pilcothink image. The compose serves both checkpoints straight
from their HF snapshot dirs (resolved at boot, fail-closed).

### 1) Pull the image (BOTH nodes)

```bash
docker pull pilcothink/vllm_spark_glm53:0.28
```

### 2) Launch — worker (rank 1) FIRST, leader ~35 s later

Pre-launch ritual per node (GB10 unified memory swap-wedges instead of OOMing;
heavy weight-load IO grows the page cache and starves the NVRM allocator — and
on this stack the GMU gate reads free RAM):

```bash
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
```

```bash
docker compose --env-file .env --env-file .env.node1 up -d   # worker
docker compose --env-file .env --env-file .env.node0 up -d   # leader
```

The wrong start order hangs the rendezvous (`DistStoreError: 1/2 clients`,
`Connection reset by peer`); always `docker compose down` on BOTH nodes between
relaunches. Cold boot ~16–17 min (upstream measured ~980 s to serving; weight
load dominates). Never benchmark right after boot.

### 3) Verify (leader)

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health   # 200 — /v1/models returns 200 even with a dead engine
curl -s http://127.0.0.1:8000/v1/models        # "id":"glm-5.3-flash", max_model_len 900096
```

End-to-end through the model-name proxy (clients use `spark-llm`):

```bash
curl -s http://spark.shawndo.intra:4000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"spark-llm","messages":[{"role":"user","content":"Say hi in one word"}],"max_tokens":32}' | jq -r '.choices[0].message.content'
```

## Boot markers (leader log)

```bash
docker logs glm53-nvfp4-0rand 2>&1 | grep -F "GID auto-detect"
#   -> "GID auto-detect: NCCL_IB_GID_INDEX=N" (not the error dump)
docker logs glm53-nvfp4-0rand 2>&1 | grep -F "Available KV cache memory"
docker logs glm53-nvfp4-0rand 2>&1 | grep -F "CUDA graph memory profiling"
#   must show the estimator DISABLED (VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0).
#   If it says "equivalent to --gpu-memory-utilization=0.86xx", the env did not
#   take and you are paying ~2.1 GiB of KV for nothing.
docker logs glm53-nvfp4-0rand 2>&1 | grep -F "GPU KV cache size"
#   -> ~1,080,115 tokens @900K ctx (9 GiB pin)
docker logs glm53-nvfp4-0rand 2>&1 | grep -F "Graph capturing finished"
docker logs glm53-nvfp4-0rand 2>&1 | grep -F "Application startup complete"
```

The engine names the largest context that fits if the pool is too small:
`estimated maximum model length is <N>` — use that number instead of guessing.

## Tuning rows (each is one `.env` line)

| You want | Set | Effect / receipt |
|---|---|---|
| the quality-verified profile | `MAX_LEN=700160` | 94/100 (166/176, HM 38/38), pool 892,139 (1.27×) upstream |
| more concurrency | `MAX_LEN=524288 MAX_SEQS=2 MNBT=8192 KV_CACHE_MEMORY=6979321856` | jetspark's 524K row (post 39). **Activations ≈ MNBT × ~800 KiB/token** (8192 → ~6.4 GiB) — raise seqs only with a smaller batch |
| faster decode (k=4) | `DFLASH_TOKENS=4` | ~20% faster decode than k=7, same tokens/step accepted (posts 17/39) |
| official k=7 | `DFLASH_TOKENS=7` | pilcothink: TG128 39.1 t/s @262K (post 1) |
| eager (no graphs) | `EAGER=1` | frees ~4 GiB of pool at capture ≤64; 0.46 GiB measured at jetspark's config (post 39) |
| staging boot | `MAX_LEN=262144 KV_CACHE_MEMORY=3221225472` | 3 GiB pin |

**GMU is NOT a portable number.** On the pilcothink build it is a **fatal**
pre-load check against *whole-system* RAM (`docs/FINDINGS.md` §2): 0.88 was
0rand's measured ceiling on HIS host. If boot dies ~45 s in with
`ValueError: Free memory on device cuda:0 (X/121.69 GiB) on startup is less than
desired GPU memory utilization`, lower GMU (0.85 — the intel recipe's value — is
a safe start) and re-measure. `--kv-cache-memory` does **not** bypass this gate.

## Known issues & gotchas

- **CUDA-graph memory estimator tax (vLLM ≥ v0.21).** In graph mode the engine
  reserves ~2.1 GiB of KV for the graph estimate; `ESTIMATE_CUDAGRAPHS=0`
  (shipped default) returns it. Re-enabling it shrinks the pool accordingly.
- **Batch/seqs are KV-budget levers, not just perf knobs** (FINDINGS §4). The
  whole 700K breakthrough was +4.11 GiB of pool: estimator off (+2.11) +
  batch/seqs trim (+2.00).
- **DFlash2 scales poorly with concurrency.** "DFlash scales really poorly with
  concurrency, MTP beats it" (jetspark, post 39) — and this checkpoint has no
  MTP head, so DFlash2 is the only option here. For 4 concurrent sessions keep
  MNBT small (1024) and expect sublinear aggregate scaling.
- **The W4A4 checkpoint is memory-tight.** It cannot run its own MTP head,
  needed four starts on jetspark's node at 512K, and loses video input +
  concurrency before the W4A16 quants do (post 17). Two Sparks are "not really
  enough" for this checkpoint at high batch (post 39).
- **No PMU / fine-grained prefix matching** in this stack — plain
  `--enable-prefix-caching` only. The intel recipe's PMU128 lane is a different
  image lineage entirely.
- **1M context**: pilcothink reports serving 1M on this stack (post 44) and
  quotes ~118 GiB free RAM per node as the prerequisite (post 17). The shipped
  profile stops at 900K (0rand's launch-verified max);
  `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` is already set, so raising `MAX_LEN` is a
  memory test, not a flag test.
- **Driver generation (repo-wide watch).** NVIDIA 610.43.02 costs ~4 GiB more
  unified memory and is boot-nondeterministic; this cluster is on 580.173.02 —
  don't upgrade blindly.
- **GPU contention.** Serves on all 2 GPUs per node. Tear down any other model
  container on BOTH nodes before starting; one recipe at a time.

## References

- Upstream: [0rand/glm-5.3-flash-nvidia-nvfp4-dflash-2x-dgx-sparks](https://github.com/0rand/glm-5.3-flash-nvidia-nvfp4-dflash-2x-dgx-sparks)
  (README 2026-09-13 state; `start.sh` @ `0227b8df` = the authoritative flag
  set; `docs/FINDINGS.md` = the memory model, failure catalogue, measurement
  protocol)
- Forum: [382939 — Let's optimize nvidia/GLM-5.3-Flash-NVFP4 for 2× DGX Spark](https://forums.developer.nvidia.com/t/lets-optimize-nvidia-glm-5-3-flash-nvfp4-for-2x-dgx-spark/382939)
  (post 1 = pilcothink's original results + 94/100; post 2 = 0rand's ~1.3M FP8
  cache + b12x bug hunt; post 17 = W4A4-vs-W4A16 fit + k=4; post 39 = the
  memory-zone model + spec-bench ladder + 2-parallel ceiling at 524K; post 41 =
  95/100 replication at 900K; post 44 = 1M-context claim)
- Weights: [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4)
  @ `09b04e5e74bca08ca8549fc736d4cdd8624bfde3`
- Drafter: [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
  (CC BY-NC-ND 4.0 — research/eval, non-commercial)
- Runtime recipe: [gpdev-Pilcothink/DGX_Spark_vllm_Dockerfile](https://github.com/gpdev-Pilcothink/DGX_Spark_vllm_Dockerfile/tree/main/0.28/GLM53-flash)
  (the `0.28` GLM53-flash lane; supplies the b12x backends)
- `research.md` in this directory — provenance, receipts, watchlist.
