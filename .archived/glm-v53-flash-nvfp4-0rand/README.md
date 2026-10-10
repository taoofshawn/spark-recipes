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
- **Default profile**: upstream's **launch-verified 1M** (1,048,576 tokens,
  2026-09-22) — **locally validated 2026-09-23** as the A/B winner: GMU 0.88,
  explicit 10.24 GB KV pin → 1,150,684-token pool (1.10× at full context;
  1,172,644 at MNBT 1024), seqs 4, batch 4096 (0rand's post-82 daily-driver
  batch; won the c2 A/B vs 1024), estimator off. The GMU 0.88 gate **passes on
  this cluster** with the 10.24 GB pin (host free RAM ~117 GiB).
- **Port** 8000 (repo convention); served name `glm-5.3-flash`; the model-name
  proxy serves clients `spark-llm` on :4000.
- **Modalities**: text, tool calling (`glm47` parser) + reasoning (`glm45`),
  thinking ON at effort `high` (explicit server-side pins). Vision is
  **unvalidated** in this recipe: `--skip-mm-profiling` is set and the W4A4
  checkpoint is memory-tight under concurrency (forum posts 17/39).

## Upstream receipts (author-reported; different hosts/protocols — do not mix)

| profile | ctx | KV pool | quality (TEB hardmode) | speed | source |
|---|---|---|---|---|---|
| **1M (shipped default)** | 1,048,576 | 1,150,684 (1.10×) @MNBT 4096 / 1,172,644 (1.12×) @1024 | not run here (upstream 900K row's 95/100 replication is the nearest quality receipt; hardmode was not re-run in the local A/B) | local A/B 2026-09-23 (temp-0 varied-prose, intra-matrix only): c1 tg1024 52.7/52.8; c2 2×tg512 66.1/77.2 agg (α 0.980) — **+14.5% c2 median vs the prior 700K default, c1 flat, ranges disjoint**; MNBT 4096 beat 1024 by +13.3% c2 at −21,960 pool tokens | upstream README 2026-09-22 (launch) + post 82 (daily driver) + this repo's 2026-09-23 A/B (research.md) |
| 900K (prior upstream default) | 900,096 | 1,080,115 (1.20×) | **95/100** (167/176) — *independent replication* | boot ~980 s; pp1024/tg1024 c1 32.0 t/s; 40.3/28.8/33.7 t/s @ depth 0/2K/8K | 0rand README (launch) + forum post 41 (quality) |
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
curl -s http://127.0.0.1:8000/v1/models        # "id":"glm-5.3-flash", max_model_len 1048576
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
#   -> ~1,150,684 tokens @1M ctx (10.24 GB pin, MNBT 4096; 1,172,644 at MNBT 1024)
docker logs glm53-nvfp4-0rand 2>&1 | grep -F "Graph capturing finished"
docker logs glm53-nvfp4-0rand 2>&1 | grep -F "Application startup complete"
```

The engine names the largest context that fits if the pool is too small:
`estimated maximum model length is <N>` — use that number instead of guessing.

## Tuning rows (each is one `.env` line)

| You want | Set | Effect / receipt |
|---|---|---|
| the prior 900K upstream demo profile | `MAX_LEN=900096 KV_CACHE_MEMORY=9663676416 MNBT=2048` | pool 1,027,894 measured locally (1.47×); upstream frames 900K/9 GiB as the conservative FIRST-BOOT demo; the 95/100 quality replication (post 41) was at this context family |
| the quality-verified profile | `MAX_LEN=700160` | 94/100 (166/176, HM 38/38), pool 892,139 (1.27×) upstream |
| more KV pool at 1M (upstream README batch) | `MNBT=1024` | pool 1,172,644 (1.12×, the upstream 1M-row number); local A/B 2026-09-23: c1/c2 tie with the prior default — the shipped MNBT 4096 is the winner instead |
| the experimental +18.4%-KV variant | see "Display-KV variant" below | pool 1,388,762 at 1M ctx (+216,118); concurrency 1.32× vs 1.12×; TEB hardmode 91/100, 0.0% error rate; requires sudo + runtime DRM reload — NOT wired into this compose recipe |
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

## Display-KV variant (upstream, EXPERIMENTAL — not wired here)

Upstream commit [`4ec03bf`](https://github.com/0rand/glm-5.3-flash-nvidia-nvfp4-dflash-2x-dgx-sparks/commit/4ec03bfc1e6a22b0ca59f73da3bd57c5d9a145ff)
(2026-09-22) added `start-display-kv.sh`: an alternative launcher that unlocks
the ~2 GiB firmware-reserved display memory on headless GB10 nodes and uses it
as KV-cache backing (technique and AGPL-3.0 allocator by coolbho3k,
[DeepSeek-v4.1-Flash-2x-DGX-Spark](https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark)).
Measured A/B on the GLM stack (2026-09-22): KV pool 1,388,762 vs 1,172,644
(+216,118, **+18.4%**), concurrency at 1M ctx 1.32× vs 1.12×, TEB hardmode
91/100 (160/176, parallel 4, seed 42) with 0.0% error rate, **no decode tax**
within fluctuation. `nvidia-smi` stays blind; GMU is unchanged.

Why it is documented but not vendored into this recipe:

- **Requires host-level changes our compose cannot express:** a runtime-only
  `nvidia_drm` reload (`modeset=1 fbdev=0`) on BOTH nodes, with **passwordless
  sudo on the worker** (`<user> ALL=(ALL) NOPASSWD: ALL`) for the remote reload;
  the head prompts once per boot. Nothing is persisted — a plain reboot
  restores boot defaults. This is a pre-bring-up host step, not a compose knob.
- **AGPL-3.0 allocator** — the shim/allocator source lives upstream under
  `display-kv/` (AGPL); vendoring it into this repo is a licensing decision
  deliberately not taken in this update pass.
- The launcher is coupled to upstream's `start.sh`/`run_cluster_dual.sh` lane,
  not our docker-compose conventions; wiring it here would be a compose opt-in
  mode with its own bring-up validation pass.

To adopt it here later: vendor `display-kv/` verbatim + a host DRM-reload
runbook step, wire `PYTHONPATH=/opt/display-kv` + `DISPLAY_KV_*` env via
compose mounts, keep `KV_CACHE_MEMORY=10240000000` (the preflight requires the
production 10.24 GB pin) and `MAX_LEN=1048576`, and bench before/after (the
pin+DEVICEMAP registration path itself carries a bandwidth cost: display
segment reads 163–168 GB/s vs 235–269 GB/s ordinary cudaMalloc; copy-engine
DMA access is catastrophic ~0.9–2.3 GB/s — attention kernels use SM loads,
which is why KV works fine). Driver-dependent: verified on 580.x; failure
reported upstream on 595.84 — re-probe after any driver change.

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
  don't upgrade blindly. **580.178.04 random hard freezes (forum #83, 2026-09-22):
  random hard freezes with no pattern, days into uptime** (Ama5u's units, running
  a week; diagnostics in amasu/glm53-flash-cluster) — stay put unless a boot test
  clears it; treat any driver bump as a re-measure event.
- **pilcothink 0.29 image exists (HELD).** `pilcothink/vllm_spark_glm53:0.29`
  (digest `sha256:82807abe…`, pushed 2026-09-17) — the 0.29 engine lane adds the
  `GLM53_CONTEXT_CUDA_GRAPH` / `GLM53_GROUPED_CONTEXT_STORE` /
  `GLM53_FUSED_DFLASH_TAPS` / `GLM53_CONTEXT_GRAPH_CACHE_V2` env family.
  ttsiodras's receipt (forum #92): GMU 0.9, batch 8192, seqs 10 @ 262K ctx —
  perf ~same as 0.28. Held: image bump requires a bring-up A/B at our shape
  (the new flags are unvalidated at seqs 4 / batch 2048 / 1M ctx); the pinned
  0.28 digest is unmoved and receipts unchanged.
- **DGX OS 7.5.0 OTA (repo-wide watch).** Boots with ~7.2 GiB less RAM (forum
  383222: 119.5 → 112.3 GiB kernel-available). The GMU 0.88 gate is a fatal
  whole-system-RAM check — do not take this OTA without lowering GMU and
  re-measuring on the cluster.
- **GPU contention.** Serves on all 2 GPUs per node. Tear down any other model
  container on BOTH nodes before starting; one recipe at a time.

## References

- Upstream: [0rand/glm-5.3-flash-nvidia-nvfp4-dflash-2x-dgx-sparks](https://github.com/0rand/glm-5.3-flash-nvidia-nvfp4-dflash-2x-dgx-sparks)
  (README 2026-09-22 state: launch-verified 1M production profile + display-KV
  variant; `start.sh` @ `0227b8df` = still the authoritative flag set, untouched
  by `4ec03bf`; `docs/FINDINGS.md` = the memory model, failure catalogue,
  measurement protocol; `display-kv/` = the experimental AGPL allocator + shim)
- Forum: [382939 — Let's optimize nvidia/GLM-5.3-Flash-NVFP4 for 2× DGX Spark](https://forums.developer.nvidia.com/t/lets-optimize-nvidia-glm-5-3-flash-nvfp4-for-2x-dgx-spark/382939)
  (post 1 = pilcothink's original results + 94/100; post 2 = 0rand's ~1.3M FP8
  cache + b12x bug hunt; post 17 = W4A4-vs-W4A16 fit + k=4; post 39 = the
  memory-zone model + spec-bench ladder + 2-parallel ceiling at 524K; post 41 =
  95/100 replication at 900K; post 44 = 1M-context claim; post 64 = pilcothink's
  W4A16-vs-W4A4 workload-fit comparison; post 80 = 0rand tested the
  local-inference-lab NVFP4 requant — "slower and constantly choked", stays on
  the nvidia quant; post 82 = 0rand's daily driver: 1M ctx, 11 GB pin, stable;
  posts 83/85/87 = 580.178.04 freezes + the display-KV unlock; post 92 =
  ttsiodras's 0.29-image receipt)
- Weights: [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4)
  @ `09b04e5e74bca08ca8549fc736d4cdd8624bfde3`
- Drafter: [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
  (CC BY-NC-ND 4.0 — research/eval, non-commercial)
- Runtime recipe: [gpdev-Pilcothink/DGX_Spark_vllm_Dockerfile](https://github.com/gpdev-Pilcothink/DGX_Spark_vllm_Dockerfile/tree/main/0.28/GLM53-flash)
  (the `0.28` GLM53-flash lane; supplies the b12x backends)
- `research.md` in this directory — provenance, receipts, watchlist.
