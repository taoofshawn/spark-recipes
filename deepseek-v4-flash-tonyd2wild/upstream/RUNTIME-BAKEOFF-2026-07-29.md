# Runtime bake-off: vLLM 0.21.1 (B12X) vs vLLM 0.25.2 (Anemll) — 2026-07-29

**Result: the older, custom-kernel runtime this repo ships is still the fastest thing we
can measure on GB10.** The newer runtime is 9% slower on peak decode, 8% slower on mean
decode, and 29% slower at 6-way concurrency. It is, however, a far better *diagnostic*
tool, and it produced the single most useful measurement of the night — see
[Why decode bursts then drops](#why-decode-bursts-then-drops-measured-not-theorised).

## Method

Same two nodes, same weights, same serve args, back to back, on an experiment lane
isolated from the production endpoint. No number here is borrowed from anyone else's run.

- Nodes: head `192.168.192.2` + worker `192.168.192.4`, TP=2, `mp` executor, RoCE fabric
- Weights: `fraserprice/DeepSeek-V4-Flash-DSpark`
- Identical args on both:

```
--kv-cache-dtype nvfp4_ds_mla --block-size 256
--max-model-len 1048576 --max-num-seqs 6 --max-num-batched-tokens 8192
--gpu-memory-utilization 0.78 --enable-prefix-caching
--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}'
```

Decode is measured temp 0, warm, best-of-2 per prompt, using the server's own
`completion_tokens` over wall time. Five content types, because **acceptance is
content-driven and a single prompt is not a benchmark** — the 30→84 tok/s spread below is
one server on one config.

## Runtime A — vLLM `0.21.1rc1.dev339` + B12X kernels (what this repo ships)

```
DECODE (temp 0, warm)            CONCURRENCY (400 tok each)        PREFILL
prompt      tok    sec   tok/s    conc  agg tok/s  per-stream      depth   tok/s
count300    600   7.12    84.3       1       61.0        61.0      8K       1513
mult12      900  11.56    77.9       2       91.7        46.9      32K      2284
json60      800  10.39    77.0       4      151.1        38.7      100K     2639
bst         600   9.34    64.2       6      197.3        33.6
story       251   7.25    34.6
PEAK 84.3   MEAN 67.6
```

KV pool at gmu 0.78: **1,548,597 tokens**.

## Runtime B — vLLM `0.25.2.dev0+g752a3a504` (`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`)

torch 2.11.0+cu130, DeepGEMM MXFP4 MoE backend, FP8 Lightning-Indexer cache,
`CompilationMode.NONE` (torch.compile **off**, breakable CUDA graphs instead).

```
DECODE (temp 0, warm)            CONCURRENCY (400 tok each)        PREFILL
prompt      tok    sec   tok/s    conc  agg tok/s  per-stream      depth   tok/s
count300    600   7.77    77.2       1       61.1        61.1      8K       1446
mult12      900  12.63    71.3       2      101.6        50.8
json60      800  11.15    71.7       4      116.4        29.1
bst         600   9.49    63.2       6      152.5        25.7      32K      2488
story       254   8.51    29.8                                     100K     2704
PEAK 77.2   MEAN 62.7
```

KV pool at gmu 0.78: **1,385,765 tokens** on this boot — but see
[KV pool is not a runtime difference](#kv-pool-is-not-a-runtime-difference-retracted); a later
boot of the *same* runtime and config reported 1,533,940, so this is not a stable figure.

## Head to head

| axis | 0.21.1 + B12X | 0.25.2 Anemll | winner |
| --- | ---: | ---: | :--- |
| decode peak | **84.3** | 77.2 | **B12X +9.2%** |
| decode mean (5 types) | **67.6** | 62.7 | **B12X +7.8%** |
| concurrency c1 | 61.0 | 61.1 | tie |
| concurrency c2 | 91.7 | **101.6** | Anemll +10.8% |
| concurrency c4 | **151.1** | 116.4 | **B12X +29.8%** |
| concurrency c6 | **197.3** | 152.5 | **B12X +29.4%** |
| prefill 8K | **1513** | 1446 | B12X +4.6% |
| prefill 32K | 2284 | **2488** | Anemll +8.9% |
| prefill 100K | 2639 | **2704** | Anemll +2.5% |
| KV pool @ gmu 0.78 | 1,548,597 | 1,385,765 | **no conclusion — see retraction below** |

Anemll wins three things, and they are real: **c2 concurrency** (+10.8%, where 0.25's
asynchronous scheduling helps) and **prefill at depth** (+8.9% at 32K, +2.5% at 100K). The
c2 win does not hold — past c2 it scales badly and finishes 29% down at c6, which is the
range an agent fleet actually runs in. The prefill win is genuine but small, and it is the
axis that matters least here: at 2.5-3k tok/s prefill, a 32K prompt costs ~10s on either
runtime, while the decode deficit is paid on every single token of every response.

### Why acceptance is not the explanation

Anemll logs **100% draft acceptance** on `count300` and still loses to B12X on that prompt.
With `k=5`, 100% acceptance means 6 tokens per step, so:

```
Anemll:   77.2 tok/s ÷ 6 = 12.9 steps/s
B12X:     84.3 tok/s ÷ 6 = 14.0 steps/s
```

Both runtimes saturate the draft. The entire gap is **step time** — B12X executes a decode
step ~9% faster.

The two runtimes differ on **both** of the things that set step time, and it is worth being
precise about which:

| | this repo (0.21.1 + B12X) | Anemll (0.25.2) |
| --- | --- | --- |
| MoE backend | `Using 'B12X' Mxfp4 MoE backend` | `Using 'DEEPGEMM_MXFP4' Mxfp4 MoE backend` |
| torch.compile | **works** — `Directly load AOT compilation…`, `torch.compile took 4.39 s` | **unsupported for this model** |

On 0.25.2, asking for compile yields:

```
`torch.compile` is turned on, but the model DeepSeek-V4-Flash-DSpark does not support it.
```

So this is **not** a case of the Anemll image merely choosing `CompilationMode.NONE` as a
default you could flip back — compile is unavailable for this model on 0.25.2 in either
direction. Meanwhile this repo's stack **does** get a compiled model, from the AOT cache, in
about 4 seconds.

The ~9% step-time gap is therefore the combination of **B12X custom kernels**
(`VLLM_USE_B12X_MOE`, `VLLM_USE_B12X_WO_PROJECTION`, `VLLM_TRITON_MLA_SPARSE`, tuned W4A16
block/tile configs) **plus a working compile path**. Both live on the old vLLM. That is the
whole reason this recipe is worth keeping there rather than chasing upstream versions.

## KV pool is not a runtime difference (retracted)

An earlier version of this document claimed B12X gave a ~11.7% larger KV pool at the same
`gpu_memory_utilization`. **That claim was wrong and is retracted.**

Re-booting identical configurations later in the session reported wildly different pools — on
**both** runtimes:

```
0.25.2  boot 1   GPU KV cache size: 1,385,765 tokens
0.25.2  boot 3   GPU KV cache size: 1,533,940 tokens   (+10.7%, Available KV memory 10.58 GiB)

0.21.1  earlier  GPU KV cache size: 1,548,597 tokens
0.21.1  later    GPU KV cache size: 1,336,656 tokens   (-13.7%, same config, same nodes)
```

Our own runtime swung **16%** between two boots of a byte-identical configuration — larger than
the 11.7% "advantage" I originally attributed to it. Available KV memory on GB10 varies with
whatever else has touched unified memory, so **a single boot's reported pool size is not a
runtime property** and must not be compared across runtimes without repeated boots.

Also worth noting from the later boot log, since it affects anyone reading pool numbers:

```
CUDA graph memory profiling is enabled (default since v0.21.0). The current
--gpu-memory-utilization=0.7800 is equivalent to --gpu-memory-utilization=0.7688 without
CUDA graph memory profiling. To maintain the same effective KV cache size as before,
increase --gpu-memory-utilization to 0.7912.
```

Both runtimes comfortably exceed the 1M calibrated context ceiling at gmu 0.78, which is the
only thing that actually matters here.

## Warm-up asymmetry (why these numbers are comparable anyway)

Worth stating because it nearly invalidated this whole comparison. Our production runtime has
a **large cold-start penalty**: immediately after `Application startup complete`, with graphs
captured and short warm-up calls sent, `count300` measured **58.5 tok/s**; after a few long
generations it settled at **83.3, 83.2, 83.1, 83.2**. The Anemll baseline figures, by
contrast, were all taken within ~1 minute of startup — so if 0.25.2 had the same penalty,
this bake-off would have been unfairly stacked against it.

It does not. Re-booted and heavily warmed (five 600-700-token generations plus short calls
before measuring), Anemll produced:

```
76.6   76.7   76.2   76.6      → peak 76.7
```

versus **77.2** measured cold on the first boot and **77.4** on the second. **0.25.2 shows no
meaningful cold-start penalty** — it front-loads FlashInfer autotune, sparse-MLA warmup and
graph capture at boot, which the older runtime defers to first traffic. So the ~9% decode gap
is real and not a warm-up artefact, and 0.25.2 deserves credit for being ready when it says
it is ready.

## Why decode bursts then drops (measured, not theorised)

The 0.25.2 runtime exposes per-position acceptance, which our production stack does not
log. This is the clearest evidence we have of the burst→sustained mechanism, captured on
three consecutive 10-second windows of one benchmark run:

```
window 1   Drafted 62.5 tok/s   Accepted 51.1 tok/s   mean accept length 5.09
           per-position: 0.968  0.912  0.848  0.728  0.632

window 2   Drafted 64.0 tok/s   Accepted 32.6 tok/s   mean accept length 3.55
           per-position: 0.812  0.602  0.461  0.367  0.305

window 3   Drafted 64.0 tok/s   Accepted 16.5 tok/s   mean accept length 2.29
           per-position: 0.680  0.344  0.141  0.078  0.047
```

**Drafted throughput is pinned at ~64 tok/s across all three windows.** The step rate never
changes. Output speed fell 51 → 16 tok/s purely because the accepted *fraction* collapsed
as the content became less predictable.

Two consequences worth internalising before tuning anything:

1. **The runtime's hard ceiling is `drafted ÷ k × (k+1)`.** On Anemll: `64 ÷ 5 × 6 = 76.8`,
   and the measured peak was 77.2. You cannot exceed it by any amount of prompt-picking —
   only by making a step cheaper.
2. **Deeper drafts buy nothing on hard content.** Positions 4 and 5 accept at 0.078 and
   0.047 in window 3. This is why `k=7` (rejected at boot, must be a multiple of
   `n_predict=5`) and `k=10` (boots, then crashes every generation) are not the missing
   speed — there is no speed there to find.

If you are chasing sustained rather than burst throughput, the only levers that move it are
the ones that reduce step time (CUDA graphs, working IB/RoCE, compile enabled, fewer nodes
for a model that fits) or that raise acceptance on genuinely hard content — which is a
draft-model-quality problem, not a flag.

## Can 0.25.2 be tuned into the lead?

Two levers target the step-time deficit directly, and both were tested together rather than
assumed:

1. **Re-enable torch.compile** (`VLLM_USE_BREAKABLE_CUDAGRAPH=0`). **Dead end, verified.**
   The mode flips to `CompilationMode.VLLM_COMPILE` and the engine then says
   ``torch.compile` is turned on, but the model ... does not support it``. There is nothing
   to recover here — compile is unavailable for DeepSeek-V4-Flash on 0.25.2 regardless.
2. **Capture the real decode batch size.** The image captures
   `[1,2,4,8,16,24,32,40,48,56,64,72]`, which **omits 36** — and 36 is exactly
   `max_num_seqs 6 x (k5 + 1)`, the steady-state decode batch. An uncaptured steady-state
   shape means graph replay misses on the hot path.

**Both tested together, in a full reboot + rebench. Split result: nothing for single-stream
decode, a large gain for concurrency.**

Single stream — unchanged to within noise:

| | Anemll default | + capture 36 |
| --- | ---: | ---: |
| count300 | 77.2 | 77.4 |
| mult12 | 71.3 | 71.3 |
| decode mean | 62.7 | 62.2 |
| drafted throughput | ~64 tok/s | ~63 tok/s |

Concurrency — a real, large gain:

| conc | Anemll default | + capture 36 | change |
| ---: | ---: | ---: | ---: |
| 1 | 61.1 | 62.5 | +2% |
| 2 | 101.6 | **114.0** | +12% |
| 4 | 116.4 | **127.3** | +9% |
| 6 | 152.5 | **174.6** | **+14.5%** |

Drafted throughput stayed pinned at ~63-64 tok/s throughout, so this is not a step-rate
change on the single-stream path — it is graph replay hitting instead of missing once the
batched decode shape is actually captured.

### This independently confirms PR #5 (credit @Wpnx330)

The gain lands exactly where the theory says it should: `max_num_seqs 6 x (k5 + 1) = 36` is
the steady-state batched decode shape, Anemll's default capture list omits 36, and adding it
is worth **~14.5% at c6**. That is the same failure mode @Wpnx330 fixed in
[PR #5](https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark/pull/5)
— a capture size that does not match `seqs x (k+1)` silently truncates graph capture under
concurrency. **This repo already derives it correctly, which is a large part of why it wins
c4 and c6.** If you run any other DS4 stack, check this first.

### …and it wrecked prefill, which is the other half of the same test

| depth | Anemll default | + capture 36 + compile requested | change |
| ---: | ---: | ---: | ---: |
| 8K | 1446 | 1282 | **-11%** |
| 32K | 2488 | 1768 | **-29%** |
| 100K | 2704 | 2017 | **-25%** |

**Honest caveat: these two levers were tested together, so the results are confounded.** The
most consistent reading of the data is that they are separable and act on different paths:

- **`cudagraph_capture_sizes` including 36 → the concurrency win.** It only affects batched
  decode graph replay, which is exactly where the gain appeared, and drafted throughput
  (single-stream step rate) never moved.
- **Requesting compile mode → the prefill loss.** Turning compile on flips `custom_ops` from
  `'all'` to `'none'` and swaps the `vllm_c` RMS-norm kernels for native PyTorch. Since
  compile then *refuses* this model, you pay for disabled custom ops and get no compilation
  back. Prefill is the compute-bound path that leans hardest on those kernels, and prefill
  is what collapsed.

I did not run the two separately, so treat the attribution above as the best explanation
rather than a proven one.

**Practical upshot for anyone on 0.25.2:** take the capture-size fix, leave compile alone —
pass `--compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,24,32,36,40,48,56,64,72]}'`
and do **not** set `VLLM_USE_BREAKABLE_CUDAGRAPH=0`. That should get the c6 gain without the
prefill regression. Untested in that combination; measure it before trusting it.

One trap worth recording: requesting compile mode on this model also flips `custom_ops` from
`'all'` to `'none'` and swaps the `vllm_c` RMS-norm kernels for native PyTorch. Because
compile then refuses the model, you get custom ops **disabled** and no compilation in
exchange — strictly worse on paper. It measured neutral here, but there is no version of
this lever that wins, so do not ship it.

**Conclusion: the ~9% step-time gap is the kernel stack itself, and it is not configurable
away from the 0.25.2 side.**

## Quality

Anemll is not broken, just slower — it passed the same garble gate as production, 5/5 clean
(700/655/539/638/243-token generations across code, prose, reasoning, tool-shaped and mixed
content; checked for soft-empty, repetition loops, CJK drift and template leakage).

## Recommendation

**Stay on the runtime this repo ships.** Concretely, the config in
[`DEFAULT-CONFIG.md`](DEFAULT-CONFIG.md) with `MTP_NUM_TOKENS=5` and
`draft_sample_method: probabilistic` remains the best measured setup on 2x DGX Spark.

Keep the Anemll image around for two things:

- **Getting started without a local build.** It is public, needs no auth, has
  `ENTRYPOINT ["vllm","serve"]` and no baked CMD — so you pass the full arg list and none
  of the baked-command landmines in [`SPEED-UPDATE-2026-07-29.md`](SPEED-UPDATE-2026-07-29.md)
  apply. It accepted every flag in our default config unchanged.
- **Diagnosing acceptance.** Its `SpecDecoding metrics` line (mean acceptance length,
  drafted vs accepted throughput, per-position rates) is the fastest way to tell a
  step-time problem from an acceptance problem. If your `Drafted throughput` is low, you
  have a step-time/fabric problem. If `Drafted` is healthy and `Accepted` is low, you are
  looking at normal content-driven variance, not a bug.

## Reproducing

```bash
# both nodes
docker pull ghcr.io/anemll/dspark-vllm-gx10:0.1.1
# head, then worker
bash ds4lane_anemll.sh 0     # on the head node
bash ds4lane_anemll.sh 1     # on the worker
# then
URL=http://<head>:8889/v1 TAG=anemll python3 bench_full.py
```

Launcher and harness: [`scripts/ds4lane_anemll.sh`](scripts/ds4lane_anemll.sh),
[`benchmarks/bench_full.py`](benchmarks/bench_full.py).

## Credits

- **@fraserprice** — `DeepSeek-V4-Flash-DSpark` weights and the DSpark speculator.
- **Anemll** — the prebuilt `dspark-vllm-gx10:0.1.1` GB10 image. It lost this bake-off but
  it is the reason the comparison was possible at all without a local vLLM build, and its
  observability is better than ours.
- **@MiaAI-Lab / Simone** — for publishing numbers on the newer runtime, which is what
  prompted actually testing it instead of assuming our stack was the ceiling.
- **@Roady001**, **@Wpnx330**, **@AndreasKunar**, **@DaveCharland**, **@paulbrav** — issue
  reports and fixes that produced the default config this was measured against.
