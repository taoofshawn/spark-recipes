---
name: troubleshoot-slowness
description: >-
  Use when a served model on the DGX Spark cluster shows degraded throughput or
  latency — Grafana/vLLM stats dropping, slow agent turns, TTFT spikes, KV-cache
  climbing, requests queuing or waiting. Covers /metrics collection, engine-log
  timeline reconstruction, host-memory and GPU checks, JIT-event triage, and
  per-recipe concurrency-cap analysis. Triggers: "server is slow", "stats
  dropped", "ttft spiked", "kv-cache high", "throughput collapsed",
  "troubleshoot <recipe>", "agent feels slow". NOT for boot failures or deploy
  problems (that is bring-up-spark-recipe) and NOT for recipe research passes
  (that is update-recipe).
---

# Troubleshoot Slowness

Evidence-first diagnosis of a *running* vLLM serve on the 2-node GB10 cluster.
Never guess from a Grafana glance: reconstruct the engine's own timeline first,
then check the four layers in order — scheduler → host memory → GPU → spec decoder.

**Core principle:** a step-change in every metric at once almost always means the
*workload changed* (concurrency arrived, context grew), not that hardware degraded.
Prove it with numbers before touching config.

## Step 0 — Identify the LIVE deployment (never assume repo defaults)

The deployed `.env` on the nodes drifts from repo defaults. Read it from the
container, not from the repo:

```bash
ssh spark-0f0b.shawndo.intra 'docker ps --format "{{.Names}} {{.Status}}"; \
  docker exec <container> printenv | grep -E "^(PORT|MPORT|NODE_RANK|HEADLESS|GMU|MAX_SEQS|MAX_LEN|MNBT|KV_CACHE_MEMORY|KV_DTYPE|ASYNC|EAGER|DFLASH_TOKENS|MODEL_REVISION|SERVED_MODEL_NAME)="'
```

Gotchas learned the hard way:
- **Port drift is real** (seen: PORT=4000 while repo default was 8000; curl on the
  default silently returned empty).
- **Timezones:** container logs are UTC, host `date` and Grafana are local (EDT = UTC−4
  in summer). Convert the user's reported timestamp BEFORE grepping logs — grepping
  the local hour finds nothing.
- `ss -tlnp` on the head may not show the port if you guessed it wrong; if curl is
  empty, you have the wrong port. Check the APIServer log lines for real traffic.

## Step 1 — Metrics snapshot (the gauges that matter)

```bash
curl -s http://127.0.0.1:<PORT>/metrics > /tmp/m.txt   # ON the head node
grep -E "vllm:(kv_cache_usage_perc|num_requests_running|num_requests_waiting_by_reason|prefix_cache_(queries|hits)_total|prompt_tokens_total|generation_tokens_total|request_success_total)" /tmp/m.txt | grep -v "^#"
```

Interpretation:
- `num_requests_waiting_by_reason{reason="capacity"}` > 0 → scheduler is full
  (MAX_SEQS and/or KV pool are binding).
- `prefix_cache_hits_total / prefix_cache_queries_total` — the long-run rate; a
  *dropping* rate means evictions started.
- KV usage percent alone is **not** a threshold — vLLM does nothing special at 40%.
  Correlation with a slowdown usually means workload grew, not that 40% is magic.

## Step 2 — Reconstruct the engine timeline from logs (usually sufficient; Prometheus optional)

vLLM logs a stats line every 10 s. The format uses CAPITALIZED `Running`/`Waiting`
and includes the KV gauge inline:

```bash
ssh spark-0f0b.shawndo.intra 'docker logs <container> 2>&1 | sed -nE "s/^.*([0-9]{2}:[0-9]{2}:[0-9]{2}).*Avg prompt throughput: ([0-9.]+) tokens\/s, Avg generation throughput: ([0-9.]+) tokens\/s, Running: ([0-9]+) reqs, Waiting: ([0-9]+) reqs, GPU KV cache usage: ([0-9.]+)%.*/\1 pp=\2 tg=\3 run=\4 wait=\5 kv=\6/p" > /tmp/tl.txt'
```

(Adjust for the version's exact wording; grep one raw line first to confirm the
format. Lines that look truncated in display are still full on disk.)

Signature table:

| pattern in timeline | cause |
|---|---|
| run jumps up + wait > 0, tg (aggregate) collapses, all at one timestamp | concurrency arrived — workload change, not hardware |
| pp (prompt) spikes 4–6K+ tok/s while run is low | giant prefills / re-prefill after eviction; decode starved by chunked-prefill budget |
| KV sawtooth (up → down → up, e.g. 56→26→55) | sessions finishing/being evicted, then re-prefilling — pool too small for the session set |
| tg fine at run=1, collapses at run 3–4 | spec-decoder batch cliff (DFlash2); see Step 5 |
| log timestamp GAP of minutes | real stall (JIT compile, swap thrash, NCCL) — correlate with Step 3/4 findings |

Compare the healthy baseline (single request rate) with the degraded window at the
same run level. If run=1 was fast before AND after the incident, hardware is fine.

## Step 3 — Host memory (GB10 is UMA: GPU memory IS system RAM)

```bash
ssh spark-0f0b.shawndo.intra 'free -g; vmstat 1 5; sar -W | tail -25'   # pswpin/s pswpout/s history
# who is swapped:
for f in /proc/[0-9]*/status; do s=$(grep VmSwap $f 2>/dev/null | tr -dc "0-9"); [ "${s:-0}" -gt 10000 ] 2>/dev/null && echo "$s $(grep -m1 ^Name $f|cut -f2)"; done | sort -rn | head
```

- GMU × 121.7 GiB + KV pin + ~13 GiB OS/CUDA must fit; the DGX OS 7.5.0 OTA boots
  with ~7.2 GiB less RAM (repo watchlist) — 0.88 GMU leaves only ~5 GiB spare.
- Swap allocated at boot (model load) = normal. Calm `si/so` = 0 since warmup = fine.
  Sustained nonzero pswpin/pswpout during serving = real thrash.
- vLLM's own pages in VmSwap means dormant code paths major-fault on first touch
  (e.g. a new sampling kernel) — explains sporadic multi-second spikes.

## Step 4 — GPU health

```bash
nvidia-smi --query-gpu=index,utilization.gpu,temperature.gpu --format=csv
nvidia-smi -q -d CLOCK,PERFORMANCE | grep -E "SM +:|Throttle|Clocks Event|Perf"
```

P0 + full SM clocks + empty throttle counters = hardware exonerated. [N/A] memory
in csv output is normal on GB10 (unified memory — use `free -g` instead).

## Step 5 — Spec-decoder & JIT triage

```bash
docker logs <container> 2>&1 | grep -iE "jit_monitor" | tail   # kernel JIT during inference
docker logs <container> 2>&1 | grep "SpecDecoding metrics"     # mean acceptance length
```

- A single `Triton kernel JIT compilation during inference` line = one-off latency
  spike (new batch/prompt shape). Only a log GAP or repeated lines indicate sustained
  impact. New shapes appear exactly when concurrency changes.
- Acceptance length swinging 2–6 across windows is normal content variation; a
  sustained drop at fixed run level is a real regression (drafter/patch drift).

## Concurrency caps — the cap is NOT MAX_SEQS

MAX_SEQS is an admission ceiling, not a performance target. The right client cap:

```
cap = min( MAX_SEQS, KV_pool_tokens ÷ expected per-session context, spec-decoder's efficient batch )
```

Validated on this cluster (receipts in each recipe's research.md):

| recipe | KV pool | cap | why |
|---|---|---|---|
| glm-v53-flash-nvfp4-0rand | ~892K | **2** | DFlash2 collapses at batch ≥3; 30→3–16 tok/s agg measured 2026-09-15 |
| deepseek-v4-flash-vision-0rand | ~2.9M | **4** | c4 aggregate 59–68 tok/s measured; 100/100 prefix-cache HITs |
| glm-v53-flash-intel-w4a16 (MTP3 lane) | ~1.9M | **~6** | MTP's flatter batch curve; DFlash lanes share the DFlash cliff |

Cap in the CLIENT (agent harness) when possible: same engine effect, no restart,
and queue-time doesn't pollute TTFT metrics. Prefer client cap > MAX_SEQS change >
recipe switch. Recipe switch (e.g. to intel MTP3) only when sustained >4 concurrent
long sessions are the requirement; GPU contention means one recipe serves at a time.

## Reporting

State: root cause (workload vs hardware vs config), the evidence line from the
timeline, current cap arithmetic, and the least-disruptive fix. Document findings
in the recipe's `research.md` (dated entry) on a branch — never on the live nodes.
