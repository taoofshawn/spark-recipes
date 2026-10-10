---
name: troubleshoot-slowness
description: >-
  Use when a served GLM-5.3-Flash lane on the 4-node DGX Spark switchless cluster shows
  degraded throughput or latency — slow agent turns, TTFT spikes, KV-cache climbing,
  requests queuing or waiting, aggregate tok/s dropping. Covers /metrics collection,
  engine-log timeline reconstruction, transport checks (four-PF ring), and per-lane
  concurrency-cap analysis. Triggers: "server is slow", "stats dropped", "ttft spiked",
  "kv-cache high", "throughput collapsed", "troubleshoot <recipe>", "agent feels slow".
  NOT for boot failures or deploy problems (bring-up-spark-recipe) and NOT for recipe
  research passes (update-recipe).
---

# Troubleshoot Slowness (4x-switchless lanes)

Evidence-first diagnosis of a *running* GLM-5.3-Flash lane on the 4-node switchless
ring. Never guess from a dashboard glance: reconstruct the engine's own timeline first.

**Core principle:** a step-change in every metric at once almost always means the
*workload changed* (concurrency arrived, context grew), not that hardware or the
transport degraded. Prove it with numbers before touching config.

## Step 0 — Identify the LIVE lane (never assume)

```bash
ssh spark-0f0b.shawndo.intra 'docker ps --format "{{.Names}} {{.Status}}"'
curl -s http://10.69.42.170:8000/v1/models | python -c "import json,sys; d=json.load(sys.stdin)['data'][0]; print(d['id'], d.get('max_model_len'))"
```

NVFP4 (`glm53-nvfp4-r*`, max_model_len 1,048,576) vs FP8 (`glm53_fp8_dflash_tp4`,
262,144). Each lane has its own tuned profile — never compare one lane's numbers against
the other lane's, and never against other operators' switched-fleet numbers: this ring
is latency-bound at c1 by design (see the lane's bench history).

## Step 1 — Engine-log timeline

FP8: `tp4ctl logs` (or `docker logs glm53_fp8_dflash_tp4` per rank). NVFP4:
`docker logs glm53-nvfp4-r0` (head) + worker ranks. Build the 10-s stats-tick timeline
(`pp= tg= run=`): find when the degradation started, what changed at that tick (new
concurrency? long prefill? KV pressure?).

**Known benign vs real signatures on this cluster:**

| signature | verdict |
|---|---|
| one-tick `pp=0 tg=0 run=1` stall between agent turns | the known prefill/KV-transfer barrier form (FP8 2026-09-28 record) — self-recovering; quantify frequency, don't restart |
| stats-tick gaps during IDLE periods | the logger going quiet — judge gaps ONLY while requests are in-system |
| `No available shared memory broadcast block found in 60 seconds` | benign — another rank compiling |
| `DEVSELECT_BUILD_FAILED` (NVFP4, rank 0) | known watch item; decode continues on the fallback — not the cause of slowness |
| NCCL WARN / QP timeouts / `Connection reset` | REAL transport trouble — capture logs, check `verify-fabric-p2.sh`, do NOT restart blind |

## Step 2 — Scheduler / admission layer

- FP8: `MAX_NUM_SEQS=6` + the bounded-admission middleware
  (`TP4_ADMISSION_MAX_ACTIVE=6`, queue limits) — requests beyond the cap QUEUE by
  design; check `/metrics` for queue depth vs the admission caps before calling it
  slowness. Agent turns with ~100K-token bodies buffer at the proxy (~every 45–90 s
  in proxy logs = big turn bodies, correlated with TTFT tail).
- NVFP4: batch-uniform verify + the profile's scheduler caps; json prefill chunking
  under the 6919-token budget is the known TTFT shape at c8/c16 (0.5 s post-Phase-E;
  if it regresses toward ~6.5 s, suspect a transport regression first).

## Step 3 — Transport (four-PF ring)

Decode far below the lane's own bench history (NVFP4 c1 ~44/77/96 tok/s prose/code/json;
FP8 c1 prose ~94) suggests transport. Check: boot logs still show all four PFs carrying
QPs on every rank (`RouteFinal`), `verify-fabric-p2.sh` ALL PASS, no NCCL WARNs since
boot. Do NOT "fix" c1 latency with config — it is ring-physics; the lever was the
four-PF transport (done) and the remaining gap is known.

## Step 4 — Host memory / GPU

Per node: `free -g` (MemAvailable), `nvidia-smi` (throttle reasons, ECC), docker stats.
The FP8 lane runs tight on rank 0 by design (GPU_MEM_UTIL 0.85; `NV_ERR_NO_MEMORY` at
0.89+). KV-cache climbing to the pin is normal; preemption churn under it is the signal.

## Step 5 — Spec-decode health

Read `/metrics` spec-decode counters before/after a warm generation (skip the first —
JIT): acceptance healthy at pos-0 ~0.6–0.9 decaying gently per position; acceptance
pinned near 1.0 or pos-0 collapsed means the drafter/aux glue broke, not slowness.
Remember: single-stream tok/s is acceptance-driven and prompt-dependent — always quote
the prompt type with any number (NVFP4 measured prose 44 / code 77 / json 96 in one
session).

## Step 6 — Warm-up

Numbers taken within ~30 min of a boot are ~30% skewed (JIT/TRITON caches). Re-measure
warm before diagnosing. Long sessions on some stacks degrade (node reboot resets GB10
allocation-pool degradation — FP8 README hard-won rule 6).

## Stop conditions

Never restart a lane to "fix" slowness without captured evidence (`docker logs` first —
the FP8 README rule). Record the diagnosis in the lane's `research.md` changelog.
