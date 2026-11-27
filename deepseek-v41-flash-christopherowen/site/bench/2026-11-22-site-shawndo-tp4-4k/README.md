# Record run — `2026-11-22-site-shawndo-tp4-4k`

First run of this recipe on the live cluster. Engine warm (JIT complete), Sonar pinned at
`110945d61238`, shape `tp4-ring-4k` with the shipped `NCCL_SWITCHLESS_BIDIRECTIONAL=2`.
Full receipts: `matrix.json` (quality gate + decode matrix), `single-stream-c1.json`.

Quality gate: 5 checks, all ok.

| point | tps mean | CI95 |
|---|---|---|
| prose-c1 | 65.267 | 64.9% |
| prose-c2 | 137.052 | 1.07% |
| prose-c4 | 245.413 | 3.67% |
| prose-c8 | 395.761 | 2.25% |
| prose-c16 | 521.325 | 15.6% |
| code-c1 | 97.141 | 0.61% |
| code-c2 | 170.430 | 1.36% |
| code-c4 | 291.333 | 1.60% |
| code-c8 | 488.612 | 1.22% |
| code-c16 | 720.849 | 0.95% |

Single-stream (closed, 60 s warm-up + 300 s in 10 windows, 109 requests): **77.88 tps ±7.45%**,
tpot p50 12.25 ms, ttft p50 0.121 s, accepted per draft 1.4485. Sonar flagged `foreign: 1` — the
server was not serving this run alone, so the point is marked non-comparable; D5's ~78 tps c1
reference holds.

Low-water MemAvailable during the run: head 23.83 GiB, workers 25.49 / 25.50 / 25.57 GiB; swap growth
0.0 GiB everywhere; no thermal slowdown on any node (GPU max 59–61 °C).

`cluster_config_sha256` for this run: `4b294da1705f23c4897adada86950cc825ffb0343d319f0db2c8ad9cbe12e0a8`.
