# bench before — 2026-09-19T11:00:34+00:00
- tier 1, rounds 3, max_conc 4, temp 0.0
- model glm-5.3-flash @ http://127.0.0.1:8000 (container: glm53-intel-w4a16)
- KV: (EngineCore pid=573) INFO 09-19 10:54:34 [kv_cache_utils.py:2274] GPU KV cache size: 1,920,956 tokens, Maximum concurrency for 1,048,576 tokens per request: 1.83x
- acceptance: 0.9976
- MemAvailable: 3.0 GiB

| cell | conc | median tok/s | min–max | rounds |
|---|---|---|---|---|
| c1_prose_short | 1 | 39.64 | 39.1–40.03 | 3 |
| c1_code_short | 1 | 40.13 | 37.56–40.13 | 3 |
| c1_prose_medium | 1 | 28.08 | 18.73–37.43 | 2 |
| c1_prose_long | 1 | 23.095 | 9.47–36.72 | 2 |
| c4_prose_short | 4 | 131.78 | 113.16–132.07 | 3 |
| pmu_replay_long | 1 | 14.16 | 14.16–14.16 | 1 |
