# bench after — 2026-09-19T10:44:37+00:00
- tier 1, rounds 3, max_conc 4, temp 0.0
- model glm-5.3-flash @ http://127.0.0.1:8000 (container: glm53-intel-w4a16)
- KV: (EngineCore pid=577) INFO 09-19 10:33:45 [kv_cache_utils.py:2274] GPU KV cache size: 1,994,013 tokens, Maximum concurrency for 1,048,576 tokens per request: 1.90x
- acceptance: 0.9932
- MemAvailable: 2.51 GiB

| cell | conc | median tok/s | min–max | rounds |
|---|---|---|---|---|
| c1_prose_short | 1 | 38.93 | 38.89–39.07 | 3 |
| c1_code_short | 1 | 39.7 | 39.62–39.73 | 3 |
| c1_prose_medium | 1 | 37.980000000000004 | 37.93–38.03 | 2 |
| c1_prose_long | 1 | 33.69 | 30.95–36.43 | 2 |
| c4_prose_short | 4 | 116.14 | 113.12–116.83 | 3 |
| pmu_replay_long | 1 | 14.34 | 14.34–14.34 | 1 |
