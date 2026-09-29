# bench after — 2026-09-19T10:39:46+00:00
- tier 1, rounds 3, max_conc 4, temp 0.0
- model glm-5.3-flash @ http://127.0.0.1:8000 (container: glm53-intel-w4a16)
- KV: (EngineCore pid=577) INFO 09-19 10:33:45 [kv_cache_utils.py:2274] GPU KV cache size: 1,994,013 tokens, Maximum concurrency for 1,048,576 tokens per request: 1.90x
- acceptance: 0.9947
- MemAvailable: 2.5 GiB

| cell | conc | median tok/s | min–max | rounds |
|---|---|---|---|---|
| c1_prose_short | 1 | 39.18 | 38.44–39.25 | 3 |
| c1_code_short | 1 | 38.8 | 37.12–39.68 | 3 |
| c1_prose_medium | 1 | 28.185 | 18.71–37.66 | 2 |
| c1_prose_long | 1 | 20.515 | 9.09–31.94 | 2 |
| c4_prose_short | 4 | 116.46 | 115.2–117.49 | 3 |
| pmu_replay_long | 1 | 15.09 | 15.09–15.09 | 1 |
