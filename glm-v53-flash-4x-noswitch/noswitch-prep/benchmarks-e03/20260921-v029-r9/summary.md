# bench after-v029-r9 — 2026-09-21T09:41:26+00:00
- tier 1, rounds 3, max_conc 4, temp 0.0
- model glm-5.3-flash @ http://127.0.0.1:8000 (container: glm53-intel-w4a16-v029)
- KV: (EngineCore pid=586) INFO 09-21 09:29:31 [kv_cache_utils.py:2315] GPU KV cache size: 1,920,956 tokens, Maximum concurrency for 1,048,576 tokens per request: 1.83x
- acceptance: 0.9979
- MemAvailable: 3.37 GiB

| cell | conc | median tok/s | min–max | rounds |
|---|---|---|---|---|
| c1_prose_short | 1 | 38.9 | 38.23–39.24 | 3 |
| c1_code_short | 1 | 39.62 | 25.38–40.17 | 3 |
| c1_prose_medium | 1 | 28.87 | 19.68–38.06 | 2 |
| c1_prose_long | 1 | 23.035 | 9.4–36.67 | 2 |
| c4_prose_short | 4 | 118.38 | 117.69–131.88 | 3 |
| pmu_replay_long | 1 | 14.29 | 14.29–14.29 | 1 |
