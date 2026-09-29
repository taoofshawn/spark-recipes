# bench before — 2026-09-19T10:24:00+00:00
- tier 1, rounds 3, max_conc 4, temp 0.0
- model glm-5.3-flash @ http://127.0.0.1:8000 (container: glm53-intel-w4a16)
- KV: (EngineCore pid=582) INFO 09-19 10:19:06 [kv_cache_utils.py:2274] GPU KV cache size: 1,867,536 tokens, Maximum concurrency for 1,048,576 tokens per request: 1.78x
- acceptance: 0.9799
- MemAvailable: 2.04 GiB

| cell | conc | median tok/s | min–max | rounds |
|---|---|---|---|---|
| c1_prose_short | 1 | 68.22 | 65.9–68.3 | 3 |
| c1_code_short | 1 | 68.72 | 61.35–68.86 | 3 |
| c1_prose_medium | 1 | 42.455 | 22.74–62.17 | 2 |
| c1_prose_long | 1 | 30.224999999999998 | 10.37–50.08 | 2 |
| c4_prose_short | 4 | 193.99 | 193.45–202.56 | 3 |
| pmu_replay_long | 1 | 12.78 | 12.78–12.78 | 1 |
