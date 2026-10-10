# bench fp8-4pf — 2026-10-10T20:12:01+00:00
- tier 1, rounds 3, max_conc 4, temp 0.0
- model glm-5.3-flash @ http://127.0.0.1:8000 (container: glm53_fp8_dflash_tp4)
- KV: (EngineCore pid=222) INFO 10-10 19:38:19 [kv_cache_utils.py:2087] GPU KV cache size: 1,194,033 tokens, Maximum concurrency for 262,144 tokens per request: 4.55x
- acceptance: 0.9808
- MemAvailable: 4.62 GiB

| cell | conc | median tok/s | min–max | rounds |
|---|---|---|---|---|
| c1_prose_short | 1 | 94.41 | 93.94–94.48 | 3 |
| c1_code_short | 1 | 90.33 | 80.95–90.33 | 3 |
| c1_prose_medium | 1 | 53.94 | 36.13–71.75 | 2 |
| c1_prose_long | 1 | 27.265 | 16.73–37.8 | 2 |
| c4_prose_short | 4 | 170.18 | 166.74–181.67 | 3 |
| pmu_replay_long | 1 | 8.11 | 8.11–8.11 | 1 |
