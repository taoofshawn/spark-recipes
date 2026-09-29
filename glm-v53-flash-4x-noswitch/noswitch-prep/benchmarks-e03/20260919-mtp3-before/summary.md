# bench before — 2026-09-19T10:08:37+00:00
- tier 1, rounds 3, max_conc 4, temp 0.0
- model glm-5.3-flash @ http://127.0.0.1:8000 (container: glm53-intel-w4a16)
- KV: None
- acceptance: 0.9968
- MemAvailable: 3.28 GiB

| cell | conc | median tok/s | min–max | rounds |
|---|---|---|---|---|
| c1_prose_short | 1 | 39.67 | 38.64–39.69 | 3 |
| c1_code_short | 1 | 39.54 | 37.65–39.99 | 3 |
| c1_prose_medium | 1 | 28.41 | 19.25–37.57 | 2 |
| c1_prose_long | 1 | 22.5 | 9.21–35.79 | 2 |
| c4_prose_short | 4 | 129.29 | 103.16–130.65 | 3 |
| pmu_replay_long | 1 | 13.79 | 13.79–13.79 | 1 |
