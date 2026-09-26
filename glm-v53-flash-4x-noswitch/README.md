# glm-v53-flash-4x-noswitch — operations README

GLM-5.3-Flash FP8 served on the **4-node switchless ConnectX-7 ring** with vLLM TP4,
DFlash2 speculative decoding, SparkCache + SIRCL, and patched NCCL — using the
[jnardiello recipe](https://github.com/jnardiello/GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless)
(E29 @ `080fe09`) **as-is**, site-configured for this cluster. Full build history and
deviations: [build.md](build.md).

**Endpoint:** `http://10.69.42.170:8000/v1` — model name `glm-5.3-flash`,
262,144-token context, **no auth/TLS — trusted LAN only**. Concurrency ≈ 5 sessions
at full context (6 dies — memory is tight by design).

## Cluster map

| rank | node | mgmt IP | role |
|---|---|---|---|
| 0 | `spark-0f0b` | 10.69.42.170 | leader — API server (`:8000`), rendezvous |
| 1 | `spark-6d14` | 10.69.42.171 | worker (rank-2 relay hop for weight fan-out) |
| 2 | `spark-6d90` | 10.69.42.172 | worker |
| 3 | `spark-6d24` | 10.69.42.173 | worker |

- SSH from the workstation via `*.shawndo.intra` (never the fabric IPs). Ring fabric:
  10.10.1.x/2.x/3.x/4.x links, MTU 9000, **odd links on each node's LEFT QSFP port
  (f0), even links on the RIGHT (f1)**.
- Container: `glm53_fp8_dflash_tp4`, image
  `ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:0d4029b3…` (content ID
  `5e32aaa1…` — verified on every node by `verify-node.sh`).
- Node paths: `~/tp4` (runtime assets), `~/glm53-flash-fp8-zai` (306 GiB FP8 weights),
  `~/glm53-dflash2-draft`, `~/nccl-patched`, `~/vllm-cache`.
- Workstation recipe checkout: `C:\Users\sdrew\code\glm-4x-noswitch`
  (run scripts from **WSL**, `cd /mnt/c/Users/sdrew/code/glm-4x-noswitch`).
- Site files (committed in this directory): `site/cluster.env`,
  `site/versions.env`, `site/sircl/`.

## Everyday commands (from the workstation checkout)

```sh
cd /mnt/c/Users/sdrew/code/glm-4x-noswitch

./scripts/tp4ctl status          # per-rank container state + endpoint probe
./scripts/tp4ctl health          # /health + live smoke chat completion
./scripts/tp4ctl fabric-check    # 2 MTU-9000 ports/rank + 8/8 jumbo pings
./scripts/verify-node.sh         # static verification (expect: 157 PASS / 0 FAIL)
./scripts/verify-node.sh --live  # + runtime signatures (E21/E22b/E27/E29, payloads)
./scripts/tp4ctl down            # stop the whole cluster
./scripts/tp4ctl up              # start (launch order 3→2→1→0 is handled for you)
./scripts/tp4ctl restart         # cycle all ranks (after any cluster.env/deploy change)
```

Start to first usable response is ~10 minutes (306 GiB load + B12X/FlashInfer warmup +
CUDA-graph capture). `/health` returning `000` during that window is normal — don't
restart mid-load; watch `docker logs` instead.

## Quick health checks

```sh
curl -s http://10.69.42.170:8000/health                          # 200 = serving
curl -s http://10.69.42.170:8000/v1/models | python3 -m json.tool

# the two post-boot functional gates (run within 2 min of first /health 200):
curl -s http://10.69.42.170:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","temperature":0,"max_tokens":64,
       "chat_template_kwargs":{"enable_thinking":false},
       "messages":[{"role":"user","content":"What is the capital of Italy? Reply with one sentence."}]}'
# pass: coherent "Rome" in content (thinking-off template adapter)

curl -s http://10.69.42.170:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","max_tokens":256,
       "messages":[{"role":"user","content":"What is the weather in Milan?"}],
       "tools":[{"type":"function","function":{"name":"get_weather",
         "description":"Get weather for a city",
         "parameters":{"type":"object","properties":{"city":{"type":"string"}},
         "required":["city"]}}}],"tool_choice":"auto"}'
# pass: tool_calls[0].function.name == get_weather with {"city":"Milan"}
```

Simple chat (thinking on by default; use the gate's `enable_thinking:false` for direct
answers):

```sh
curl -s http://10.69.42.170:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","max_tokens":512,
       "messages":[{"role":"user","content":"Hello, who are you?"}]}'
```

## Logs

On any node: `sudo docker logs -f glm53_fp8_dflash_tp4`. The stream is **chatty by
design** — the E20 memory probe emits a ~4 KB JSON line every second per rank plus
per-second metrics. Docker log rotation is configured cluster-wide (`256m × 4` files
per container in `/etc/docker/daemon.json`), so it cannot fill the disk; old rotated
logs cap at ~1 GB/node. To quiet the probe itself you would edit the mounted
`gpu_worker.py` override (`interval_seconds=1.0`) and re-deploy — a recipe change,
not done.

If a line contains `E20_MEMORY_PROBE` or scheduler/metrics INFO lines: normal. Real
trouble looks like: NCCL timeouts/hangs at init, `Connection reset` between ranks,
OOM (`NV_ERR_NO_MEMORY`), or a rank container exiting while others stay up.

## Invariants — do not break

- **One model at a time.** All recipes use every reserved GPU; run `tp4ctl down`
  before starting anything else.
- **Never hand-edit generated files**: per-node netplan/iptables (`scripts/node/etc/*`,
  rendered by `render-netplan.sh`) and `~/tp4/*` on the nodes (pushed by
  `deploy.sh`). Edit `cluster.env` in the checkout, then re-render/deploy.
- **Measured knobs are not shared tuning knobs**: `GPU_MEM_UTIL=0.85`, the 16 GiB KV
  pool (`--kv-cache-memory-bytes`), `MAX_NUM_SEQS=6`, KV dtype, backend flags, spec
  schedule (k=7/3). Don't tune them casually; every one has a measured baseline and a
  matching `TP4_ENV` rollback overlay (see `cluster.env` header + upstream
  `docs/operations.md`).
- **Kernel/driver are pinned and held** on all four nodes:
  `6.17.0-1032-nvidia` + `580.173.02` (the 7.x kernel breaks ring NCCL — forum
  383023). Don't run `apt upgrade` on the nodes or lift the holds until NVIDIA ships
  a fixed kernel.
- **Recipe changes flow through the upstream update flow** (upstream repo + forum
  review), then `./scripts/deploy.sh` + `./scripts/tp4ctl restart` — never edit files
  directly on the nodes.
- **Benchmarking**: never right after a boot (cold JIT/warmup skews results ~30%);
  use `stream:false` and read `usage.completion_tokens` (streamed deltas under-report).
- Recipe rollback lanes (from `cluster.env` header): e.g.
  `TP4_ENV=scripts/node/reference/baseline-20260925-e28b.env` for the pre-E29 recipe —
  coordinated `down` → `deploy.sh` → `up` with the overlay set.

## Host / fabric notes

- Ring cabling: 4 DACs, one per QSFP port; **left port = odd links (L1, L3), right =
  even (L2, L4)**. Serial-based peer checks are unreliable on the fresh nodes (EEPROM
  reads are crossed between PCIe views) — verify with ARP/LLDP signatures if a link is
  suspect (see build.md §7b).
- Docker on ranks 2-3 uses the **classic overlay2 store** (containerd snapshotter
  disabled in daemon.json so the image ID matches the recipe pin on all nodes).
- IOMMU passthrough is active (`iommu.passthrough=1` in the kernel cmdline).
- Autostart: rank 0's `tp4-autostart.service` launches the whole cluster at boot —
  after a rank-0 reboot just wait; don't double-start with `up`.

## When something's wrong

1. `./scripts/tp4ctl status` — which ranks are up/down?
2. `./scripts/tp4ctl fabric-check` — 8/8 jumbo pings? A miss = fabric problem: fix the
   fabric first, never "repair one serving rank in isolation".
3. `./scripts/verify-node.sh` — the FAIL rows say which step owns each check.
4. Rank-0 container log for the failing phase (weights load, NCCL init, graph capture).
5. Full restart before deeper surgery: `tp4ctl down` → `tp4ctl up`. Stop and report
   anything that survives that.

References: [build.md](build.md) (build log + deviations) · upstream
[operations.md](https://github.com/jnardiello/GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless/blob/main/docs/operations.md)
· [fabric.md](https://github.com/jnardiello/GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless/blob/main/docs/fabric.md)
· forum thread
[382459](https://forums.developer.nvidia.com/t/glm-5-3-flash-on-tp4-dgx-sparks-switchless/382459)
