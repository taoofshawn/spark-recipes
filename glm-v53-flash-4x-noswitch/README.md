# glm-v53-flash-4x-noswitch — operations README

GLM-5.3-Flash FP8 served on the **4-node switchless ConnectX-7 ring** with vLLM TP4,
DFlash2 speculative decoding, SparkCache + SIRCL, and patched NCCL — using the
[jnardiello recipe](https://github.com/jnardiello/GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless)
(E29 @ `080fe09`) **as-is**, site-configured for this cluster. Full build history and
deviations: [build-record.md](noswitch-prep/build-record.md).

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
- Node paths (2026-09-27 cleanup — nothing recipe-owned in `$HOME`): `~/.local/tp4`
  (deployed runtime; `tp4ctl` on PATH via `~/.local/bin`), weights + drafter in the
  default HF cache (`~/.cache/huggingface/hub/models--zai-org--GLM-5.3-Flash/…`,
  `models--incoai--GLM-5.3-Flash-DFlash2/…`), `~/.local/lib/nccl-patched`,
  `~/.cache/tp4-vllm-cache` (container `/cache`).

## Locations (post-reorganization, 2026-09-26)

All paths below are relative to the **repo root** (the `spark-recipes` checkout).

| path (from repo root) | what it is |
|---|---|
| `glm-v53-flash-4x-noswitch\` | this recipe's branch dir — full tree annotated in "How this recipe fits together" below |
| `glm-v53-flash-4x-noswitch\upstream\` | the **live recipe checkout** (jnardiello @ `080fe09`, detached). All tp4ctl/verify/deploy scripts run from here. Carries 5 site modifications: `upstream/scripts/node/bootstrap/versions.env`, `upstream/scripts/node/nccl/SHA256SUMS` (adopted `afe5f486…`), `upstream/scripts/node/nccl/build.sh` (PATCH_FILE fix), `upstream/CHANGELOG.md`, plus the site path rewrites across `upstream/scripts/` (research.md §2.5) |

**Sparks:**

| node | git clones | per-node runtime assets |
|---|---|---|
| spark-0f0b | `~/code/spark-recipes` — on branch `glm-v53-flash-4x-noswitch` (kept current) | `~/.local/tp4/`, HF cache (weights + drafter), `~/.local/lib/nccl-patched/`, `~/.cache/tp4-vllm-cache/` |
| spark-6d14 | `~/code/spark-recipes` — on branch `glm-v53-flash-4x-noswitch` (switched from stale `main` on 2026-09-27) | same as 0f0b |
| spark-6d90, spark-6d24 | `~/code/spark-recipes` — cloned on branch `glm-v53-flash-4x-noswitch` (2026-09-27) | same as 0f0b (nodes consume `~/.local/tp4`) |

**Remote repos:**

| repo | role |
|---|---|
| `taoofshawn/spark-recipes` (origin) | canonical for the branch; tip: `3a0a1e2` |
| `github.com/jnardiello/GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless` | upstream reference @ `080fe09`; our checkout is detached — adopt upstream changes periodically by diffing against it (see `research.md` §2: 5 local mods in `upstream/` that must survive an update) |

Update playbook: `research.md` (what not to overwrite, update procedure, changelog).
Build history: `noswitch-prep/build-record.md`.

## How this recipe fits together — read this first

There are **three tiers**: the repo (source of truth, on the workstation + a clone on
every spark), the deployed runtime (a copy the repo pushes onto each spark), and the
serving container (built from a pinned image, assembled at launch from the runtime
tier). Nothing runs out of the repo on the sparks; the repo only *produces* the
runtime tier. Changing anything always flows: **repo edit → commit/push → pull on the
node checkout → `deploy.sh` (copies into `~/.local/tp4/`) → `tp4ctl restart`.**

### Tier 1 — repo layout (workstation, branch `glm-v53-flash-4x-noswitch`)

```
glm-v53-flash-4x-noswitch/
├── README.md              ← this file: ops orientation only
├── research.md            ← audit trail + UPDATE PLAYBOOK (read §2 before any update)
├── noswitch-prep/         ← THIS SITE's material (not upstream's)
│   ├── cluster.env        ← site config mirror (the live one is upstream/cluster.env)
│   ├── versions.env       ← kernel/driver pins (mirror of the upstream re-pin)
│   ├── build-record.md    ← closed historical build log (2026-09-26 bring-up)
│   ├── sircl/             ← generated per-rank SIRCL peer files + checksums
│   ├── benchmarks-e03/    ← archived pre-E29 benchmark results (reference only)
│   ├── bench-scripts/     ← archived one-off benchmark scripts
│   └── scripts/           ← 9 site build/ops helpers (kernel hold, hostkeys, relay…)
└── upstream/              ← the LIVE recipe checkout (jnardiello @ 080fe09, detached)
    ├── cluster.env        ← ACTIVE site config (gitignored — exists only where copied;
    │                        see research.md §5 for why node clones need it re-copied)
    ├── cluster.env.example← annotated public template; documents every knob
    ├── scripts/
    │   ├── tp4ctl         ← cluster controller: status/up/down/restart/health/logs
    │   ├── deploy.sh      ← deploys runtime tier to every node's ~/.local/tp4/
    │   ├── verify-node.sh ← static per-node verification (the 157-check gate)
    │   ├── check-f0.py    ← live identity check vs the frozen E29 baseline
    │   ├── launcher/launch-glm53-tp4.sh ← per-rank container assembly (docker run)
    │   └── node/          ← node-side assets: bootstrap pins (versions.env), NCCL build,
    │                        model manifests, moe-configs, vLLM override files,
    │                        reference/ = TP4_ENV rollback baselines (E21→E29)
    ├── third_party/       ← SparkCache connector/encoder + SIRCL bundle (vendored)
    └── docs/              ← upstream ops/install/fabric docs + frozen benchmarks
```

### Tier 2 — deployed runtime on each spark: `~/.local/tp4/`

`deploy.sh` copies this tree to every node (with SHA-256 manifests that
`verify-node.sh` checks). The serving container is assembled from it at launch:

```
~/.local/tp4/
├── tp4ctl                 ← the controller (also on PATH via ~/.local/bin/tp4ctl)
├── cluster.env            ← the active recipe — defines NODES, image pin, all knobs,
│                            and the -v mount list; changing it + restart = redeploy
├── launch-glm53-tp4.sh    ← builds the docker command for one rank
├── flusher-unconditional.sh ← page-cache flusher (runs while weights load)
├── sparkcache/            ← KV-cache connector + encoder, mounted into the container
├── sircl/                 ← SIRCL bundle + runtime incl. per-rank peer envs (rank0-3)
├── moe-configs/           ← fused-MoE kernel config for E=288/N=512 on GB10
├── overrides/             ← vLLM .py overrides, bind-mounted OVER the image's vLLM
├── experiments/e03/       ← scheduler + drafter code + kv-transfer configs (E03→E29)
├── reference/             ← frozen rollback recipes (TP4_ENV=.../baseline-*.env)
└── node/model-manifests/  ← per-revision weight manifests for integrity verification
```

Weights/drafter are NOT in this tree — they live in the default HF cache
(`~/.cache/huggingface/hub/models--…/snapshots/<rev>`, snapshot files hardlinked to
their blobs so the bind mount works inside the container), plus the patched NCCL at
`~/.local/lib/nccl-patched/` and the container's scratch/vLLM-JIT cache at
`~/.cache/tp4-vllm-cache/` (mounted as `/cache`).

### Tier 3 — the running stack, end to end

1. `tp4ctl up` (leader, or autostart systemd unit on rank 0 after reboot):
   fabric-check (2× MTU-9000 ports/rank, 8 jumbo pings) → cache flusher on →
   teardown → **launch ranks 3→2→1→0** (workers first so the rendezvous/store is
   listening when the head joins).
2. Per rank, `launch-glm53-tp4.sh` sources `cluster.env`, verifies paths/payload
   checksums, then `docker run`s the digest-pinned image with: the weights snapshot →
   `/model`, drafter snapshot → `/draft`, patched NCCL → `/opt/patched-nccl`
   (preloaded via `LD_PRELOAD`), cache → `/cache`, and every `overrides/`/
   `experiments/` `.py` mounted over the image's vLLM install — that is how the
   E21/E22b/E27/E29 features ship without rebuilding the image.
3. Rank 0 runs the API server on host port 8000; ranks 1–3 run headless workers.
   Health = `GET /health` 200 (never `/v1/models`). Boot to first token ≈ 10 min
   (306 GiB weight load + autotune + CUDA-graph capture).
4. Verify after any (re)start: both functional gates (below) within 2 min, then
   `python3 scripts/check-f0.py` — compares the live ranks' container config, mounts,
   env and boot receipts against the frozen E29 baseline (`docs/historical_benchmarks/
   baselines/2026-09-25-e29/`). CHECK PASS = the cluster serves the exact measured
   identity.

Revision provenance: the weights snapshot carries `.glm53-fp8-synced` (the pinned
model rev, written after manifest verification) and the drafter snapshot carries
`.cache/huggingface/download/config.json.metadata` — `verify-node.sh` and
`check-f0.py` read both; plain `hf download <repo> --revision <rev>` on any node
re-verifies the cache in seconds without downloading.


## Everyday commands (from the workstation checkout)

```sh
cd <repo>/glm-v53-flash-4x-noswitch/upstream   # <repo> = the spark-recipes checkout (WSL: /mnt/c/.../spark-recipes)

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
- **Never hand-edit generated files**: per-node netplan/iptables (`upstream/scripts/node/etc/*`,
  rendered by `render-netplan.sh`) and `~/.local/tp4/*` on the nodes (pushed by
  `deploy.sh`). Edit `upstream/cluster.env` (mirrored in `noswitch-prep/cluster.env`), then re-render/deploy.
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
  suspect (see build-record.md §7b).
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

References: [build-record.md](noswitch-prep/build-record.md) (build log + deviations) · upstream
[operations.md](https://github.com/jnardiello/GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless/blob/main/docs/operations.md)
· [fabric.md](https://github.com/jnardiello/GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless/blob/main/docs/fabric.md)
· forum thread
[382459](https://forums.developer.nvidia.com/t/glm-5-3-flash-on-tp4-dgx-sparks-switchless/382459)
