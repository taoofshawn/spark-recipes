# FROM-ZERO — rebuild this lane on four wiped DGX Sparks

Everything here runs from **this recipe's directory alone**. No sibling recipe is read, and
none needs to exist.

> **Blast radius.** **[REBOOT]** = needs a reboot. **[DISRUPTIVE]** = stops a serving lane.
> This runbook is for a **rebuild**; a routine start is `./bin/start.sh`.

## 0. What you need

| | |
|---|---|
| nodes | four DGX Spark (GB10), as in `site/fabric/cluster.env` (`NODES`) |
| network | four QSFP56 cables, switchless ring rank0→1→2→3→0 |
| mgmt | wireless mgmt LAN (`MGMT_IF`) reachable by the head |
| login | SSH head→workers as one user, passwordless |
| disk | checkpoint ≈**475 GiB per node** (one local copy per node — a ring has no shared path) + Engram tables |
| privileges | passwordless sudo on all four |

## 1. OS / driver / docker baseline

Driver pre-installed; `nvidia-smi --query-gpu=driver_version`, `docker version`,
`nvidia-ctk --version` on every node. **[DISRUPTIVE]** installing
`nvidia-container-toolkit` restarts docker.

## 2. Kernel pin + holds — **[REBOOT]**

Same pins file as the other switchless recipes:
`site/fabric/scripts/node/bootstrap/versions.env` (kernel `6.17.0-1032-nvidia`, driver
`580.173.02`, `rdma-core>=50`, `docker>=29.2.1`, `nvidia-ctk>=1.20.0`). **The 7.x kernel
breaks ring NCCL.**

```bash
cd site/fabric
bash scripts/node/host/kernel-align-fresh.sh --help
bash scripts/node/host/hold-existing.sh --help
```

## 3. Host bootstrap — per node

```bash
cd site/fabric
bash scripts/bootstrap-node.sh spark-0f0b --rank 0 --check
bash scripts/bootstrap-node.sh spark-0f0b --rank 0 --apply
# repeat ranks 1..3 (spark-6d14, spark-6d90, spark-6d24)
```

## 4. Netplan

Committed expectation: `site/fabric/netplan/<node>/` (both files per node), generated from
`site/fabric/cluster.env`.

```bash
cd site/fabric
bash scripts/render-netplan.sh --check                  # committed == rendered
bash scripts/deploy-host.sh --check --etc               # read-only audit
```

**[DISRUPTIVE]** install on every node:

```bash
sudo cp site/fabric/netplan/<node>/40-cx7.yaml /etc/netplan/40-cx7.yaml
sudo cp site/fabric/netplan/<node>/tp4-fabric-iptables.env /etc/default/tp4-fabric-iptables
sudo netplan try          # 120 s auto-rollback; all four addresses must come up
```

Both PCIe views of every cable are addressed (`<L>.<N>` and `<L>.<N+2>`); the launcher's
ring preflight requires the full four-HCA advertisement that depends on it. This recipe
does **not** use the upstream generic ring template (`41-sparkring-plane2.yaml`,
sparkring-built lib) — that is a different contract.

## 5. Fabric iptables + sysctl + IOMMU

```bash
cd site/fabric
bash scripts/deploy-host.sh --etc
sudo systemctl daemon-reload && sudo systemctl enable --now tp4-fabric-iptables
bash scripts/node/host/tp4-iommu.sh --apply     # grub IOMMU drop-in; takes effect next boot [REBOOT]
```

## 6. NCCL: build the four-PF library and prove its identity

Vendored build tree: `site/fabric/nccl-4pf/`.

```bash
scp -r site/fabric/nccl-4pf <node>:~/
ssh <node> 'DOCKER="sudo -n docker" ~/nccl-4pf/build-4pf.sh'
# stage: ~/.local/lib/nccl-patched-4pf/libnccl.so.2.30.7 (+ libnccl.so.2 symlink)
python3 site/fabric/check_switchless_nccl.py \
  --library ~/.local/lib/nccl-patched-4pf/libnccl.so.2.30.7 \
  --sha256 8de68def30bcde20fabded468bd60f0df82ea12d3ead02d5120c7feda7a41a87
```

`site/env.switchless` carries the same SHA as `SWITCHLESS_NCCL_SHA256`; the launcher also
refuses a stock library by grepping for the `SWITCHLESS_RING_ONLY` marker.

**Rollback** (only if a boot stalls in NCCL init): this lane has **no** 2-PF build tree. Its
rollback is a donor-image extraction, recorded in
`upstream-sparkring-archive/profile-deepseek-v41-flash-cycle/README.md` (pull the pinned
sparkring image, `docker cp` its `libnccl.so.2`). That is self-contained and is **not**
needed for a four-PF boot.

## 7. Image

```bash
bash site/build-cluster.sh
```

Head + workers. It stage-fetches the pinned SGLang canary tree
(`upstream/scripts/fetch-sglang-canary.sh`, deterministic refs) and rsyncs it to each worker
because `runtime/sglang-canary/` is gitignored upstream. The base image is digest-pinned:
`lmsysorg/sglang:dev-dsv41@sha256:3dbc3130…` via `BUILD_ARGS` — the vendored Dockerfile's
`ARG BASE_IMAGE` default is the **mutable** tag, so the digest must ride `BUILD_ARGS` in
every build path.

## 8. Weights + Engram — per node

```bash
bash site/materialize-miaai-weights.sh <node>    # or `local` on the node itself
```

Hardlinks the HF-cache snapshot `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e10`
(≈475 GiB, 48 shards; the DSpark draft layers are **inside** the checkpoint — no separate
drafter) into `~/.local/ds41f-miaai/weights/`. **Zero weight modification.**

Then pack the Engram tables once per checkpoint:

```bash
bash site/start-cluster.sh pack     # idempotent per layer; completed shards are skipped
```

## 9. Serve

```bash
bash site/start-cluster.sh doctor   # connectivity + GPU + weights + ring preflight, all 4 ranks
./bin/start.sh                      # guard, then start-cluster.sh serve
```

`doctor` blocks on real problems, including a **`MemAvailable ≥ 100 GiB`** check — if a rank
fails it after a long uptime, reboot that node (fragmentation), not the lane. ~8 min to
`/health` from local NVMe.

## 10. Gates — within minutes of `/health` 200

- `/v1/models` advertises `deepseek-v41-flash`, `max_model_len` 1,048,576.
- Coherent response + forced tool call (dsv4.1 parser round-trip).
- Corruption probe: emoji/CJK, 0 × U+FFFD.
- Per rank: 0 NCCL WARNs; `SWITCHLESS_RING_ONLY` marker present; the
  `# [encoding_dsv41-placeholder-sanitize]` tag in the container's patched sources; the
  adapter's thinking-floor warning line.
- Cache adapters armed (v2.6.4): `DSV41_SWA_THIN receipt: mode=…`, `DSV41_SWA_CONE` (N 8192),
  swan module tag `00505fe850b7` with no fallback, `decode yield (share 33)`. A `_thin`/`_cone`
  *not installed* line means an engine pin drifted.
- `SpecDecoding`/DSpark acceptance > 1.
- Record `max_total_num_tokens` and the `DSV4 pool sizes:` line (`full=` / `swa=` / `c4=`).
- `bash site/fabric/preflight.sh` → `ALL CHECKS PASS`.

## 11. Re-vendoring upstream (the site-mod discipline)

This tree carries **4 site modifications** that must be re-applied after any upstream
re-vendor — see `upstream/VENDORED-AT.md`:

1. the thinking floor,
2. the `DEFAULT_THINKING` injection,
3. the 41→2 `RUN` collapse in `Dockerfile.canary-roce` (the pinned base exceeds buildkit's
   max depth; upstream's packaging tests assert the standalone form, and the site patch
   accepts both),
4. the `start.sh` env whitelist.

Do not restate them here — `VENDORED-AT.md` is the source of truth, and it records the pin.

## 12. Known-bad signatures

| symptom | cause |
|---|---|
| `wrong listener count` at `ncclCommInitRank` | `NCCL_IB_EXTENDED_IPV4_GIDS=1` missing |
| QP RTR timeout / init hang | fabric wiring — run `preflight.sh`, never restart blind |
| launcher refuses to start, container names exist | leftovers — `./bin/stop.sh` removes them |
| `doctor` fails on `MemAvailable` | reboot that rank |
| boot reads slow | warm-up; never bench cold |
| **stop order** | `start-cluster.sh stop` already handles it (rank 0 first) — a worker starting while an old head still listens hangs on the rendezvous |

## 13. What is deliberately NOT here

- **No cross-lane coordination** — `./bin/start.sh` refuses while anything else serves
  (`bin/guard.sh`, fail-closed); `./bin/stop.sh` captures logs then removes this lane's
  containers.
- **No performance baselines** — they live in `research.md` with dates and caveats.
- **No reimage drill claim** — portability is validated by inspection, `--dry-run`, and the
  one-node NCCL rebuild drill.