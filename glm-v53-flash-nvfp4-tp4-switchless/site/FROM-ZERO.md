# FROM-ZERO — rebuild this lane on four wiped DGX Sparks

Everything here runs from **this recipe's directory alone**. No sibling recipe is read, and
none needs to exist. Follow it top to bottom; every step names what it needs and what it
proves.

> **Blast radius.** Steps marked **[REBOOT]** need a node reboot. Steps marked **[DISRUPTIVE]**
> stop a serving lane. Everything else is additive or read-only. This runbook is for a
> **rebuild**, not for a routine switch — a routine start is `./bin/start.sh`.

## 0. What you need before starting

| | |
|---|---|
| nodes | four DGX Spark (GB10), as in `site/fabric/cluster.env` (`NODES`) |
| network | four QSFP56 cables in a switchless ring rank0→1→2→3→0 |
| mgmt | the wireless mgmt LAN (`MGMT_IF`, 10.69.43.0/16) reachable by the head |
| login | SSH from the head to every worker as the same user, passwordless |
| disk | weights are **local to every node** (≈tens of GiB for the checkpoint + drafter) |
| privileges | passwordless sudo on all four (some steps use `sudo -n`) |

## 1. OS / driver / docker baseline

Ubuntu for GB10 with the NVIDIA driver **already installed** at the pinned version. Nothing
in this recipe pins the OS image.

```bash
# on every node
nvidia-smi --query-gpu=driver_version --format=csv,noheader   # expect: see step 2
docker version --format '{{.Server.Version}}'
nvidia-ctk --version
```

**[DISRUPTIVE]** `nvidia-container-toolkit` install restarts docker.

## 2. Kernel pin + holds — **[REBOOT]**

The pins live in `site/fabric/scripts/node/bootstrap/versions.env` (kernel
`6.17.0-1032-nvidia`, driver `580.173.02`, `rdma-core>=50`, `docker>=29.2.1`,
`nvidia-ctk>=1.20.0`). **The 7.x kernel breaks ring NCCL** — do not upgrade past the pin.

```bash
cd site/fabric
bash scripts/node/host/kernel-align-fresh.sh --help    # read it first: it reboots at the end
bash scripts/node/host/hold-existing.sh --help         # apt-mark hold on the installed pins
```

Re-run `bash preflight.sh` afterwards: it reports `kernel 6.17.0-1032-nvidia OK` and the
driver, and warns if `apt-mark showhold` is empty.

## 3. Host bootstrap — per node

```bash
cd site/fabric
bash scripts/bootstrap-node.sh spark-0f0b --rank 0 --check    # audit only, changes nothing
bash scripts/bootstrap-node.sh spark-0f0b --rank 0 --apply    # packages, sudoers, /etc, ssh-mesh, layout
# repeat for spark-6d14 (rank 1), spark-6d90 (2), spark-6d24 (3)
```

`--phase <list>` restricts the run (`packages,sudoers,etc,ssh-mesh,layout`). The sudoers
phase is the only one that can need an interactive password, and it runs first for exactly
that reason. **`-n`-less sudo on a node** will make every later `--check` fail with
"needs sudo -n".

## 4. Netplan: render, verify, install

The committed expectation is `site/netplan/<node>/` (both files per node) — reachable from
`site/fabric/` as `../netplan/<node>/`. Never hand-edit
them — they are generated from `site/fabric/cluster.env`.

```bash
cd site/fabric
bash scripts/render-netplan.sh --check        # committed == rendered from the facts
bash scripts/deploy-host.sh --check --etc   # what would change on each node (read-only)
```

Install on every node (**[DISRUPTIVE]** on a serving host):

```bash
sudo cp ../netplan/<node>/40-cx7.yaml /etc/netplan/40-cx7.yaml
sudo cp ../netplan/<node>/tp4-fabric-iptables.env /etc/default/tp4-fabric-iptables
sudo netplan try          # 120 s auto-rollback; confirm all four addresses come up
```

The each-cable `/24` carries an address on **both PCIe views** (`enp1s0f0np0` =
`<L>.<N>` and `enP2p1s0f0np0` = `<L>.<N+2>`). The four-PF NCCL listener requires the full
cross product; a 2-address plan will silently break it.

## 5. Fabric iptables + sysctl + IOMMU

```bash
cd site/fabric
bash scripts/deploy-host.sh --etc            # installs the /etc set + netplan, sha-verified, never activating
sudo systemctl daemon-reload && sudo systemctl enable --now tp4-fabric-iptables
```

The grub IOMMU drop-in (`node/etc/default/grub.d/zz-tp4-perf.cfg`) takes effect on the next
boot — **[REBOOT]**. `bash scripts/node/host/tp4-iommu.sh --apply` handles it, and writes a
sentinel so `deploy-host.sh` stops re-pushing until you `--revert`.

## 6. NCCL: build the four-PF library, stage it, prove its identity

The build tree is vendored: `site/nccl-4pf/` (6 files, patch SHAs asserted at `:44-50`).
Run it **on a node** — it needs docker and a pinned NCCL source clone.

```bash
# on each node (or build once and copy the .so; the SHA is the contract)
scp -r site/nccl-4pf <node>:~/
ssh <node> 'DOCKER="sudo -n docker" ~/nccl-4pf/build-4pf.sh'
# stage the result where the launcher expects it
~/.local/lib/nccl-patched-4pf/libnccl.so.2.30.7
~/.local/lib/nccl-patched-4pf/libnccl.so.2 -> libnccl.so.2.30.7
```

Identity check (this is the one that matters — a marker grep is not an identity check):

```bash
python3 site/fabric/check_switchless_nccl.py \
  --library ~/.local/lib/nccl-patched-4pf/libnccl.so.2.30.7 \
  --sha256 8de68def30bcde20fabded468bd60f0df82ea12d3ead02d5120c7feda7a41a87
```

**Rollback** (only if a boot stalls in NCCL init): this recipe vendors its own 2-PF tree at
`site/nccl-2pf/`. Build it, stage to `~/.local/lib/nccl-patched` (`afe5f486…`), point
`NCCL_HOST_DIR` there in `site/env.switchless`, and re-apply the staged-transport mods in the
`upstream/scripts/transport.sh` comment. Keep both directories; the two libs hard-reject each
other's config.

## 7. Image

```bash
cd upstream
docker build -f Dockerfile.roce -t glm53-roce:v11-b58f34ea .
```

Minutes, no CUDA compile. Base/digest pins: upstream `docs/install.md`. Every node builds the
same tag (`IMAGE` in `site/env.switchless`).

## 8. Weights — per node, zero modification

```bash
bash ../site/materialize-weights.sh   # from upstream/, or `bash site/materialize-weights.sh` from the recipe root
```

Hardlinks the HF-cache snapshots into `~/.local/nvfp4-tp4/weights/{nvidia-glm53-nvfp4,glm53-dflash2}`
so bind-mounts see real files (HF snapshots use relative symlinks into `blobs/`, which do not
resolve inside a container mount). **Zero weight bytes change** — hardlinks, not copies. The
pinned nvidia snapshot ships `chat_template.jinja`; if a future pin stops shipping it, pass the
template as the 5th argument.

Checkpoints: `nvidia/GLM-5.3-Flash-NVFP4` @ `da920bb`, drafter `incoai/GLM-5.3-Flash-DFlash2`
@ `bf582e4e` (**CC BY-NC-ND 4.0** — no derivative re-encodes).

## 9. Serve

`/health` on the head is the readiness signal; boot takes ~7 min cold (JIT caches cold), ~3–4 min
warm. **Never `docker start` a preserved container after a power-cut** — serve fresh.

## 10. Gates — within minutes of `/health` 200

- `/v1/models` advertises `glm-5.3-flash` with `max_model_len` 1,048,576.
- Corruption probe: emoji + CJK generation, count U+FFFD → **must be 0**.
- Forced tool call round-trips.
- Boot log per rank: `Connected all rings` and RouteFinal naming **all four** PFs
  (`rocep1s0f0/f1` + `roceP2p1s0f0/f1`); **0 NCCL WARNs**.
- `bash site/fabric/preflight.sh` → `ALL CHECKS PASS`.

Routine starts do **not** re-run performance benches; benches belong to config/weight-change
sessions.

## 11. Known-bad signatures

| symptom | cause |
|---|---|
| `wrong listener count` at `ncclCommInitRank` | `NCCL_IB_EXTENDED_IPV4_GIDS=1` missing |
| QP RTR timeout / init hang | fabric wiring — run `preflight.sh`, never restart blind |
| `NV_ERR_NO_MEMORY` on the head | GPU memory util too high |
| boot reads ~30% slow | JIT/TRITON warm-up — wait ~30 min, never bench cold |
| `DEVSELECT_BUILD_FAILED` ×1 per rank at boot | expected on this driver stack (stream-captured conditional-node graphs); zero action |

## 12. What is deliberately NOT here

- **No cross-lane coordination.** Only one lane can serve at a time; `./bin/start.sh` refuses
  when anything else is running (`bin/guard.sh`, fail-closed) and `./bin/stop.sh` captures logs
  then removes this lane's containers.
- **No performance baselines.** They live in `research.md` with their dates and caveats.
- **No reimage drill claim.** Portability is validated by inspection, `./bin/start.sh --dry-run`,
  and a one-node NCCL rebuild drill — not by a full wipe.