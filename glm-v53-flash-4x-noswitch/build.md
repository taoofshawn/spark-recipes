# glm-v53-flash-4x-noswitch — build plan (2-node crossover → 4-node switchless ring)

Brings the cluster from its current state (2 sparks with a CX-7 crossover cable, DeepSeek
2-node recipes) to running the GLM-5.3-Flash FP8 recipe **as-is** on a 4-node switchless
ring.

Source: NVIDIA forum thread
[382459 — GLM 5.3 Flash on TP4 DGX Sparks (switchless)](https://forums.developer.nvidia.com/t/glm-5-3-flash-on-tp4-dgx-sparks-switchless/382459)
→ recipe repo [jnardiello/GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless](https://github.com/jnardiello/GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless)
(main @ tree `3479b25`, fetched 2026-09-24; includes the Sep 23 E22b update).

This is intentionally **not** a spark-recipes recipe adoption: we use the upstream repo's own
infrastructure-as-code (`agent-preflight.sh`, `render-netplan.sh`, `bootstrap-node.sh`,
`tp4ctl`) verbatim. `build.md` here is the site-customized runbook for *our* four nodes.

---

## 1. Current state (verified by SSH on 2026-09-24)

| node | role now | mgmt IP (`enP7s7`) | CX-7 ports | kernel |
|---|---|---|---|---|
| `spark-0f0b` | leader, rank 0, DeepSeek API server | `10.69.42.170` | **one** crossover cable → 6d14, plugged in the **left** QSFP port: `enp1s0f0np0` UP (192.168.0.170) + `enP2p1s0f0np0` UP (192.168.1.170) = the port's two PCIe views; `enp1s0f1np1` DOWN, `enP2p1s0f1np1` DOWN (right port, uncabled) | `6.17.0-1032-nvidia` |
| `spark-6d14` | worker, rank 1 | `10.69.42.171` | same layout, 192.168.0.171 / 192.168.1.171 | `6.17.0-1032-nvidia` |
| `spark-6d90` | **not up yet** — fresh install | expected `10.69.42.172` (confirm) | unknown | **7.x** → Phase 1b downgrade |
| `spark-6d24` | **not up yet** — fresh install | expected `10.69.42.173` (confirm) | unknown | **7.x** → Phase 1b downgrade |

Key facts established:

- **Port map (per the [NVIDIA DGX Spark clustering doc](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html)):**
  each Spark has **two QSFP ports** (left = closest to the ethernet port, right), each
  capped at 200 Gb/s and appearing as **two** Linux interfaces — one per PCIe Gen5 x4 link
  from the NIC into the SoC: left port = `enp1s0f0np0`/`rocep1s0f0` + `enP2p1s0f0np0`/
  `roceP2p1s0f0`; right port = `enp1s0f1np1`/`rocep1s0f1` + `enP2p1s0f1np1`/`roceP2p1s0f1`.
- The **existing single crossover** is in the left port on both nodes: that is why
  `enp1s0f0np0` (192.168.0.x) and `enP2p1s0f0np0` (192.168.1.x) are both UP to the same
  peer. The right port is uncabled (both `f1` netdevs Down). The current 2-node DeepSeek
  recipes use **both** PCIe views of the left port (`IB_PORTS=rocep1s0f0,roceP2p1s0f0`);
  the 4-node recipe uses **one view per port** and leaves the P2 views UP/MTU-9000 but
  unaddressed.
- **No template deviation:** the DGX Spark port map matches the recipe's verified GX10
  convention exactly — addressed ports are one PCIe view per QSFP port
  (`enp1s0f0np0` + `enp1s0f1np1` → `NCCL_IB_HCA=rocep1s0f0,rocep1s0f1`), and the second
  PCIe views are the excluded duplicates. `FABRIC_IFACES`/`NCCL_IB_HCA` keep the template
  values (§5). `enp1s0f1np1` only comes Up once the right port is cabled.
  `agent-preflight.sh` proposes values from sysfs discovery and blocks on ambiguity — it
  remains the authority.
- Existing nodes run kernel `6.17.0-1032-nvidia` — the **known-good** kernel. The newer
  `7.0.0-1019-nvidia` breaks switchless-ring NCCL/RoCE (`ibv_reg_mr_iova2` ENOMEM,
  forum thread 383023). Fresh installs ship 7.x kernels → Phase 1b downgrade required.
- `spark-0f0b`: 1.9 TB free on `/`, repo clone at `~/code/spark-recipes` (2-node DeepSeek
  recipes — stays for rollback, untouched by this build). No containers running at the
  time of the inventory.

## 2. Final state

Four GB10 nodes in a closed switchless ConnectX-7 ring — **4 DACs, one per QSFP port**
(both ports on every node, MTU 9000), running the upstream E22b recipe as-is:

- GLM-5.3-Flash FP8 (`zai-org/GLM-5.3-Flash` @ `690b7052`), vLLM TP4, 262,144-token context,
  DFlash2 speculative decoding, SparkCache + SIRCL transport, patched NCCL 2.30.7
  (tree-skip overlay, `NCCL_ALGO=Ring`).
- API on rank 0, port 8000, **no auth/TLS** — trusted LAN only.
- Measured upstream: 56.6 tok/s code decode C1, 41.6 C1 E2E, 95.9 C4 aggregate,
  ~2,600 tok/s cold prefill, TTFT ~0.4 s. **Concurrency limit ≈ 5 sessions** (6 dies at
  262K ctx — memory is tight, jacopo in thread 382459 post 7).

Ring and addressing (upstream-recommended `10.10.<link>.<1-based-node>` plan, unchanged):

| link | ends | addresses |
|---|---|---|
| L1 (rank0↔rank1) | spark-0f0b ↔ spark-6d14 | 10.10.1.1 ↔ 10.10.1.2 |
| L2 (rank1↔rank2) | spark-6d14 ↔ spark-6d90 | 10.10.2.2 ↔ 10.10.2.3 |
| L3 (rank2↔rank3) | spark-6d90 ↔ spark-6d24 | 10.10.3.3 ↔ 10.10.3.4 |
| L4 (rank3↔rank0) | spark-6d24 ↔ spark-0f0b | 10.10.4.4 ↔ 10.10.4.1 |

The old 192.168.0.x / 192.168.1.x crossover subnets are **replaced** by the generated
netplan (`40-cx7.yaml`) — they do not survive this build (restorable by reverting netplan).

## 3. Cabling map (physical)

Port convention (identical on every node, per the NVIDIA doc): **left port** = the QSFP
port closest to the ethernet port (`enp1s0f0np0`/`enP2p1s0f0np0` when cabled), **right
port** = `enp1s0f1np1`/`enP2p1s0f1np1`. Each ring link uses one cable into one QSFP port;
the recipe addresses one PCIe view per port (`enp1s0f0np0` = left, `enp1s0f1np1` = right)
and leaves the `P2` views UP with MTU 9000 but unaddressed.

Cable pull-tab faces up, insert smoothly without force. Approved cables: Amphenol
NJAAKK-N911 (0.4 m; NJAAKK0006 = 0.5 m) or Luxshare LMTQF022-SD-R, ≥200 Gb/s,
Ethernet-only config.

| step | action | cable |
|---|---|---|
| 1 | spark-0f0b **left** port ↔ spark-6d14 **left** port — **already cabled (the current crossover); leave in place** → this is L1 | existing |
| 2 | spark-6d14 **right** port ↔ spark-6d90 **left** port → L2 | new DAC |
| 3 | spark-6d90 **right** port ↔ spark-6d24 **left** port → L3 | new DAC |
| 4 | spark-6d24 **right** port ↔ spark-0f0b **right** port → L4 | new DAC |

Every node ends with exactly one cable per QSFP port: left port faces one ring neighbor,
right port the other (L1 stays on the left ports of both 0f0b and 6d14, which is where the
existing crossover already sits). This matches the recipe's ring convention — the left
(addressed `enp1s0f0np0`) port faces the node's first-listed `FABRIC_TARGETS` peer, the
right (`enp1s0f1np1`) port the second.

All four DACs must link at their rated speed, MTU 9000 (jumbo pings in Phase 9 verify all
eight directed edges). **Carrier state alone cannot identify the peer** — confirm each edge
with cable EEPROM serials read at both ends:

```sh
ssh <node> 'sudo -n ethtool -m <fabric-iface> | grep "Vendor SN"'
```

Both ends of one cable must report the same serial. On the two fresh nodes, identify left
vs right the same way (left = closest to the ethernet port; the cabled port's two netdevs
come Up while the other pair stays Down). After cabling, `ip -br link` must show exactly
two netdev pairs Up (`enp1s0f0np0`+`enP2p1s0f0np0`, `enp1s0f1np1`+`enP2p1s0f1np1`) on
every node. Stop before addressing if any edge is ambiguous.

## 4. Prerequisites

Workstation: `bash`, `ssh`, `scp`, `rsync`, `shasum`, `curl`, Python 3.9+ (plus
`Jinja2==3.1.6` for `./scripts/check.sh`). Nodes: Ubuntu + NVIDIA driver + Docker with GPU
support + `rdma-core` (DGX OS fresh installs ship all four; `bootstrap-node.sh --check`
verifies), ≥330 GiB free per node, `bash`/`ssh`-reachable over the mgmt LAN.

SSH aliases for the two new nodes must exist in the workstation ssh config
(`spark-6d90.shawndo.intra`, `spark-6d24.shawndo.intra`) and each node's `hostname -s`
must equal its cluster alias (`spark-6d90`, `spark-6d24`) — `deploy-host.sh` refuses a
hostname/alias mismatch.

**Kernel checkpoint (fresh nodes):** fresh installs ship **7.x** kernels — they must be
downgraded to `6.17.0-1032-nvidia` first (Phase 1b), because `7.0.0-1019-nvidia` breaks
switchless-ring NCCL/RoCE (thread 383023: `ibv_reg_mr_iova2` ENOMEM → NCCL/RoCE dead on
TP4 rings).

**Operator-supplied payloads (hard prereq for the current recipe):** the SparkCache
connector/encoder and the SIRCL bundle/runtime are **not redistributed** by the repo.
They must be provided to `~/tp4/sparkcache/` and `~/tp4/sircl/` per
`docs/install-from-zero.md` §8 (Phase 8). DFlash2 drafter (`incoai/GLM-5.3-Flash-DFlash2`,
2.34 GB) is **CC BY-NC-ND 4.0 — non-commercial**.

## 5. `cluster.env` — site values (everything else stays upstream as-is)

Copy `cluster.env.example` → `cluster.env` in the recipe checkout, then set **only** these:

```bash
# Topology — aliases equal hostname -s; ssh targets resolve via *.shawndo.intra
NODES="spark-0f0b spark-6d14 spark-6d90 spark-6d24"
MGMT_IPS="10.69.42.170 10.69.42.171 10.69.42.172 10.69.42.173"   # confirm .172/.173 when nodes are up
MASTER_IP="10.69.42.170"

# Management interface — already matches the template value verified on our nodes
MGMT_IF=enP7s7

# Fabric ring — keep the recommended 10.10.<L>.<N> plan unchanged:
# rank0: 10.10.1.1 + 10.10.4.1, rank1: 10.10.1.2 + 10.10.2.2,
# rank2: 10.10.2.3 + 10.10.3.3, rank3: 10.10.3.4 + 10.10.4.4
FABRIC_TARGETS=(
  "10.10.1.2 10.10.4.4"
  "10.10.1.1 10.10.2.3"
  "10.10.2.2 10.10.3.4"
  "10.10.3.3 10.10.4.1"
)

# Fabric interfaces — the DGX Spark map matches the template values as-is:
# addressed = one PCIe view per QSFP port (left enp1s0f0np0, right enp1s0f1np1),
# excluded = the second PCIe views (enP2p1s0f0np0, enP2p1s0f1np1)
FABRIC_IFACES="enp1s0f0np0 enp1s0f1np1 enP2p1s0f0np0 enP2p1s0f1np1"

# HCA — the two addressed RoCE devices, one per QSFP port
NCCL_IB_HCA="rocep1s0f0,rocep1s0f1"
NCCL_IB_GID_INDEX=-1

NETPLAN_RENDERER=NetworkManager
RELAY_DEST='sdrew@10.10.2.3'   # rank 2 (spark-6d90) via the direct rank1→rank2 link
```

Login user on the nodes is `sdrew` (confirm identical on both fresh nodes — the bootstrap
flow refuses mixed users). Everything else in the file (IMAGE digest pin, model/draft
revisions, KV pool, E21/E22b flags, `EXTRA_DOCKER_ENV`) **stays exactly as upstream** —
that is the "recipe as-is" part.

## 6. Phases

### Phase 0 — prepare current nodes

1. Ensure no DeepSeek/2-node model containers run:
   `ssh spark-0f0b.shawndo.intra 'docker ps'` and same on 6d14 (sudo password required on
   the nodes — run interactively, not with `BatchMode`).
2. Leave the existing crossover cable in place (it becomes L1) until Phase 3; the DeepSeek
   recipes stay in `~/code/spark-recipes` as rollback (they will not work after re-cabling
   until the netplan is reverted).

### Phase 1 — rack + boot the new nodes

Rack spark-6d90 and spark-6d24, let the fresh OS come up, confirm: `hostname -s`, mgmt IP
on `enP7s7` (expected .172/.173), `uname -r` (**fresh installs ship 7.x kernels → Phase 1b
downgrade required**), Docker + rdma-core present, SSH keys installed for `sdrew`.
Disk ≥330 GiB free each.

### Phase 1b — downgrade the fresh nodes to the known-good kernel (7.x → 6.17.0-1032-nvidia)

**Why:** the 7.0.0-1019-nvidia kernel breaks switchless-ring NCCL/RoCE (`ibv_reg_mr_iova2`
ENOMEM, forum thread 383023). Both existing nodes run `6.17.0-1032-nvidia` — replicate that
exact set before the fabric or netplan is configured.

Do this **before** Phase 4 (the preflight's RoCE checks fail on a 7.x kernel) and before
Phase 6's `/etc` phase (netplan/GRUB activation must happen on the good kernel). One node
at a time, node fully idle.

Reference package set on spark-0f0b (version `6.17.0-1032.32`, `nvidia-hwe-24.04` flavour):

- `linux-image-6.17.0-1032-nvidia`, `linux-modules-6.17.0-1032-nvidia`
- `linux-modules-nvidia-580-open-6.17.0-1032-nvidia`, `linux-modules-nvidia-fs-6.17.0-1032-nvidia`
- `linux-headers-6.17.0-1032-nvidia`, `linux-tools-6.17.0-1032-nvidia`
- metas: `linux-nvidia-hwe-24.04`, `linux-image-nvidia-hwe-24.04`,
  `linux-headers-nvidia-hwe-24.04`, `linux-modules-nvidia-580-open-nvidia-hwe-24.04`,
  `linux-modules-nvidia-fs-nvidia-hwe-24.04`, `linux-tools-nvidia-hwe-24.04`

On each fresh node (spark-6d90 first, then spark-6d24):

```sh
uname -r                                        # expect 7.* → downgrade required
dpkg -l | grep -E 'linux-(image|modules|headers)-.*7\.'   # enumerate the 7.x set to remove
sudo apt update

# 1. install the known-good 6.17.0-1032 set
sudo apt install -y \
  linux-image-6.17.0-1032-nvidia linux-modules-6.17.0-1032-nvidia \
  linux-modules-nvidia-580-open-6.17.0-1032-nvidia linux-modules-nvidia-fs-6.17.0-1032-nvidia \
  linux-headers-6.17.0-1032-nvidia linux-tools-6.17.0-1032-nvidia

# 2. point the hwe-24.04 metas at 6.17 so they stop pulling 7.x
sudo apt install -y \
  linux-nvidia-hwe-24.04=6.17.0-1032.32 \
  linux-image-nvidia-hwe-24.04=6.17.0-1032.32 \
  linux-headers-nvidia-hwe-24.04=6.17.0-1032.32

# 3. remove the 7.x kernel set so GRUB boots 6.17 by default
sudo apt purge -y <7.x packages enumerated above>

# 4. hold everything so an update cannot reintroduce 7.x until NVIDIA fixes
#    the regression (watch thread 383023; lift the hold only then)
sudo apt-mark hold linux-nvidia-hwe-24.04 linux-image-nvidia-hwe-24.04 \
  linux-headers-nvidia-hwe-24.04 linux-modules-nvidia-580-open-nvidia-hwe-24.04 \
  linux-modules-nvidia-fs-nvidia-hwe-24.04

# 5. driver-userspace checkpoint: must match the 580-open kernel modules
nvidia-smi   # compare the driver version against spark-0f0b (580-series)
# if the fresh install ships a newer userspace driver, align it with spark-0f0b's
# nvidia-driver*/nvidia-utils* versions before rebooting — a mismatch breaks CUDA

# 6. reboot and verify
sudo reboot
```

After the reboot, verify on the node: `uname -r` = `6.17.0-1032-nvidia`, `nvidia-smi`
works, `ibdev2netdev` shows the four CX-7 devices (two Up once the ring is cabled in
Phase 3), and `ibv_devinfo` reports the HCAs. If the node fails to boot the 6.17 kernel,
select it from the GRUB menu before assuming the purge failed.

Expected: both fresh nodes boot `6.17.0-1032-nvidia` with a working `nvidia-smi` and
visible CX-7 devices, packages held. Stop on any failure — do not continue to the preflight
on a 7.x kernel.

### Phase 2 — get the recipe checkout (workstation)

```sh
git clone https://github.com/jnardiello/GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless.git \
  ~/code/glm-4x-noswitch
cd ~/code/glm-4x-noswitch
pip install 'Jinja2==3.1.6' && ./scripts/check.sh   # offline check, no GPU needed
```

### Phase 3 — cable the ring (§3 table)

Add the three new DACs per the table (the existing crossover stays as L1). Verify serials
at both ends of every edge before addressing.

### Phase 4 — read-only preflight

```sh
TP4_HOSTS='sdrew@spark-0f0b sdrew@spark-6d14 sdrew@spark-6d90 sdrew@spark-6d24' \
  ./scripts/agent-preflight.sh --report /tmp/tp4-preflight.json
```

Expected: `result: ready`, one GB10 + two active RDMA ports per node, proposed
HCA/GID/render values match §5, no foreign workload. **Verify the proposed
`FABRIC_IFACES`/`NCCL_IB_HCA` are the template map (`enp1s0f0np0 enp1s0f1np1
enP2p1s0f0np0 enP2p1s0f1np1`, `rocep1s0f0,rocep1s0f1`) — anything else, stop and
reconcile.**
Report stays at `/tmp`, mode 0600, outside the checkout.

### Phase 5 — site config

Fill `cluster.env` per §5, then:

```sh
./scripts/render-netplan.sh --write
./scripts/render-netplan.sh --check      # all eight generated files match
./scripts/deploy-host.sh --check         # shows the intended network drift (old crossover netplan replaced)
```

**Cable-map checkpoint:** before applying, read the generated
`scripts/node/etc/<alias>/40-cx7.yaml` for each rank and confirm the port↔IP mapping
matches the §3 cable map (left port = first-listed `FABRIC_TARGETS` peer, right port =
second). If a node's mapping is flipped relative to its physical cables, re-cable or stop
and reconcile — do not hand-edit the generated file.

### Phase 6 — bootstrap hosts

```sh
. ./cluster.env
r=0
for n in $NODES; do
  ./scripts/bootstrap-node.sh "$n" --rank "$r" --check
  r=$((r + 1))
done
```

Fresh nodes will show `TODO`/`FAIL` (sudoers, packages, `/etc`, ssh-mesh, layout,
autostart). Then, per node (sudoers first — it may prompt for the account password; all
later phases use `sudo -n`):

```sh
./scripts/bootstrap-node.sh spark-6d90 --rank 2 --apply --phase sudoers
./scripts/bootstrap-node.sh spark-6d90 --rank 2 --apply --phase packages,etc,ssh-mesh,layout,autostart
```

(the `/etc` phase installs netplan, sysctl, fabric iptables and the GRUB drop-in; netplan
activation can bounce fabric links / drop SSH). Repeat `--check` until no `TODO`/`FAIL`.
Rank 0 additionally needs passphrase-less SSH to all four mgmt addresses (ssh-mesh phase).
Reboot rolling (rank3→rank2→rank1→rank0, wait for each) if kernel/GRUB changed.

### Phase 7 — build + install patched NCCL

```sh
scripts/node/nccl/build.sh --dry-run
scripts/node/nccl/build.sh --host spark-6d14 --dest '$HOME/nccl-build-repro' --jobs 20
scripts/node/nccl/install-nccl.sh \
  --from 'spark-6d14:$HOME/nccl-build-repro/nccl/build/lib/libnccl.so.2' \
  --expect-sha <sha-printed-by-build>
```

Every node must end with the same SHA-256 at `$NCCL_DIR/libnccl.so.2`
(`~/nccl-patched/libnccl.so.2`). Builds are not bit-reproducible — adopt the new checksum
via the candidate flow in `scripts/node/nccl/README.md`.

### Phase 8 — artifacts

1. **Image** on all four nodes: `sudo -n docker pull $IMAGE` → every node must print
   exactly `IMAGE_ID` (`ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:0d40…`,
   content id `5e32aaa1…`).
2. **FP8 weights** (~306 GiB, 62 shards, `zai-org/GLM-5.3-Flash` @ `690b7052`): install
   `huggingface_hub[cli]` in `~/.hfenv` on rank 0, `./scripts/deploy.sh`, then
   `fetch-fp8-weights.sh` (dry-run first; use `XFER_HOSTS` + `RELAY_RANK2=1` to fan out
   over the ring with rank 2 relayed through rank 1).
3. **DFlash2 drafter**: `hf download incoai/GLM-5.3-Flash-DFlash2 --revision $DRAFT_REV
   --local-dir ~/glm53-dflash2-draft` on every node.
4. **SparkCache + SIRCL payload** (operator-supplied, §4): prepare with
   `scripts/prepare-sparkcache.py` + the replay-views `prepare.py`, place under
   `~/tp4/sparkcache/` and `~/tp4/sircl/` on **every** rank per `docs/install-from-zero.md`
   §8. Record the per-rank SIRCL hashes once in `scripts/node/sircl/SHA256SUMS.site`.
   **If these payloads cannot be supplied, stop and report — the current recipe will not
   start without them** (`SPARKCACHE_MODE=on` verifies hashes and the launcher refuses to
   start a rank whose payload does not match).

### Phase 9 — deploy + verify + start

```sh
./scripts/deploy.sh
./scripts/deploy-host.sh
./scripts/deploy-host.sh --run tp4-iommu.sh --apply      # GRUB update; rolling reboot to activate
./scripts/verify-node.sh --full-model
./scripts/tp4ctl fabric-check     # 2 addressed MTU-9000 ports/rank + 8 successful jumbo pings
./scripts/tp4ctl status
./scripts/tp4ctl up               # only if autostart hasn't already loaded
./scripts/verify-node.sh --live
./scripts/tp4ctl health
```

Expected: static verify passes; `/health` → 200; all runtime signatures present
(mHC flag, replay connector pin, draft-budget scheduler marker, `E21_BF16_RESIDUE_W8A16_READY`
(67 modules) and `E22_DRAFTER_W8A16_READY` (30 modules) on every rank). Run the post-boot
functional gates (`docs/operations.md`) within two minutes of readiness, then
`./scripts/check-f0.py` for the accepted identity.

Smoke test from a trusted-LAN client:

```sh
curl http://10.69.42.170:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","temperature":0,"max_tokens":64,
       "chat_template_kwargs":{"enable_thinking":false},
       "messages":[{"role":"user","content":"Reply with READY."}]}'
```

## 7. Standing invariants & gotchas

- **Never hand-edit** generated netplan/iptables files — they derive from `cluster.env`
  via `render-netplan.sh`.
- Start/stop/recover only via `scripts/tp4ctl`; never repair one serving rank in isolation.
- Warm-up: fresh boot is slower until a few hundred tokens of traffic pass — don't benchmark
  immediately after boot. Bench with `stream:false` and `usage.completion_tokens`.
- No `sudo -n` available until the sudoers phase lands; node sudo needs a password over
  interactive SSH before that.
- Rollback of the fabric = revert the generated netplan on the nodes; rollback of the
  recipe = `TP4_ENV=scripts/node/reference/baseline-20260923-e21.env` (E21, no E22b).
- The old 2-node DeepSeek recipes remain usable only if the crossover netplan is restored.
- `gpu_memory_utilization=0.85` and the 15 GiB KV pool are measured values — do not tune.

## 8. Source references

- Thread: https://forums.developer.nvidia.com/t/glm-5-3-flash-on-tp4-dgx-sparks-switchless/382459
  (E22b update announced in post 10, 2026-09-23; concurrency-5 answer in post 7)
- Recipe repo: https://github.com/jnardiello/GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless
  (`docs/install-from-zero.md`, `docs/fabric.md`, `docs/operations.md`,
  `cluster.env.example`, `scripts/node/nccl/README.md`)
- Kernel regression: https://forums.developer.nvidia.com/t/dgx-spark-regression-kernel-7-0-0-1019-nvidia-causes-nccl-roce-ibv-reg-mr-iova2-enomem-6-17-0-1032-works/383023
- Port mapping / cabling: https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html
  (left port = closest to ethernet port; each QSFP port appears as two interfaces, one per
  PCIe Gen5 x4 link; approved cables Amphenol NJAAKK-N911 / Luxshare LMTQF022-SD-R)
- Local context: `glm-4spark.md` (4x topology research, untracked)
