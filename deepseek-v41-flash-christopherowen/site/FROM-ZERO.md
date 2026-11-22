# FROM-ZERO — rebuild this lane on four wiped DGX Sparks

Everything here runs from **this recipe's directory alone**. No sibling recipe is read, and
none needs to exist. Routine start/stop is `./bin/start.sh` / `./bin/stop.sh` — this runbook
is for a rebuild.

> **Blast radius.** **[REBOOT]** = needs a reboot. **[DISRUPTIVE]** = stops a serving lane or
> restarts a host service. Do not run this with another lane serving: every lane uses all
> reserved GPUs and binds `:8000`.

## 0. What you need

| | |
|---|---|
| nodes | four DGX Spark (GB10, 128 GB unified), DGX OS 26.09.2 or later — as in `site/fabric/cluster.env` (`NODES`) |
| network | four QSFP56 cables, switchless ring rank0→1→2→3→0 |
| mgmt | `wlP9s9` (10.69.43.0/16) reachable from the head; passwordless SSH head→workers as one user with passwordless `sudo` |
| disk | ≈480 GiB per node (weights + Engram tables read from disk + caches); the **build host** also needs 64 GiB free memory |
| kernel | **stock 4 KiB pages** (`6.17.0-1032-nvidia`). This lane does *not* need the 64 KiB kernel or the memory saver (that is upstream's qualified `tp4-ring`; we run the 4 KiB `tp4-ring-4k`) |

## 1. Host baseline — per node

Upstream's narrow management-plane policy (watchdog, SSH protection, bounded boot gate,
headless boot). It never stops a serving container and never reboots on a network outage.

```bash
cd upstream
scripts/host-recovery check      # read-only; prints what differs
scripts/host-recovery apply      # [DISRUPTIVE] reboots only on a real system stall afterwards
```

The `splash` change takes effect at the next reboot: **[REBOOT]**.

**Do not** apply upstream's "management port off wireless" recommendation — this site's
management network *is* the wireless `wlP9s9`. `sudo nmcli radio wifi off` would cut the
cluster off. Likewise skip the 64 KiB kernel, the memory saver and the modeset migration:
they are a host baseline shared by all four lanes, deliberately not adopted (see
`../../AGENTS.md`).

## 2. Fabric — [DISRUPTIVE]

This recipe carries its own copy of the shared fabric facts and the committed netplan under
`site/fabric/` (identical contract to the other three lanes).

```bash
cd site/fabric
bash scripts/render-netplan.sh --check           # committed == rendered
bash scripts/deploy-host.sh --check --etc        # read-only audit of /etc

sudo cp netplan/<node>/40-cx7.yaml /etc/netplan/40-cx7.yaml
sudo cp netplan/<node>/tp4-fabric-iptables.env /etc/default/tp4-fabric-iptables
sudo netplan try                                 # 120 s auto-rollback
sudo systemctl daemon-reload && sudo systemctl enable --now tp4-fabric-iptables
```

All four PCIe views are addressed; **this lane advertises two** (`rocep1s0f0`, `rocep1s0f1`)
because a ring has one spare rail per neighbour and sparknet refuses four PFs on a ring.
Then:

```bash
bash preflight.sh                                # host contract; prints CONTRACT_DIGEST
../../site/gate-gid-slot.sh                      # exactly one IPv4 RoCE v2 GID per routed PF
```

The NCCL library this lane uses is **inside its image** (`local.spark-ds41f.nccl.tree`).
Nothing is installed at `~/.local/lib/nccl-patched*` for this lane; the host four-PF library
remains for the other lanes and `preflight.sh` still verifies it as host state.

## 3. Site configuration and weights

```bash
cd upstream
python3 bin/spark doctor                         # configuration OK
```

The model is the already-present HF cache snapshot
`models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e10…` mounted read-only at
`/hub`; there is **no materialization step** in this lane (unlike the SGLang lane's hardlink
tree). Verify per node:

```bash
ls ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e10*/ | wc -l
```

## 4. Image — on the head, with the lane down

```bash
cd upstream
python3 bin/spark build prepare                  # fetches 8 pinned sources, applies 702+14+3+4 patches
python3 bin/spark build check                    # verifies patch heads/trees
free -g | sed -n 2p                             # need >= 64 GiB MemAvailable
python3 bin/spark --cluster-config config/cluster.json build image --apply
python3 bin/spark --cluster-config config/cluster.json build distribute --apply
```

Needs no `spark-ds41f`/`^dsv41-` container running and ≥64 GiB free memory. Distribution
streams the image over the management link (~30 GB ≈ 6 min/node) and re-checks content
identity per node. The result must carry five tree labels equal to
`releases/spark-ds41f-1120/cluster.json`'s `expected_labels` (see `site/lane.env`
`EXPECTED_LABELS`).

## 5. Serve

```bash
cd ../            # the recipe directory
./bin/start.sh --dry-run
./bin/start.sh
```

`cluster start` defragments memory, arms a startup memguard, launches **rank 3 → 2 → 1 → 0**,
waits for `/v1/models`, then swaps to the steady memguard. Expect 4–10 min cold.

## 6. Gates — within minutes of `/v1/models` 200

- one name `deepseek-v41-flash`, `max_model_len` 1,048,576;
- a chat round-trip and a forced tool call with `chat_template_kwargs:{"thinking":true}`;
- emoji/CJK probe → 0 × U+FFFD;
- per rank: 0 `NCCL WARN`; the channel lines; only `rocep1s0f0`/`rocep1s0f1` used;
- the boot log's `GPU KV cache size` / `Maximum concurrency` lines, and `SpecDecoding`
  acceptance > 1;
- one ~993K-token prefill with MemAvailable recorded before/after;
- `bin/start.sh` from another lane **refuses** while this one serves.

## 7. Known-bad signatures

| symptom | cause |
|---|---|
| `cluster start` refuses "no release benchmark qualifies this shape" | expected: `--unqualified-shape` (already in `bin/start.sh` via `site/lane.env`) until our own run is promoted |
| a routed PF has 0 or 2 IPv4 RoCE v2 GIDs | `site/gate-gid-slot.sh` names it; fix the netplan, never tune around it |
| `cluster sync` says "the deployed commit does not carry it" | the `profile_committed()` patch was lost — see `upstream/VENDORED-AT.md` |
| `doctor`: "tracked secret-like path" | the node map was renamed to `*.local.json`; keep `config/nodes-ring4.json` |
| `build prepare`: missing patch | the closure was hand-edited; regenerate it |
| image build refuses | ≥64 GiB free memory and no `spark-ds41f` container |
| NCCL hangs / QP RTR timeout | fabric wiring — `site/fabric/preflight.sh` first |
| the container exists from a previous run | `./bin/stop.sh` (captures logs, removes) |
| boot reads slow | JIT warm-up; never bench cold |

## 8. What is deliberately NOT here

- **No cross-lane coordination** — `./bin/start.sh` refuses while anything else serves
  (`bin/guard.sh`, fail-closed), and `./bin/stop.sh` captures logs then removes this lane's
  containers.
- **No performance baselines** — they live in `research.md` with dates and caveats.
- **No host-baseline migration** (64 KiB kernel, memory saver, modeset, NVMe tuning): shared
  with the other three lanes, out of scope for this recipe.