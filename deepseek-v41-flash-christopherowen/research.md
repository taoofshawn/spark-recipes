# research.md — `deepseek-v41-flash-christopherowen`

Provenance, decisions, measurements and watch items for the second DeepSeek-V4.1-Flash lane.
`README.md` is the ops doc; this file is the audit trail.

---

## 2026-11-22 — adoption, Phase 1 (vendored recipe, no bring-up)

**Status: recipe present and statically validated. Nothing built, nothing deployed, no lane
touched.** Phase 2 (build → distribute → boot → gates → bench) is written as a separate
handoff for an off-spark model, because it needs the sparks stopped:
**`.research/HANDOFF-ds41f-inrepo-phase2-20261122.md`**, executed off branch
`adopt-ds41f-inrepo`.

### Provenance

| | |
|---|---|
| upstream | `github.com/christopherowen/spark-ds41f` (Apache-2.0) |
| release | **`spark-ds41f-1120`** (annotated tag, 2026-11-20) → commit `d3d40d941f2752fea95aa770eb10d41b5ad5aa1a` |
| forum thread | [NVIDIA DGX Spark, "New Deepseek 4.1 flash recipe"](https://forums.developer.nvidia.com/t/384438) (`t384438`, 204 posts, last activity 2026-11-22) |
| layout | **in-repo**, vendored under `upstream/` — not a fork. Rationale and the four obstacles that forced it: `.research/HANDOFF-ds41f-inrepo-20261122.md` |
| vendored files | 1,426 (closure-only: CLI, build inputs, 786 patches, manifests, docs, sonar corpus, `host/recovery/`) |
| excluded | ~30,800 files: bench receipts under `experiments/**/{runs,receipts,results}`, `tests/`, `state/`, `site/` — none read by build/doctor/cluster/bench |
| byte identity | the 1,426 vendored files were extracted from a `spark-ds41f-1120` checkout and validated to carry **no CR bytes** in any text file; the build closure (730 files: 5 series + 725 patches) is byte-exact, so `git am` sees what upstream shipped |
| weights | `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e10` — the release lock's `v41-flash-1017` snapshot, already on all four nodes; **zero weight modification** |

### Why this lane, given we already serve DeepSeek-V4.1-Flash

The existing lane is SGLang/knapcio with the host four-PF NCCL library. This one is vLLM +
sparknet `oneshot-ring4` with **NCCL inside its own image**, and a promotion contract (release
locks, per-tree image labels, qualified shapes, per-request acceptance receipts). It is an
independent implementation of the same model on the same hardware — useful as a comparison
lane, and the only lane here whose image content is attested by five git-tree labels.

### Decisions

| # | decision | why |
|---|---|---|
| D1 | **Vendor in-repo, do not fork** | `deployment.repository` is already `taoofshawn/spark-recipes` on every node, so `cluster sync`/`git pull` carry the recipe with no second checkout. A fork would need a new repo, a second push target, and a second pull path per node |
| D2 | **Four site patches in the vendored tree, recorded in `upstream/VENDORED-AT.md`** | upstream's tooling assumes its checkout *is* the repo root. The patches are minimal and each one is load-bearing (see VENDORED-AT: secret scan, dirty check, `profile_committed`, tracked `site.json`) |
| D3 | **Node map is `config/nodes-ring4.json`** | `*.local.json` is failed by upstream's secret scan when tracked, and `config/nodes.json` is the 3-node triangle default that `cluster sync` rewrites unconditionally |
| D4 | **2-PF subset per neighbour** | sparknet rejects four PFs on a ring (`a neighbour ring has no spare rail`). This lane advertises `rocep1s0f0`/`rocep1s0f1`; the host keeps all four (shared netplan, contract digest unchanged) |
| D5 | **`tp4-ring-4k` at the external 4 KiB receipt's sizing, `--unqualified-shape`** | our host is `6.17.0-1032-nvidia`, stock 4 KiB pages, Secure Boot on, no memory saver. Upstream's *qualified* `tp4-ring` needs the 64 KiB kernel + memory saver. `docs/setup.md` records an external site running exactly our variant (1M ctx / 16 seqs / KV 10.5 GiB) with MemAvailable 18–21 GiB through a 993K prefill. We adopt that receipt and gate it ourselves (Phase 2 G6/G8) rather than change the host baseline |
| D6 | **`NCCL_IB_MERGE_NICS=1` left as upstream ships it** | merge only applies to ≥2 stripes per link; our map has one HCA per neighbour, so the value is inert. Upstream's own ring example pairs two PCIe roots per neighbour and needs it. Changing it would be an unmeasured deviation from the release lock |
| D7 | **`NCCL_SWITCHLESS_BIDIRECTIONAL=2` kept, A/B scheduled** | upstream's shipped default assumes two PCIe roots; ours is one root. Phase 2 measures `=1` vs unset and keeps the winner |
| D8 | **Served name `deepseek-v41-flash`** | same as the SGLang lane, so GLM↔DeepSeek is the only proxy edit and `SPOOF_FROM` already contains it. Two lanes on one name is safe because only one serves at a time |
| D9 | **Guard table is per-recipe, with all four lanes' container names** | upstream's interlock matches only `spark-ds41f` and `^dsv41-`; it cannot see `glm53-*`. The one-lane-at-a-time precondition is this recipe's `bin/guard.sh` |
| D10 | **`deployment.branch` = the adoption branch, not `main`** | `cluster sync` refuses any commit not published on that branch. Pinned to `main` while the recipe is unmerged, every sync aborts "not published on origin/main". `bin/start.sh` enforces the same pin on the local checkout (wrong branch, or tracked edits → exit 4). Flip both to `main` in the merge commit |
| D11 | **Container removed on stop (repo teardown rule)** | upstream's `cluster start` refuses an existing exact container without `--replace`, and its memguard units can be left armed. Logs are captured per node first; JIT caches live in `{checkout}/cache` |

### What Phase 1 verified (all read-only, no container touched)

| check | result |
|---|---|
| `upstream/bin/spark doctor` | `configuration OK` |
| `upstream/bin/spark doctor --live` | `configuration OK; live cluster matches with 10 warnings`, 0 ERROR |
| `cluster start` (plan) | `4 nodes, TP4, context 1,048,576, KV 10.5 GiB, image spark-ds41f:1120`; startup order 3→2→1→0 with startup memguards |
| `render spark-0f0b` | `--served-model-name deepseek-v41-flash`, `NCCL_IB_HCA==rocep1s0f0,rocep1s0f1`, `NCCL_SOCKET_IFNAME=wlP9s9`, **no** `NCCL_IB_GID_INDEX`, `SPARKNET_ROCE_SUBNETS={"rocep1s0f0":"10.10.1.0/24","rocep1s0f1":"10.10.4.0/24"}`, `VLLM_HOST_IP=10.69.43.170` |
| `site/gate-gid-slot.sh` (live, all 4 nodes) | PASS — exactly one IPv4 RoCE v2 GID on each of the 4 routed PFs, 16/16 |
| `site/fabric/preflight.sh` | the host fabric contract, unchanged from the other three lanes (`CONTRACT_DIGEST` identical) |
| `bash -n` on every shell script in the recipe | clean |
| `bin/start.sh --dry-run` / `bin/stop.sh --dry-run` (throwaway copy) | guard verdict, drift assertion, and the exact delegate commands |

The 10 `doctor --live` warnings are the host baseline all four lanes share (64 KiB kernel,
memory saver, NVMe APST/coalescing, headless boot, modeset, UEFI order on two nodes,
de-fragmentation, image not yet built). Full list and dispositions:
`.research/HANDOFF-ds41f-inrepo-20261122.md` §6.1. **Do not run `sudo nmcli radio wifi off`** —
this site's management network *is* `wlP9s9`, and upstream's "management port off wireless"
warning assumes a wired mgmt port.

### Toolchain notes (gotchas found while building this)

- upstream imports `fcntl` → **its CLI only runs on Linux**. `doctor`/`build`/`cluster` must
  run on a node; the workstation can only hold the tree.
- the vendor script and the site appliers **must write LF**. A CRLF checkout on the Windows
  workstation produced `$'\r': command not found` in `cluster.env` and would have made
  `git diff d3d40d94` report every patched file. Appliers now force `newline="\n"`, and the
  recipe ships `.gitattributes` `* -text` so a future CRLF checkout cannot rewrite patch bytes.
- `host/recovery/` is required by `scripts/host-recovery` (setup stage 1) but was not in the
  build closure — added to the vendor list. 10 extra files.

### Phase 2

Written as `.research/HANDOFF-ds41f-inrepo-phase2-20261122.md` — execute it off the
`adopt-ds41f-inrepo` branch, on a session not hosted on the sparks. Build → distribute → boot
→ gates G1–G10 → D7 A/B → bench. The recipe stays `--unqualified-shape` until a run of ours is
promoted into `upstream/manifests/suites/`.

### Watchlist

| item | state at adoption | action |
|---|---|---|
| upstream `main` | **83 commits past the pin**, tip `candidate-68` (2026-11-22) — candidate work (LSA, CopyDrafter, adaptive verification, c32), never a deployment branch | deploy only a promoted `spark-ds41f-*` tag; re-vendor = new pin + re-verify the four patches + re-run `doctor` |
| forum `t384438` | 204 posts, active daily | read before any re-pin; the thread is where external 4 KiB receipts appear |
| our own qualified shape | none — `tp4-ring-4k` is unqualified | after Phase 2's bench, promote the run into `upstream/manifests/suites/` and drop `--unqualified-shape` |
| 64 KiB kernel / memory saver | not adopted | out of scope; a host-wide change that would affect every lane |

---

## Template for the next entry

```markdown
## YYYY-MM-DD — <what changed>
**what:** …
**measured before/after:** … (numbers, and how they were measured)
**gotchas:** …
**files:** …
```