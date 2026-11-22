# REUSE — provenance of the vendored fabric material in this recipe

Everything under `site/fabric/` is a **copy** of material that originates in another recipe
in this repository, which itself vendored it from upstream. D1 (full vendoring) requires the
copy: this recipe must work with every sibling recipe directory deleted.

## Source

| | |
|---|---|
| repository | `glm-v53-flash-fp8-tp4-switchless/upstream/` (this repo) |
| upstream origin | `github.com/jnardiello/GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless` |
| pin at copy time | `ed365a6` upstream / repo commit `ef36408` |
| license | MIT — Copyright (c) 2026 Jacopo Nardiello (see `REUSE-LICENSE-MIT-Nardiello.txt`) |
| copied | 2026-11-20 |

The host-align scripts (`scripts/node/host/{kernel-align-fresh,hold-existing,driver-align-173,docker-log-rotation,verify-hostkeys}.sh`) came from that recipe's `noswitch-prep/scripts/` (site-written, same MIT terms).

## What was copied, and what changed

| file | change |
|---|---|
| `scripts/render-netplan.sh`, `scripts/render-netplan.md` | reads this recipe's `site/fabric/cluster.env` via the vendored `lib/common.sh`; default output dir is this recipe's committed netplan dir; comment text rewritten to describe the contract rather than a sibling recipe. **No logic change.** |
| `scripts/lib/common.sh` | only `TP4_REQUIRED_KEYS` / `TP4_SITE_KEYS` reduced to the fabric-facts key set (this recipe's `cluster.env` deliberately carries no serving values — those live in `site/env.switchless`). Everything below those two lists is upstream code, unmodified. |
| `scripts/bootstrap-node.sh` | same key-list consequence; renders/stages from `NETPLAN_DIR` (this recipe's committed netplan); the upstream `autostart` phase (a systemd unit) is **removed** — this recipe is started by `bin/start.sh`. Phase semantics `packages`, `sudoers`, `etc`, `ssh-mesh`, `layout` and every `--check/--apply/--phase/--only` flag are unchanged. |
| `scripts/deploy-host.sh` | same; `/etc` set + netplan install only, sha-verified, never activating. Host-align scripts stage in `~/.local/tp4-fabric/host/` (lane-neutral host state) instead of another lane's `~/.local/tp4/host/`. |
| `scripts/node/etc/**`, `scripts/node/host/tp4-iommu.sh`, `scripts/node/bootstrap/versions.env`, `scripts/node/ssh-config.example` | verbatim |
| `scripts/node/host/{kernel-align-fresh,hold-existing,driver-align-173,docker-log-rotation,verify-hostkeys}.sh` | verbatim |

Each copied script carries a two-line provenance header naming the source path and this pin.
Upstream headers were not stripped.

## NCCL build trees

| tree | origin | notes |
|---|---|---|
| `nccl-4pf/` | this repo's own site build (othexmr/switchless-nccl `94c2669` + a one-line site GID patch), previously only in the NVFP4 recipe | 6 files; patch SHAs asserted at the top of `build-4pf.sh:44-50`; builds `~/.local/lib/nccl-patched-4pf` = `8de68def…` |
| `nccl-2pf/` | *(NVFP4 only)* `glm-v53-flash-fp8-tp4-switchless/upstream/scripts/node/nccl/` (MIT, Nardiello), 2-PF SparkRing family | the documented rollback so it is executable from this recipe alone; builds `~/.local/lib/nccl-patched` = `afe5f486…` |

`build-4pf.sh` is byte-identical in all three recipes (sha256 `0d86c2e1…`): a change to it is a
deliberate three-recipe change, exactly like the contract digest. The NVFP4 origin copy lost its
"mirrors the FP8 recipe" prose for the same reason.

## What was NOT copied

Serving-side material stays where it is in this recipe: `site/env.switchless` (image, weights,
KV, context, container name) and `upstream/` (the vendored serving stack). This recipe's
`site/fabric/` deliberately contains **no** serving values, so the two cannot drift against
each other.

## Regenerating

The netplan is generated, never hand-edited:

```bash
cd site/fabric
bash scripts/render-netplan.sh --write   # regenerate the committed expectation
bash scripts/render-netplan.sh --check   # the acceptance test
```

`preflight.sh` runs the same `--check` and prints `CONTRACT_DIGEST`, which must equal the
value in every other recipe's facts file.