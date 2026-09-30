# research.md — glm-v53-flash-nvfp4-4x-noswitch

Audit trail, provenance, and the decision record for the zero-weight-modification
NVFP4 lane on the 4-node switchless ring. `README.md` stays the running ops doc only.

## 1. Provenance

| what | value |
|---|---|
| upstream source | `github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4` |
| vendored pin | `770d115` (2026-09-29 tip; `3edfbc9..770d115` diffed pre-vendor: docs + the switchless transport only — no serving-kernel changes) |
| vendored on | 2026-09-29, `.git` stripped, detached |
| checkpoint | [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) @ `09b04e5e74bca08ca8549fc736d4cdd8624bfde3` — **stock, zero modification** (already in all four nodes' default HF caches) |
| drafter | [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) @ `bf582e4e` — stock bf16 (no `drafter_fp8.py` re-encode; CC BY-NC-ND 4.0) |
| image | `glm53-roce:v11-b58f34ea`, built per-node from upstream `Dockerfile.roce` (RoCEnante present but inert under `TRANSPORT=switchless`) |
| site files | `site/env.switchless` (the `.env`), `site/materialize-weights.sh`, `upstream/profiles/stock-nvfp4.env`, `upstream/VENDORED-AT.md` |

Why knapcio (not alexellis / jnardiello / tonyd2wild): only stack that (a) anchors on the
exact nvidia checkpoint with its NVFP4 experts untouched, (b) ships a switchless-ring
transport mode, (c) is the measured fastest (switched fleet, one boot, 2026-09-29: c1
decode prose 90.3 / code 124.0 / structured 168.8 tok/s; c16 aggregate 331–879; cold
prefill 2.8–3.5K tok/s to 128K; 3.6-min boot; GSM8K 246/250, HumanEval 95.7%).

## 2. Decision record (2026-09-29, owner-confirmed)

1. **Name** `glm-v53-flash-nvfp4-4x-noswitch` (parallel to the FP8 recipe).
2. **NCCL: inspect-first.** Hardware facts (owner, 2026-09-29): each inter-node link is a
   200G QSFP56 cable presenting **2×100G virtual PFs** (one per PCIe root — the 2-node
   recipes' `IB_PORTS=rocep1s0f0,roceP2p1s0f0` pair is one cable's two functions; the
   `.f1` names are the right-hand port). The four-PF contract therefore asks for devices
   that demonstrably exist; the gap is **configuration, not hardware**: the FP8 ring
   addresses only one PF per port (2 of 4 per node — its "two addressed fabric interfaces
   per node" invariant). Bring-up step 1: (a) netplan-address the second PF of each port
   on all four nodes (fabric plan extends 4 → 8 logical links; site range widened to
   `10.10.0.0/21` in site/env.switchless), (b) run `scripts/check_switchless_nccl.py` on
   the existing site rebuild (`2.30.7-1`, SHA `afe5f486…`) — configuring PFs alone is not
   sufficient: "a two-device-only release is not sufficient merely because its filename is
   the same" (upstream docs/switchless.md), so the build must also support four devices.
   If it fails, build othexmr's four-PF NCCL (`feat/four-pf-source-integration` @
   `94c2669`) against the site addressing. **Known risk:** upstream's four-PF build was
   validated on `10.100.224.0/22`; our ring is `10.10.0.0/21` — a site rebuild+validation
   is likely. Fallback lane: the FP8 recipe's proven fabric/NCCL wiring against this image
   (jnardiello's fidelity arm booted the tonyd2wild image family TP4 on exactly this
   fabric; do NOT set `NCCL_SWITCHLESS_RING_ONLY`/`NCCL_IB_MERGE_NICS=0` — failed RDMA QP
   connect there).
3. **Weights: already on all four nodes** (owner, default HF cache). No fan-out needed;
   materialization (`site/materialize-weights.sh`) hardlinks the snapshots into
   real-file dirs because `start.sh` bind-mounts `MODEL_DIR` directly (`/model`) and HF
   snapshots use relative symlinks into `blobs/` that break inside a container mount.
   Hardlinks = zero copy, zero disk, provably unmodified bytes.
4. **v1 profile: 1M ctx on a conservative pool.** `MAX_MODEL_LEN=1048576`,
   `KV_BYTES=21474836480` (20 GiB/rank → ~3.25M fp8-KV tokens at the measured ~40.6K
   tokens/GiB → 3.1× full-1M concurrency). Owner explicitly chose the 3.1× conservative
   ratio; 24→28 GiB/rank rungs are proven ceilings on identical hardware (tonyliu312
   4,545,221 tokens @ 28 GiB/rank, head rank binding) but gate each rung.
5. **Zero weight modification is the product, not an accident** — if the lane is too
   slow/unpleasant the fallback is knapcio's lossless8 target (`scripts/build_lossless8.sh`
   + `drafter_fp8.py`, 0.25% output error, ~10–15% decode gain + ~9 GiB KV) — revisit
   decision explicitly, don't drift into it.

## 3. The stock-compat delta (what `profiles/stock-nvfp4.env` strips and why)

`current.env` @ `770d115` assumes the lossless8 target and the fp8blk drafter for:

| stripped switch | assumes |
|---|---|
| `QMIX_FP8_BLOCK=1`, `VLLM_TEST_FORCE_FP8_MARLIN=1`, `QMIX_DEBUG_LAYERS=1` | lossless8 overlay loader filter + sentinel (no overlay exists on the stock pack) |
| `GLM_DENSE_FAST=1`, `GLM_DENSE_FAST_PREFILL=1` | dense layers on 8-bit grids (stock pack's dense layers are BF16) |
| `GLM_DS_DRAFT_HEAD_FP8=1`, `GLM_LV_DRAFT_FP8_KV=1` | the fp8blk-converted drafter (this lane serves stock bf16) |

Everything else is inherited live (source-order: `stock-nvfp4.env` → `current.env`), so
upstream kernel/scheduler wins keep arriving. **Watch item:** any upstream
`profiles/current.env` change needs a stock-compat review of new switches — a new
QMIX/DENSE/drafter-FP8 switch inherited silently is the failure mode. Expected cost of
the strip list: ~10–15% decode (−7.5 ms/step dense read) + ~1.1 ms drafter graph;
measured at bring-up, not assumed.

## 4. Receipts carried into this recipe

| claim | receipt |
|---|---|
| nvidia pack = corruption-free (0/0/0 U+FFFD; attention kept BF16, 132-entry ignore list) | tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark README, issue #23 |
| stock nvidia pack boots unmodified on the tonyd2wild image family (TP2) | same repo, DFlash2 launcher default since 2026-09-24 |
| stock nvidia pack TP4 = unmeasured publicly; the "does not come up" failure applies only to the built attn-quantized checkpoint | tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark README (`NVFP4_PATCH` section) |
| native MTP impossible on the nvidia pack (BF16 MTP head; no sm121 backend serves NVFP4 target + unquantized draft MoE) → DFlash2 is the only drafter | tonyd2wild issue #23 |
| ~40.6K fp8-KV tokens per GiB (block 2304, DFlash2 slot-share) | tonyliu312: 4,545,221 @ 28 GiB/rank; tonyd2wild 1M lane: 3,834,498 @ 24 GiB/rank |
| knapcio switched-fleet performance + quality gates | upstream README @ `770d115` (sparkDash 1.8.8, RigMark screen, qeval/GSM8K/HumanEval) |
| NVFP4-experts-vs-FP8-checkpoint on the same engine: code +40%, JSON +50% | upstream README (Marlin W4A16 vs official FP8 checkpoint, same quality gate) |
| precision ladder: W4A4 experts 16–21% MoE error (rejected); 8-bit dense 0.25%; NVFP4 experts = pack quant | upstream README rejected-experiments + weights docs |
| switchless transport = community-contributed, CPU-tested only upstream; ring decode penalty unquantified | upstream `docs/switchless.md` (othexmr PR #1) |
| mamba-align correctness fix must be active | upstream issue #2 + `GLM_MAMBA_ALIGN_FIX=1` (inherited) + `bench/prefix_scan.py` gate |

## 5. Watchlist / open items

- **Second-PF netplan plan** — address the unconfigured PF (one per port, 2/node) on all
  four nodes; 8 logical links need 8 /24 subnets within `10.10.0.0/21`; record the final
  per-link map here once rendered (extend the FP8 recipe's netplan approach).
- **NCCL four-device verdict** on the existing `afe5f486…` rebuild — record pass/fail and,
  if a four-PF build is made, its SHA256 + validation notes here.
- **1M ctx on this stack** — knapcio validated 262K switched; tonyliu312 validated 1M on
  the sibling tonyd2wild stack. First 1M-context gate on this exact stack is ours to
  record.
- **Ring decode penalty** — the lane-defining number; measure
  (`accept_probe`/`conc_bench`) and compare against knapcio's switched tables.
- **GLM_ROCE_ALLREDUCE / GATHER_ROUTE behavior under `TRANSPORT=switchless`** — CPU-tested
  upstream, never hardware-qualified; if the ring misbehaves, first suspects.
- **glm47 tool-call `tool_choice:required` handling** — the FP8 recipe needed a
  structural-tag override (TC-45); upstream here ships its own `glm47_moe.py` overlay.
  Verify forced tool calls at bring-up before declaring parity with the FP8 lane.
- **Upstream churn** — knapcio's `main` moves fast (two releases + a hotfix in the last
  three days of September); re-run the VENDORED-AT update procedure on any adopted change.

## Changelog

- **2026-09-29** — Recipe created: vendored knapcio `770d115`, site env + zero-mod
  profile + weight materialization script; decisions 1–5 recorded above. Uncommitted,
  awaiting owner review.
