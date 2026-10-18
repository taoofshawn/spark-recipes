# VENDORED-AT

- **upstream:** `github.com/FujitsuPolycom/sparkring`
- **pin:** commit `ad1853fc4c4cf4caad5e4e900ded44c22a57481d` (2026-10-17 tip = release
  **2026.10.5** target `ad1853fc`; the shipped release tag points at the same commit)
- **vendored:** 2026-10-17, shallow clone → subset copied (see scope below), `.git` not
  shipped
- **provenance chain:** sparkring's DeepSeek-V4.1-Flash lane itself credits
  [tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark)
  (MIT) for the seven runtime patches (SM12x page fixes by Kai); the image is built from
  `vllm/vllm-openai:nightly-8a728663c1c3…` + `vllm-project/vllm dsv41-feat @ e47aa780…` +
  FlashInfer `0.7.0rc1 @ 07869c61…` — all pins recorded in
  [image-receipt.json](image-receipt.json) and `build-image.sh`. SparkRing's own license
  is Apache-2.0 ([LICENSE-sparkring](LICENSE-sparkring)); their
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) is vendored alongside.

## Scope of the vendor (what was copied, what was left)

Copied (the **vLLM cycle lane** — NOT the installer/toolchain-image lane, whose image
`ghcr.io/fujitsupolycom/sparkring@sha256:ac313f04…` is a different runtime contract):

| vendored path | upstream source @ `ad1853fc` |
|---|---|
| `build-image.sh`, `README.md`, `image-receipt.json`, `patches/`, `tools/` | `runtime/deepseek-v41-gb10/*` (pristine) |
| `deepseek_v41_cycle_serve.sh` | `scripts/deepseek_v41_cycle_serve.sh` (2 site mods, below) |
| `deepseek-v41-flash-cycle.json` | `recipes/deepseek-v41-flash-cycle.json` (the machine-readable serving contract) |
| `profile-deepseek-v41-flash-cycle/` | `profiles/deepseek-v41-flash-cycle/` (guide + env template) |
| `docs/DEEPSEEK_V41_FLASH_QUICKSTART.md` | `docs/DEEPSEEK_V41_FLASH_QUICKSTART.md` |
| `THIRD_PARTY_NOTICES.md`, `LICENSE-sparkring` | repo root |

Left in upstream (referenced by receipts, not executed here): the installer itself,
`runtime/images/` toolchain (CUDA 13.4.2 + NCCL 2.32.3 + dual-PCI-domain patch),
`runtime/deepseek-v41-sglang/`, SIRCL/net-sircl, relay/hairpin services,
`performance/records/` (cited in research.md), `profiles/deepseek-v41-flash-tp4/`
(installer-lane profile; its `checkpoints.1017` entry is cited as the 1017 pin receipt).

## Site modifications (upstream files changed — MUST be re-applied on re-vendor)

| file | mod | why |
|---|---|---|
| `build-image.sh` (env only, no file change) | **#3 (2026-10-18, bring-up)**: the pinned base `vllm/vllm-openai:nightly-8a728663c1c3…` is GONE from Docker Hub (tag purged AND digest `sha256:1891ada1…` unpullable — "manifest schema unsupported"). Substituted `vllm/vllm-openai:nightly-dev-arm64-cu13.0.1-591bb95` (2026-09-15, arm64) locally re-tagged under the pinned name — no script edit. Verified in-image `torch 2.13.0+cu130` = the receipt's exact torch line, and the branch-critical op ABI: `torch.ops._moe_C.topk_softplus_sqrt` schema carries the branch's 12-arg signature (…is_padding, bias_vl, image_sentinel_lo). Consequence: rebuilt layer/image IDs will NOT match `image-receipt.json` (receipt stays as the upstream comparison target; actual IDs recorded in research.md). **First attempt used `nightly-dev-arm64-cu13.0.1-728d3ad` (2026-09-04) — REJECTED: its `_moe_C_stable_libtorch.abi3.so` (Aug-19 build inside that image) registers a 10-arg `topk_softplus_sqrt`, boot dies in `profile_run → speculator.propose` with "_moe_C::topk_softplus_sqrt() expected at most 10 argument(s)". The stable `.so` files are NOT rebuilt by build-image.sh — they come from the base, so the base must postdate the branch's fork-era main.** |
| `deepseek_v41_cycle_serve.sh` | **#1**: the HCA-count preflight `= 2` → `= 2 \|\| = 4` | the site four-PF contract (`NCCL_IB_HCA=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1`, both PCIe views of every cable) names FOUR devices; upstream's cycle contract is 2 PFs. Everything else in the launcher's transport validation is already the four-PF values (Ring algo, MERGE_NICS=0, subnet-aware routing, SKIP_TREE_CONNECT, SWITCHLESS_RING_ONLY) |
| `deepseek_v41_cycle_serve.sh` | **#2**: two added `-e` lines in the docker command: `NCCL_IB_EXTENDED_IPV4_GIDS=1 -e NCCL_IB_PRESERVE_PCI_DOMAIN=1 -e NCCL_IB_ROUTE_DIAGNOSTICS=1 -e NCCL_DEBUG_SUBSYS=INIT,NET` | required by the shared four-PF NCCL build (`~/.local/lib/nccl-patched-4pf`): without EXTENDED_IPV4_GIDS the four-PF lib fails at `ncclCommInitRank` with "wrong listener count" (AGENTS.md failure table), without PRESERVE_PCI_DOMAIN the P2-view GIDs (`roceP2p1s0*`) resolve to the wrong subnet. The upstream cycle launcher carries neither var because its SparkRing 2.30.7 2-PF lib doesn't need them |

Both mods are confined to `deepseek_v41_cycle_serve.sh`; `patches/*` are byte-identical
to upstream (`md5sum -c patches/MD5SUMS` passes inside `patches/`).

## Transport decision (why the site env points at the four-PF lib, not the donor)

Upstream's cycle recipe preloads SparkRing patched NCCL 2.30.7 extracted from donor
`ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:67dc0ae4…`. This site instead
bind-mounts the site's shared four-PF build `~/.local/lib/nccl-patched-4pf`
(SHA `8de68def…` — the SAME lib the GLM lanes use, validated on this exact fabric +
netplan) into the same mount path (`/opt/sparkring/nccl/libnccl.so.2`), so the launcher's
preloading mechanism is untouched. Rationale:

1. **Seamless switching** (the lane's premise): the GLM lanes' contract
   (`NCCL_SWITCHLESS_RING_ONLY=1`, `MERGE_NICS=0`, subnet-aware routing, extended IPv4
   GIDs, 4-PF HCA, MTU 9000) is already what sparkring's TP4 profile itself sets —
   same HCA list, same GID 3, same flags. One netplan + one NCCL lib serve all three
   lanes; `verify-fabric-p2.sh` remains the switch preflight.
2. The donor extraction procedure (docker pull of a ~23 GB image + `docker cp`) is a
   provision-time cost; the four-PF lib is already staged on all four nodes.

Rollback (if a boot fails in NCCL init): replace `NCCL_SO_HOST_PATH` in
`../site/env.switchless` with the donor path (`/path/to/libnccl.so.2` from the donor
image per the upstream quickstart) and REMOVE site mod #2's three transport vars — that
restores upstream's validated 2-PF cycle contract; also set `NCCL_IB_HCA` back to
`rocep1s0f0,rocep1s0f1` (and drop the four extra fabric addresses if netplan stays
additive — it is, the 2-PF addressing still exists underneath).

## Update procedure

1. `git ls-remote https://github.com/FujitsuPolycom/sparkring` → new tip; diff
   `ad1853fc..NEW` for the vendored subset paths (scope table above).
2. Re-check `recipes/deepseek-v41-flash-cycle.json` serving deltas and the seven patch
   MD5s (`patches/MD5SUMS`); re-apply the two site mods to
   `deepseek_v41_cycle_serve.sh` (both are grep-able: "SITE MOD #1/#2").
3. Any new checkpoint: sparkring's `profiles/deepseek-v41-flash-tp4/config.json`
   `checkpoints` block is the pin receipt pattern; verify config.json + shard count are
   byte-identical before re-pinning (the 1017 adoption receipt is in `../research.md`).
