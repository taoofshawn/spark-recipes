# VENDORED-AT — upstream-knapcio

- **upstream:** `github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4`
- **pin:** commit `540bf07` (2026-11-06, tip; "Merge PR #68: loop abort named error exit,
  history repair for broken tool calls") — release **v2.6.4** (`d39e8c5`) plus #68
- **previous pin:** `aff52fa` (2026-10-24, pre-v2.6), vendored 2026-10-26
- **re-vendored:** 2026-11-06 (43 upstream commits skipped: v2.6 → v2.6.1 → v2.6.2 →
  v2.6.3 → v2.6.4 → #68); `.git` stripped (detached; adopted, not tracked)
- **scope:** the whole tree — this IS the serving stack (launcher `start.sh`/`start-tp4.sh`,
  image build `Dockerfile.canary-roce`, adapters, boot.py, benchmarks, qeval). The site
  integration lives OUTSIDE in `../site/` (mirroring how the NVFP4 lane wraps its vendored
  knapcio upstream).
- **why this stack (2026-10-26 vehicle swap from MiaAI-Lab, pin `64b02dea`):** the
  2026-10-25/26 retention investigation closed the idle-retention case (`--swa-full-tokens-ratio
  0.3` + the #35515 tail-deprioritization) but the agent-turn shape still re-prefilled —
  attributed then to the DSV4 compressor-state ring. knapcio's tree tracks the same dsv4.1
  SGLang branch with a larger, credited backport set (#41303/#42901/#44001/#41259/#45410/
  #43549/#44751/#45301/#40424/#43448/#45457/…/#44923 — ~24 more than the MiaAI pin carried),
  a re-qualified production profile on this exact checkpoint (2026-10-22, issue-16
  reproducer: 1,445 requests, 0 failures), and a more active upstream. The previous stack is
  archived at `../upstream-miaai-archive/`.
- **why the v2.6.4 refresh (2026-11-06):** v2.6.4 ships the SWA prefix-cache retention work
  that closes the agent-turn re-prefill at the eviction layer — `DSV41_SWA_THIN` (superseded
  chunk-boundary SWA checkpoints evicted first), `--swa-prefix-tails 384` (was 128),
  `DSV41_SWA_CONE_REPLAY` (a lost resume point costs a one-chunk replay instead of a
  whole-context re-prefill), plus `DSV41_DECODE_YIELD` and `DSV41_SWA_NATIVE_DECODE`.
  Adoption rationale and the site gate: `../research.md` 2026-11-06.

## Third-party sources (staged by scripts/fetch-sglang-canary.sh; pinned, not tracked)

| component | pin | license |
|---|---|---|
| SGLang dsv4.1 branch python tree | `f80c91a4b97e52e17fbcff088aa6b47b6270f6b9` (SAME pin the MiaAI stack used, and the SAME pin v2.6.4 targets) + the fetch script's credited backport set (unchanged `aff52fa` → `540bf07`) | Apache-2.0 |
| sglang-kernel / sgl-deep-gemm | 0.4.7 / 0.2.0 (aarch64 PyPI wheels, installed by the Dockerfile) | upstream |
| b12x SG17 (RoCEnante) + b12x_next | vendored under `runtime/b12x*` (his repo); v2.6.2 adds the b12x `d7937d7f` claim-barrier patch, v2.6 adds the `roce_twoshot` package (opt-in, off here) | Apache-2.0 |
| FlashInfer modules | `runtime/flashinfer_pce` (v2.6.2, prefill native extra pages), `runtime/flashinfer_v41c` (v2.6.4, swan decode), `runtime/flashinfer_fi7676` (v2.6.3 proxy fence) — all built from the image's own approved source closure, sha256-pinned .so | FlashInfer, Apache-2.0 |
| base image | `lmsysorg/sglang:dev-dsv41` — pinned to digest `sha256:3dbc3130…` via `BUILD_ARGS` in `../site/env.switchless` (the vendored Dockerfile's `ARG BASE_IMAGE` defaults to the MUTABLE tag; the digest pin must ride BUILD_ARGS in every build path) | — |

`runtime/sglang-canary/` is **gitignored** in his repo: the Dockerfile COPYs a tree that
`scripts/fetch-sglang-canary.sh` stages on the host BEFORE the build. The site's
`build-cluster.sh` runs that fetch on the head stack (once; idempotent pinned refs) and
rsyncs the staged tree to each worker build tree — `git archive` cannot carry it.

## Known path (2026-11-06): the SWA retention fix lands on the unified radix cache

The pinned canary tree `f80c91a4b` instantiates **`UnifiedRadixCache`** for this lane — the
radix-cache selection chain falls through to `_create_unified_radix_cache` and builds
components `[FULL, SWA]` (`sglang/srt/mem_cache/registry.py:146`). The legacy
**`SWARadixCache` is never constructed**, so `swa_radix_cache.py` (and the site's former
#35515 tail-eviction patch aimed at it) was dead code on this lane; it is REMOVED as of this
re-vendor (research.md 2026-11-06).

v2.6.4's `adapter/swa_thin.py` hooks the live class, with sha256 engine pins that were
verified **in the deployed image** on 2026-11-06:

| pinned engine source | sha256 (prefix) |
|---|---|
| `UnifiedRadixCache.cache_unfinished_req` | `d64c72cfc5f6b129…` |
| `UnifiedLRUList._add_node_after` | `7b84b69969bc0473…` |
| `UnifiedLRUList._remove_node` | `e0101fd38b9dc801…` |

`adapter/swa_cone.py`'s nine additional pins (schedule_batch / schedule_policy /
forward_batch_info / unified_tree_core / swa component / common) all matched on the same
image. Both adapters install-or-refuse (drift = logged no-op), never half-install.
NOTE: `swa_cone` additionally requires `--enable-decoder-swa-bounded-replay` at launch
(`cache.decoder_swa_bounded_replay`) — the 2026-11-05 site delta that disabled that flag is
reverted for this reason (research.md 2026-11-06).

## NCCL transport (external prerequisite, NOT in this tree)

Unchanged from the MiaAI adoption: `NCCL_HOST_DIR` points at the cluster's four-PF build
`~/.local/lib/nccl-patched-4pf` (othexmr @ `94c2669` + one-line site GID patch, SHA
`8de68def…`; `SWITCHLESS_RING_ONLY` marker verified on all four nodes 2026-10-21). His
launcher overlays it onto the container's pip NCCL (`NCCL_OVERLAY_PIP=1`, ring-mode
default) and refuses a stock build by grepping the marker; the four-PF advertisement flags
(`NCCL_IB_EXTENDED_IPV4_GIDS/PRESERVE_PCI_DOMAIN/ROUTE_DIAGNOSTICS/QPS_PER_CONNECTION`)
ride `EXTRA_CONTAINER_ENV` in `../site/env.switchless`.

## Site modifications to upstream files — MUST be re-applied on re-vendor

Three site modifications are carried: the **thinking-mode completion floor**, the
**SGLANG_DEFAULT_THINKING explicit-kwarg injection** (both ported from the archived MiaAI
tree 2026-10-26), and the **`Dockerfile.canary-roce` RUN collapsing** (2026-10-26). The
**#35515 SWA tail-eviction backport was REMOVED on 2026-11-06** (dead code, see "Known
path" above). The image placeholder sanitize is NOT a site mod here — knapcio ships his own
stricter `runtime/patch_encoding_dsv41.py` and runs it in `Dockerfile.canary-roce` (same
sanitize-and-warn semantics in `_validate_no_image_sp_tokens` and the text-block branch;
our MiaAI-tree variant is NOT carried). Everything else is supplied by variable
(`IB_HCA`, `NCCL_HOST_DIR`, `PORT`, `SERVED_MODEL_NAME`, `WORKER_HOSTS`, `MODEL_DIR`, …).

| file | change | re-apply |
|---|---|---|
| `adapter/encoding_compat.py` | `DEFAULT_MAX_NEW_TOKENS_FLOOR = 16384`, `configured_max_new_tokens_floor()`, `request_enables_thinking_explicitly()`, the floor branch at the top of `apply_request_max_tokens()`, the `SGLANG_DEFAULT_THINKING` explicit-kwarg injection in `install_serving_chat`'s `_process_messages` wrapper, and the floor line in the boot-time warning | re-add the blocks after upstream's `DEFAULT_MAX_NEW_TOKENS` / `configured_max_new_tokens()`. The floor gate MUST stay explicit-only: an engine-default fallback there rewrites fixed budgets sent by measurement clients (2026-10-23 contamination). The injection MUST stay: the reasoning parser keys off `chat_template_kwargs`, so a default-on request otherwise leaks the think block into `content` unsplit (verified live 2026-10-23). Upstream `encoding_compat.py` was UNCHANGED `aff52fa`→`540bf07`, so the site file is a clean superset |
| `Dockerfile.canary-roce` (RUN collapsing) | the 41 per-script verify/test `RUN`s (now 2 upstream blocks: 27 + 12) are collapsed into `&&`-chained single RUNs — the pinned base carries a deep layer chain (~180 history entries) and one-layer-per-script exceeded buildkit's max depth mid-build ("failed to prepare … max depth exceeded", hit 2026-10-26 on spark-0f0b). Every script still runs, in order, &&-gated; only layer count/cache granularity change | re-collapse after any upstream Dockerfile change that adds per-script RUNs (2026-11-06 re-apply done with `.research/collapse-dockerfile-runs.ps1`, ranges 71-97 and 99-110 of the v2.6.4 file) |
| `start.sh` | `-e "DSV41_MAX_NEW_TOKENS_FLOOR=${DSV41_MAX_NEW_TOKENS_FLOOR:-16384}"` under each of the two `DSV41_MAX_NEW_TOKENS` `-e` lines (head + worker heredoc) | re-add both lines; `start.sh`'s `-e` whitelist is why this cannot ride in the site env alone. v2.6.4 adds `DSV41_LOOP_NAMED_EXIT` / `DSV41_HISTORY_REPAIR` to the same whitelist (upstream's own change, present) |
| `.env.example`, `.env.tp4.example` | `DSV41_MAX_NEW_TOKENS_FLOOR` knob + rationale after `DSV41_MAX_NEW_TOKENS` | re-add the rows |
| `tests/test_max_new_tokens.py` | replaced with the site superset: upstream's cap tests + the floor boundary tests (explicit-only gate, engine-default-is-ignored, cap-bounded floor, `max_completion_tokens` floor, floor-off), incl. the end-to-end install-ordering guard for the 2026-10-23 contamination. This file runs at image build (Dockerfile line `test_max_new_tokens.py`) so the floor gate is enforced in-image | re-copy from `../upstream-miaai-archive/tests/test_max_new_tokens.py`; upstream did not touch this file `aff52fa`→`540bf07`. 20/20 pass on the re-vendored tree (2026-11-06) |
| — (not carried) | MiaAI-tree `runtime/patch_encoding_dsv41.py` variant | knapcio's own `runtime/patch_encoding_dsv41.py` is stricter (whole-block anchors + completeness self-checks) and already wired; ours was only needed because the MiaAI tree's layout differed |
| — (REMOVED 2026-11-06) | `runtime/patch_swa_tail_eviction.py` (sglang #35515 backport) + its two Dockerfile lines | deleted: it patched `SWARadixCache.evict()`, a class this tree never instantiates (see "Known path"). `DSV41_SWA_THIN` is the unified-cache equivalent, shipped by upstream. Do not resurrect |

Why the floor: with thinking on, a request sized for a non-reasoning client (MMLU's
`max_tokens=256`) spends the whole budget in the reasoning segment and returns
`finish_reason=length` with empty content — a silent zero. Receipts and rejected
alternatives: `../research.md`, 2026-10-22 "benchmark analysis: thinking-mode budget
starvation" and 2026-10-23 "measurement contamination". knapcio's README table is
profile-scoped and does not document the cap/floor pair — the recipe README and this file
are the floor's documentation.

## Update procedure

1. `git ls-remote https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4 main`
2. Diff the pinned tree (`540bf07`) against the new tip: launcher/image/adapter changes
   are adoptable after the in-image test suite passes (the Dockerfile runs the adapter
   tests at build, including the floor suite and the `test_swa_thin`/`test_swa_cone`
   engine-pin suites).
3. Re-check `scripts/fetch-sglang-canary.sh` pins changed — new backport refs there
   cascade into the image build (per-node rebuild + gates, a bench-recipe-update window).
   The canary base ref `f80c91a4b` did NOT move `aff52fa`→`540bf07`.
4. Re-apply any site modifications recorded above.
5. Note: the watchlist's sglang re-pin plan (research.md §6 item 8 — move the canary to a
   release line carrying #40313) is now known to be a drift/migration item only: the pinned
   tree already ships `UnifiedRadixCache`, which is what this lane runs. That plan
   supersedes this update procedure when it runs.