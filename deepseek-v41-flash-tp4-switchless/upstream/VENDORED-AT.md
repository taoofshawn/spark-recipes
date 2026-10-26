# VENDORED-AT — upstream-knapcio

- **upstream:** `github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4`
- **pin:** commit `aff52fa` (2026-10-24 16:27 +0200, tip; "PR #57 follow-up: credit and
  changelog")
- **vendored:** 2026-10-26, full clone → `.git` stripped (detached; adopted, not tracked)
- **scope:** the whole tree — this IS the serving stack (launcher `start.sh`/`start-tp4.sh`,
  image build `Dockerfile.canary-roce`, adapters, boot.py, benchmarks, qeval). The site
  integration lives OUTSIDE in `../site/` (mirroring how the NVFP4 lane wraps its vendored
  knapcio upstream).
- **why this stack (2026-10-26 vehicle swap from MiaAI-Lab, pin `64b02dea`):** the
  2026-10-25/26 retention investigation closed the idle-retention case (`--swa-full-tokens-ratio
  0.3` + the #35515 tail-deprioritization) but the agent-turn shape still re-prefills —
  root cause is the DSV4 compressor-state ring, a stack-level design gap the MiaAI tree
  does not address. knapcio's tree tracks the same dsv4.1 SGLang branch with a larger,
  credited backport set (#41303/#42901/#44001/#41259/#45410/#43549/#44751/#45301/#40424/
  #43448/#45457/…/#44923 — ~24 more than the MiaAI pin carried), a re-qualified production
  profile on this exact checkpoint (2026-10-22, issue-16 reproducer: 1,445 requests, 0
  failures), and a more active upstream. The previous stack is archived at
  `../upstream-miaai-archive/`.

## Third-party sources (staged by scripts/fetch-sglang-canary.sh; pinned, not tracked)

| component | pin | license |
|---|---|---|
| SGLang dsv4.1 branch python tree | `f80c91a4b97e52e17fbcff088aa6b47b6270f6b9` (SAME pin the MiaAI stack used) + the fetch script's credited backport set | Apache-2.0 |
| sglang-kernel / sgl-deep-gemm | 0.4.7 / 0.2.0 (aarch64 PyPI wheels, installed by the Dockerfile) | upstream |
| b12x SG17 (RoCEnante) + b12x_next | vendored under `runtime/b12x*` (his repo) | Apache-2.0 |
| base image | `lmsysorg/sglang:dev-dsv41` — pinned to digest `sha256:3dbc3130…` via `BUILD_ARGS` in `../site/env.switchless` (the vendored Dockerfile's `ARG BASE_IMAGE` defaults to the MUTABLE tag; the digest pin must ride BUILD_ARGS in every build path) | — |

`runtime/sglang-canary/` is **gitignored** in his repo: the Dockerfile COPYs a tree that
`scripts/fetch-sglang-canary.sh` stages on the host BEFORE the build. The site's
`build-cluster.sh` runs that fetch on the head stack (once; idempotent pinned refs) and
rsyncs the staged tree to each worker build tree — `git archive` cannot carry it.

## NCCL transport (external prerequisite, NOT in this tree)

Unchanged from the MiaAI adoption: `NCCL_HOST_DIR` points at the cluster's four-PF build
`~/.local/lib/nccl-patched-4pf` (othexmr @ `94c2669` + one-line site GID patch, SHA
`8de68def…`; `SWITCHLESS_RING_ONLY` marker verified on all four nodes 2026-10-21). His
launcher overlays it onto the container's pip NCCL (`NCCL_OVERLAY_PIP=1`, ring-mode
default) and refuses a stock build by grepping the marker; the four-PF advertisement flags
(`NCCL_IB_EXTENDED_IPV4_GIDS/PRESERVE_PCI_DOMAIN/ROUTE_DIAGNOSTICS/QPS_PER_CONNECTION`)
ride `EXTRA_CONTAINER_ENV` in `../site/env.switchless`.

## Site modifications to upstream files — MUST be re-applied on re-vendor

Three site modifications are carried (all ported from the archived MiaAI tree on
2026-10-26): the **thinking-mode completion floor**, the **SGLANG_DEFAULT_THINKING
explicit-kwarg injection**, and the **#35515 SWA tail-eviction backport**. The image
placeholder sanitize is NOT a site mod here — knapcio ships his own stricter
`runtime/patch_encoding_dsv41.py` and runs it in `Dockerfile.canary-roce` (same
sanitize-and-warn semantics in `_validate_no_image_sp_tokens` and the text-block branch;
our MiaAI-tree variant is NOT carried). Everything else is supplied by variable
(`IB_HCA`, `NCCL_HOST_DIR`, `PORT`, `SERVED_MODEL_NAME`, `WORKER_HOSTS`, `MODEL_DIR`, …).

| file | change | re-apply |
|---|---|---|
| `adapter/encoding_compat.py` | `DEFAULT_MAX_NEW_TOKENS_FLOOR = 16384`, `configured_max_new_tokens_floor()`, `request_enables_thinking_explicitly()`, the floor branch at the top of `apply_request_max_tokens()`, the `SGLANG_DEFAULT_THINKING` explicit-kwarg injection in `install_serving_chat`'s `_process_messages` wrapper, and the floor line in the boot-time warning | re-add the blocks after upstream's `DEFAULT_MAX_NEW_TOKENS` / `configured_max_new_tokens()`. The floor gate MUST stay explicit-only: an engine-default fallback there rewrites fixed budgets sent by measurement clients (2026-10-23 contamination). The injection MUST stay: the reasoning parser keys off `chat_template_kwargs`, so a default-on request otherwise leaks the think block into `content` unsplit (verified live 2026-10-23) |
| `Dockerfile.canary-roce` + `runtime/patch_swa_tail_eviction.py` | after his sanitize step: `COPY runtime/patch_swa_tail_eviction.py /opt/dsv41/runtime/` + `RUN python3 …` — site backport of sgl-project/sglang#35515 (open upstream, @QAQEthan; mooted on main by #40313 deleting `SWARadixCache`, but the pinned canary tree `f80c91a4` still runs it). `evict()` skips candidates within one page-aligned window of a branch end and retries them only as a fallback — a tombstoned tail makes the whole branch unmatchable and the tombstone-leaf deletion then frees the branch's full KV | re-copy the patch (kept in-tree at `runtime/patch_swa_tail_eviction.py`) and re-add the two Dockerfile lines after the sanitize step. The patch's own anchor check fails the build if the canary tree drifts; verified 3/3 anchors against the freshly staged `f80c91a4` tree on 2026-10-26 |
| `start.sh` | `-e "DSV41_MAX_NEW_TOKENS_FLOOR=${DSV41_MAX_NEW_TOKENS_FLOOR:-16384}"` under each of the two `DSV41_MAX_NEW_TOKENS` `-e` lines (head + worker heredoc) | re-add both lines; `start.sh`'s `-e` whitelist is why this cannot ride in the site env alone |
| `.env.example`, `.env.tp4.example` | `DSV41_MAX_NEW_TOKENS_FLOOR` knob + rationale after `DSV41_MAX_NEW_TOKENS` | re-add the rows |
| `tests/test_max_new_tokens.py` | replaced with the site superset: upstream's cap tests + the floor boundary tests (explicit-only gate, engine-default-is-ignored, cap-bounded floor, `max_completion_tokens` floor, floor-off), incl. the end-to-end install-ordering guard for the 2026-10-23 contamination. This file runs at image build (Dockerfile line `test_max_new_tokens.py`) so the floor gate is enforced in-image | re-copy from `../upstream-miaai-archive/tests/test_max_new_tokens.py` |
| — (not carried) | MiaAI-tree `runtime/patch_encoding_dsv41.py` variant | knapcio's own `runtime/patch_encoding_dsv41.py` is stricter (whole-block anchors + completeness self-checks) and already wired; ours was only needed because the MiaAI tree's layout differed |

Why the floor: with thinking on, a request sized for a non-reasoning client (MMLU's
`max_tokens=256`) spends the whole budget in the reasoning segment and returns
`finish_reason=length` with empty content — a silent zero. Receipts and rejected
alternatives: `../research.md`, 2026-10-22 "benchmark analysis: thinking-mode budget
starvation" and 2026-10-23 "measurement contamination". knapcio's README table is
profile-scoped and does not document the cap/floor pair — the recipe README and this file
are the floor's documentation.

## Update procedure

1. `git ls-remote https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4 main`
2. Diff the pinned tree (`aff52fa`) against the new tip: launcher/image/adapter changes
   are adoptable after the in-image test suite passes (the Dockerfile runs the adapter
   tests at build, including the floor suite).
3. Re-check `scripts/fetch-sglang-canary.sh` pins changed — new backport refs there
   cascade into the image build (per-node rebuild + gates, a bench-recipe-update window).
   If the canary base ref moves, the `patch_swa_tail_eviction.py` anchors must be
   re-verified (the build fails on drift by design).
4. Re-apply any site modifications recorded above.
5. Note: the watchlist's sglang re-pin plan (research.md §6 item 8 — move the canary to a
   release line carrying #40313) makes most of this site-mod list moot on the next major
   migration; that plan supersedes this update procedure when it runs.
