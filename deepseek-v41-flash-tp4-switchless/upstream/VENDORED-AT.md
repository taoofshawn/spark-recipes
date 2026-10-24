# VENDORED-AT — upstream-miaai

- **upstream:** `github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks`
- **pin:** commit `64b02dea8681fb1d8e3a659ea432a8e9e9ba3995` (2026-10-17 23:52 +0300, tip;
  "start-tp4-switchless.sh: TP4 launcher profile for a switchless ring")
- **vendored:** 2026-10-21, full clone → `.git` stripped (detached; adopted, not tracked)
- **scope:** the whole tree — this IS the serving stack (launcher `start.sh` +
  `start-tp4-switchless.sh`, image build `Dockerfile.canary-roce`, adapters, boot.py,
  benchmarks, qeval). The site integration lives OUTSIDE in `../site/` (mirroring how
  the NVFP4 lane wraps its vendored knapcio upstream).

## Third-party sources (fetched at image build, sha256-checked by runtime/sources.manifest)

| component | pin | license |
|---|---|---|
| SGLang dsv4.1 branch python tree | `f80c91a4b97e52e17fbcff088aa6b47b6270f6b9` | Apache-2.0 |
| sglang-kernel / sgl-deep-gemm | 0.4.7 / 0.2.0 (aarch64 PyPI wheels) | upstream |
| b12x SG17 (RoCEnante) | rhys101 @ `66aa6bd` frozen archive | Apache-2.0 |
| b12x_next (routed MoE at EP1) | local-inference-lab/b12x @ `a7d7d29` + `b12x_next-compact-n64-m64.patch` | Apache-2.0 |
| roce_ring | FujitsuPolycom/sparkring @ `f16b5f4` + `roce_ring-sparkring-f16b5f4.patch` | Apache-2.0 |
| RoCEnante overlay patch | rhys101 sglang-rocenante.patch @ `66aa6bd` | — |
| base image | `lmsysorg/sglang:dev-dsv41@sha256:3dbc3130…` | — |

All URLs + sha256 live in `runtime/sources.manifest`; `scripts/fetch_runtime.sh` enforces
them at build. The pinned base digest was verified pullable from spark-0f0b on
2026-10-21 (`docker pull` succeeded; manifests present).

## NCCL transport (external prerequisite, NOT in this tree)

The launcher expects a patched NCCL at `NCCL_HOST_DIR` carrying the
`SWITCHLESS_RING_ONLY` marker (doctor/serve grep for it). This site points
`NCCL_HOST_DIR` at the cluster's existing four-PF build
`~/.local/lib/nccl-patched-4pf` (othexmr @ `94c2669` + one-line site GID patch, SHA
`8de68def…`, marker verified present on all four nodes 2026-10-21) — the SAME library
the GLM lanes and the retired sparkring lane use. No new NCCL build is required. The
four site flags (`NCCL_IB_EXTENDED_IPV4_GIDS/PRESERVE_PCI_DOMAIN/ROUTE_DIAGNOSTICS/
QPS_PER_CONNECTION`) go through `EXTRA_CONTAINER_ENV` per
`docs/switchless-ring.md` "Devices past the second are never advertised".

## Site modifications to upstream files — MUST be re-applied on re-vendor

One site modification is carried (2026-10-22): the **thinking-mode completion floor**
(`DSV41_MAX_NEW_TOKENS_FLOOR`). Everything else is still supplied by variable
(`IB_HCA`, `NCCL_HOST_DIR`, `PORT`, `SERVED_MODEL_NAME`, `WORKER_HOSTS`, `MODEL_DIR`, …).

| file | change | re-apply |
|---|---|---|
| `adapter/encoding_compat.py` | `DEFAULT_MAX_NEW_TOKENS_FLOOR = 16384`, `configured_max_new_tokens_floor()`, `request_enables_thinking_explicitly()`, and the floor branch at the top of `apply_request_max_tokens()` | re-add the four blocks after upstream's `DEFAULT_MAX_NEW_TOKENS` / `configured_max_new_tokens()`. The gate MUST stay explicit-only: an engine-default fallback there rewrites fixed budgets sent by measurement clients (2026-10-23 contamination) |
| `start.sh` | `-e "DSV41_MAX_NEW_TOKENS_FLOOR=${DSV41_MAX_NEW_TOKENS_FLOOR:-16384}"` under each of the two `DSV41_MAX_NEW_TOKENS` `-e` lines (head + worker) | re-add both lines; `start.sh`'s `-e` whitelist is why this cannot ride in the site env alone |
| `.env.example`, `.env.tp4.example`, `README.md` | `DSV41_MAX_NEW_TOKENS_FLOOR` knob + rationale | re-add the rows |
| `tests/test_max_new_tokens.py`, `tests/test_switchless_ring_integration.py` | floor boundary tests incl. the engine-default-is-ignored and install-ordering guards; floor env pinned in the launcher-invocation assertions | re-add; both suites must pass (`test_max_new_tokens` also runs at image build) |

Why: with thinking on, a request sized for a non-reasoning client (MMLU's
`max_tokens=256`) spends the whole budget in the reasoning segment and returns
`finish_reason=length` with empty content — a silent zero. The floor therefore
covers explicit thinking requests only: the engine-wide `SGLANG_DEFAULT_THINKING`
decides rendering, and letting it authorize a budget raise rewrote the fixed
`max_tokens` of kwarg-less measurement clients (llama-benchy's throughput
probes, tool-eval-bench `--spec-bench`), which stopped at EOS instead of at
their budget and made a `tg128` cell a different experiment. Receipts and
rejected alternatives: `../research.md`, 2026-10-22 "benchmark analysis:
thinking-mode budget starvation" and 2026-10-23 "measurement contamination".

## Update procedure

1. `git ls-remote https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks main`
2. Diff the pinned tree (`64b02dea`) against the new tip: launcher/image/adapter changes
   are adoptable after their in-image test suite passes (`Dockerfile` runs it at build).
3. Re-check `runtime/sources.manifest` pins changed — new commits there cascade into
   the image build (per-node rebuild + gates, a bench-recipe-update window).
4. Re-apply any site modifications recorded above.
