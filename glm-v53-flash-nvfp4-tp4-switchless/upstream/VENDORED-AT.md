# VENDORED-AT

- **upstream:** `github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4`
- **pin:** commit `770d1153062aa916b06591411c61d5c593ea0f03` (2026-09-29 tip; superset of the
  `3edfbc9` release — the range `3edfbc9..770d115` adds only docs plus the switchless
  transport itself, verified by diffing the compare patch before vendoring)
- **vendored:** 2026-09-29, cloned then `.git` stripped (detached; adopted, not tracked)
- **provenance chain:** knapcio's stack consolidates tonyd2wild's image/overlays
  (`ghcr.io/tonyd2wild/vllm-glm53-flash` base, vLLM `487ecf187`), alexellis's launch line,
  jnardiello's/FujitsuPolycom's scheduler ideas, and b12x's RoCEnante RDMA collectives.
  See upstream `CREDITS.md`.

## Site additions (files WE added — everything upstream is unmodified)

| file | what |
|---|---|
| `VENDORED-AT.md` | this file |
| `profiles/stock-nvfp4.env` | site-added serving profile: sources upstream `profiles/current.env`, then strips the lossless8/fp8-drafter-only switches and applies the shawndo site serving values (1M ctx, 20 GiB/rank KV pin, port 8000, `glm-5.3-flash`) |

The recipe's premise is the stock `nvidia/GLM-5.3-Flash-NVFP4` checkpoint with no
conversion step, so unlike the FP8 glm-v53-flash-fp8-tp4-switchless recipe there is no lossless8/drafter-fp8
build to wire in. The site `.env` lives OUTSIDE upstream (see `../site/env.switchless`)
because upstream's `.gitignore` ignores `.env`; bring-up copies it to `upstream/.env` on
the head (ignored file, node tree stays pristine).

### Site modifications (upstream files changed — MUST be re-applied on re-vendor)

| file | mod | why | status |
|---|---|---|---|
| `scripts/transport.sh` | **#1 (REVERTED, Phase E 2026-10)** one line in the `switchless` branch of `transport_args`: `-e NCCL_SKIP_TREE_CONNECT=1` | (Historical, staged 2-PF transport) The then-pinned site NCCL build (`~/.local/lib/nccl-patched`, SHA `afe5f486…`, the josephdrose/SparkRing patch family proven by the FP8 recipe) implemented ring-only tree/PAT skipping under `NCCL_SKIP_TREE_CONNECT`, not the four-PF build's `NCCL_SWITCHLESS_RING_ONLY`. | **Removed.** The Phase-E four-PF build is ring-only under `NCCL_SWITCHLESS_RING_ONLY` (which the upstream render already sets); its hardened listener path also parses `NCCL_SKIP_TREE_CONNECT` as a compatibility alias, but the explicit flag is the four-PF contract, not the alias. |
| `scripts/transport.sh` | **#2 (REVERTED, Phase E 2026-10)** in the same `printf`: `NCCL_IB_MERGE_NICS=1` → `=0` (restored to the upstream value) | (Historical, staged 2-PF transport) With two separate virtual devices, NCCL pairs channel c → vdev (c mod 2) with no subnet knowledge; the rank3→rank0 channel got 10.10.1.x at rank 0 against 10.10.4.x at rank 3 and QP RTR timed out (`ibv_modify_qp` 110, first boot 2026-10-10). The merged vdev let the SparkRing lib's subnet-aware routing pick the correct PF. | **Restored to upstream `=0`.** The four-PF contract requires separate dual-PF listeners + subnet-aware routing (and the four-PF lib's hardened switchless path hard-rejects `MERGE_NICS != 0`). |

As of Phase E (research.md changelog 2026-10), the functional lines of the `switchless`
branch of `transport_args` are identical to upstream pin `770d115` — both site mods were
transient staged-transport accommodations and are gone (only the explanatory comment block
remains as a site annotation). On re-vendor at a new tip, verify the switchless render
sets `NCCL_SWITCHLESS_RING_ONLY=1` and `NCCL_IB_MERGE_NICS=0` with no site env lines.

## Update procedure

1. `git ls-remote` the upstream repo; diff `770d115..HEAD` (or re-vendor at the new tip).
2. Classify every change. Anything touching `profiles/current.env` requires a **stock-compat
   review** of new switches (see the strip list + watch items in
   `../research.md` §3) before inheriting them.
3. Re-check `docs/switchless.md` and `scripts/check_switchless_nccl.py` — the switchless
   transport contract may tighten (NCCL patch list, four-PF requirement).
4. Re-check the serving receipts (`docs/results/`) — numbers in `../README.md` /
   `../research.md` are pinned to this commit's claims.
