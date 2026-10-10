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
conversion step, so unlike the FP8 4x-noswitch recipe there is no lossless8/drafter-fp8
build to wire in. The site `.env` lives OUTSIDE upstream (see `../site/env.switchless`)
because upstream's `.gitignore` ignores `.env`; bring-up copies it to `upstream/.env` on
the head (ignored file, node tree stays pristine).

### Site modifications (upstream files changed — MUST be re-applied on re-vendor)

| file | mod | why | how to re-apply |
|---|---|---|---|
| `scripts/transport.sh` | **#1** one line added to the `switchless` branch of `transport_args`: `-e NCCL_SKIP_TREE_CONNECT=1` (marked with a `SITE MOD` comment block) | The site NCCL build (`~/.local/lib/nccl-patched`, SHA `afe5f486…`, the josephdrose/SparkRing patch family proven by the FP8 recipe) implements ring-only tree/PAT skipping under `NCCL_SKIP_TREE_CONNECT`, not the four-PF build's `NCCL_SWITCHLESS_RING_ONLY`. Without it, NCCL init on a switchless ring attempts tree connects that cannot succeed (the FP8 launcher always sets this env). `transport.sh` rejects `NCCL_*` keys in `EXTRA_ENV`, so there is no non-mod way to pass it. | After re-vendoring, add `-e NCCL_SKIP_TREE_CONNECT=1` to the `printf` in the `switchless` branch of `transport_args` (switchless case only; the switched render stays byte-identical to upstream). |
| `scripts/transport.sh` | **#2** in the same `printf`: `-e NCCL_IB_MERGE_NICS=0` → `=1` (marked in the same `SITE MOD #2` comment block) | With two separate virtual devices, NCCL pairs channel c → vdev (c mod 2) with no subnet knowledge; the rank3→rank0 channel got 10.10.1.x at rank 0 against 10.10.4.x at rank 3 and QP RTR timed out (`ibv_modify_qp` 110, first boot 2026-10-10). The FP8 launcher's proven value is the default (merged vdev + subnet-aware routing picks the correct PF per connection). | After re-vendoring, change `NCCL_IB_MERGE_NICS=0` to `=1` in the switchless `printf`. **Revert to `=0` in Phase E** (the four-PF build requires separate dual-PF listeners). |

The Phase-E four-PF NCCL upgrade (research.md) supersedes this mod: once the rebuilt
four-PF library is pinned, `NCCL_SWITCHLESS_RING_ONLY` works natively and this site line
becomes redundant (drop it then — othexmr's docs call the alias-only tree-skip bypass
weaker than the four-PF build's strict listener admission).

## Update procedure

1. `git ls-remote` the upstream repo; diff `770d115..HEAD` (or re-vendor at the new tip).
2. Classify every change. Anything touching `profiles/current.env` requires a **stock-compat
   review** of new switches (see the strip list + watch items in
   `../research.md` §3) before inheriting them.
3. Re-check `docs/switchless.md` and `scripts/check_switchless_nccl.py` — the switchless
   transport contract may tighten (NCCL patch list, four-PF requirement).
4. Re-check the serving receipts (`docs/results/`) — numbers in `../README.md` /
   `../research.md` are pinned to this commit's claims.
