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

**Zero upstream-file modifications by design** — this recipe's premise is the stock
`nvidia/GLM-5.3-Flash-NVFP4` checkpoint with no conversion step, so unlike the FP8
4x-noswitch recipe there is no lossless8/drafter-fp8 build to wire in. The site `.env`
lives OUTSIDE upstream (see `../site/env.switchless`) because upstream's `.gitignore`
ignores `.env`; bring-up copies it to `upstream/.env` on the head (ignored file, node
tree stays pristine).

## Update procedure

1. `git ls-remote` the upstream repo; diff `770d115..HEAD` (or re-vendor at the new tip).
2. Classify every change. Anything touching `profiles/current.env` requires a **stock-compat
   review** of new switches (see the strip list + watch items in
   `../research.md` §3) before inheriting them.
3. Re-check `docs/switchless.md` and `scripts/check_switchless_nccl.py` — the switchless
   transport contract may tighten (NCCL patch list, four-PF requirement).
4. Re-check the serving receipts (`docs/results/`) — numbers in `../README.md` /
   `../research.md` are pinned to this commit's claims.
