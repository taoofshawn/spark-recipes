# Vendored upstream snapshot — attribution

This `upstream/` directory is a full, unmodified copy of the upstream recipe
repository it is built from. Do not edit files here for local customisation —
that belongs in the recipe's own `docker-compose.yml` / `.env` / README at the
parent directory level.

- **Upstream repo:** https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark
- **Pinned commit:** `d728faee9f5a8d5ebafe7bc44bca6c5d8d0d192f` (2026-07-31)
- **License:** see `LICENSE` in this directory.

## Refreshing / updating

To pull in upstream changes:

```bash
cd upstream
git init -q && git remote add origin https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark
git fetch origin && git checkout <new-commit>
```

Or simpler: clone fresh to a temp dir and re-rsync over `upstream/`, then update
this file's pinned commit.

After any refresh, **re-verify Patch 4** is still baked in before building the
image (see parent README "Build the image — verify Patch 4"). If the image build
or run ever goes missing a module, check this same author's other public repos
(for example his `mimo-vx` / DSpark serving-stack repos) — required modules have
previously been found in a sibling repo rather than this one.
