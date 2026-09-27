# VENDORED-AT.md — glm-v53-flash-4x-noswitch/upstream

This `upstream/` directory is a **detached, live checkout** of:

- **Repo:** `https://github.com/jnardiello/GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless`
- **Pinned commit:** `080fe09` (E29 recipe: R10/sparkring image, DFlash2 k=7/3,
  SparkCache+SIRCL vendored under `third_party/`, patched NCCL 2.30.7-1 flow)
- **Vendored:** 2026-09-26 (moved from the standalone workstation checkout
  `C:\Users\sdrew\code\glm-4x-noswitch`, `.git` deleted); committed into the
  `glm-v53-flash-4x-noswitch` branch of `taoofshawn/spark-recipes` at commit
  `9cd37a0`.

## Local modifications carried on top of the pin (do NOT overwrite on update)

Full details and the update procedure: `../research.md` §2.

1. `scripts/node/bootstrap/versions.env` — SITE RE-PIN: KERNEL
   `6.17.0-1032-nvidia`, DRIVER `580.173.02`, `OS_RELEASE=""` (site runs mixed
   24.04.4/24.04.5).
2. `scripts/node/nccl/SHA256SUMS` — this site's adopted NCCL rebuild:
   `afe5f48626284eae89988516e450c7f20cc303904ba4b7083b88aa3ca1e9b85f` (not
   bit-reproducible; upstream's hash will not match).
3. `scripts/node/nccl/build.sh` — SITE FIX: re-asserts the vendored NCCL
   `PATCH_FILE` after `tp4_load_env` (cluster.env's PATCH_FILE clobbers it).
4. `CHANGELOG.md` — dated site section `2026-09-26 — shawndo 4x DGX Spark site`
   at the bottom.
5. `scripts/**`, `cluster.env`, `cluster.env.example` — SITE PATHS: `$HOME/tp4` →
   `$HOME/.local/tp4`, `~/nccl-patched` → `$HOME/.local/lib/nccl-patched`,
   `~/vllm-cache` → `$HOME/.cache/tp4-vllm-cache`, weights → HF-cache snapshot paths
   (details: `../research.md` §2.5). Rewrites are mechanical; re-apply to any file an
   update touches that references the old paths.

## Refresh procedure

Diff the pin against the upstream repo's new HEAD, adopt only genuinely-upstream
changes, re-verify the 4 mods survived, run `./scripts/check.sh` (offline, needs
`Jinja2==3.1.6`), then update the pin here and add a dated entry to
`../research.md` §5. Site config lives outside this tree in `../noswitch-prep/`.
