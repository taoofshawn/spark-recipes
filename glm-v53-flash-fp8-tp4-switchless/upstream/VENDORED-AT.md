# VENDORED-AT.md — glm-v53-flash-fp8-tp4-switchless/upstream

This `upstream/` directory is a **detached, live checkout** of:

- **Repo:** `https://github.com/jnardiello/GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless`
- **Pinned commit:** `ed365a6` (E36 recipe: E31-MB memory-bounded layer + SparkCache disk
  capacity (200 GiB cap / 160 GiB watermark via the ram-budget connector), E35
  confidence-based verify length, E36 INT8 W8A16 shared lm_head; same pinned image
  digest `sha256:0d40…` as the previous pin)
- **Vendored:** 2026-09-26 (moved from the standalone workstation checkout
  `C:\Users\sdrew\code\glm-4x-noswitch`, `.git` deleted); committed into the
  `glm-v53-flash-4x-noswitch` branch of `taoofshawn/spark-recipes` at commit
  `9cd37a0`. Refreshed to `ed365a6` on 2026-09-30 (branch
  `glm-v53-flash-fp8-tp4-switchless-sparkcache-diskbound`).

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
6. `scripts/render-netplan.sh` — SITE EXTENSION (shawndo, 2026-10-10): opt-in
   `FABRIC_P2_VIEWS` cluster.env flag (default 0 = upstream behavior byte-for-byte).
   When 1, the second PCIe views (FABRIC_IFACES positions 3/4) are ADDRESSED — one
   address per node per cable subnet, last octet = f-octet + 2 — and the yaml emission
   order groups each P2 view after its parent f view. This matches the NVFP4 lane's
   Phase-E-deployed live netplan (shared fabric), so `render-netplan.sh --check` and
   `deploy-host.sh` regeneration are safe again; the old 2-address output would strip
   the P2 addresses and break the NVFP4 lane. Re-apply on update: the flag plumbing is
   the three `FABRIC_P2_VIEWS` blocks (doc header, validation, `render()` emission
   order + address rule); see `../noswitch-prep/cluster.env` `FABRIC_P2_VIEWS` comment.
7. `scripts/nccl_gid_check.py` — SITE FIX (shawndo, 2026-10-10, found by the gated
   T1 four-PF bring-up): the addressed-netdev allowlist was hard-coded to the FIRST TWO
   `FABRIC_IFACES` entries (the upstream 2-PF contract), so with the four-PF HCA list
   the P2-view HCAs could never pass (`no usable automatic IPv4-mapped RoCE v2 fabric
   GID` at the launcher preflight — every launcher boot attempt failed before Docker).
   Fix: the allowlist is the FIRST `len(hcas)` entries — N=2 reproduces upstream
   behavior byte-for-byte, N=4 admits the P2 views. 2-PF control re-verified PASS on
   spark-0f0b after the fix. Re-apply on update: the one slice + comment block in
   `validate_selection`.
8. `scripts/node/overrides/vllm/v1/worker/gpu/structured_outputs.py` — SITE OVERRIDE
   (mod 8, 2026-10-10): exact image copy + the bare `assert num_masks == len(mapping)`
   in `apply_grammar_bitmask` replaced by a state-dumping RuntimeError
   (GRAMMAR-DIAG-WORKER). Re-apply: re-extract the image file, re-apply the assert
   block (see the file header).
9. `scripts/node/overrides/vllm/v1/structured_output/__init__.py` — SITE OVERRIDE
   (mod 8, 2026-10-10): exact image copy + a per-call row-accounting INFO line in
   `grammar_bitmask` (GRAMMAR-DIAG-SCHED, gated by `VLLM_GRAMMAR_DIAG=1`). Re-apply:
   re-extract + re-apply the DIAG block before the `return bitmask_tensor.numpy()`.
10. `scripts/node/overrides/vllm/distributed/device_communicators/shm_broadcast.py` —
   SITE OVERRIDE (mod 10, 2026-10-10): exact image copy + a pickle-boundary
   observation in `enqueue` (GRAMMAR-DIAG-PICKLE, gated by `VLLM_GRAMMAR_DIAG=1`).
   Re-apply: re-extract + re-apply the block at the top of `enqueue`.
11. `scripts/node/experiments/e03/e36-lm-head-w8a16/model_runner.py` (THE effective
   runtime model_runner — the E36 cluster.env append substitutes the e35-runner-k
   mount path with this file) — SITE GRAMMAR-FIX (9b, 2026-10-10): (a) the E35
   runner-k trim (`e35_trim`) records `(req_id, k)` and grammar steps
   (`has_structured_output_requests`) never trim; (b) `sample()` re-slices the
   engine-built grammar bitmask to the verified-prefix layout (first k draft rows +
   bonus row) whenever a trim did fire. Root cause + evidence: `../research.md`
   2026-10-10 changelog. Re-apply on update: re-diff against the new image's file,
   re-apply the three `SITE GRAMMAR-FIX` blocks + the `SITE GRAMMAR-DIAG` blocks in
   `sample()`/`prepare_inputs`, refresh `SHA256SUMS`.

## Refresh procedure

Diff the pin against the upstream repo's new HEAD, adopt only genuinely-upstream
changes, re-verify the 4 mods survived, run `./scripts/check.sh` (offline, needs
`Jinja2==3.1.6`), then update the pin here and add a dated entry to
`../research.md` §5. Site config lives outside this tree in `../noswitch-prep/`.
