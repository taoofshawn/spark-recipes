# research.md — glm-v53-flash-entrpi: maintenance notes

Working notes for future update/maintenance sessions on this recipe. The
README is the deploy doc; this file is the memory.

## Current state (2026-09-05)

- Adopted 09-04 (v2.3-tier1, branch `glm-v53-flash-entrpi`), fully
  provisioned on the cluster (image pulled, weights hardlinked, MXFP8
  drafter downloaded on both nodes). NOT YET BOOTED on this cluster —
  DS4 vision currently owns :4000 and only one recipe serves at a time.
- **09-05 update pass: NO upstream changes.** Entrpi kit HEAD still
  63f254f (v2.3-tier1), vLLM fork still f223ff9, `docs/FINDINGS.md` +
  `docs/COMPARISON.md` unchanged, image digest confirmed unmoved
  (`sha256:44b2dbaf…`, published 2026-09-02). Nothing to adopt; the
  only new signals are watchlist items below.
- Sibling lane: `glm-v53-flash-miaai` carries the hotfix-maintained
  variant (same weights, different runtime). Only one of the two serves
  at a time.

## Environment facts (do not rediscover)

- Each DGX Spark exposes ONE GPU; TP=2 spans both nodes. Worker first,
  leader ~35 s later. Port 4000 (repo convention; Entrpi's own default
  is 8000 — anything copied from their docs must have the port fixed).
- Image: `ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:v2.3-tier1` —
  ENTRYPOINT `/opt/nvidia/nvidia_entrypoint.sh` (exec's its args, so
  compose `command: bash -lc …` works as written).
- vLLM path inside the image: `/usr/local/lib/python3.12/dist-packages`
  (same as the miaai lane — this is the local-inference-lab preview
  lineage, NOT /opt/venv or /opt/env).
- Weights: flat dir `/home/sdrew/models/glm53-exl3` (139 files,
  hardlinked to the Mia-AiLab HF snapshot blobs — nlink=2, zero extra
  disk; do NOT `rm -rf` the HF cache or these links break).
- Drafter: `/home/sdrew/models/glm53-dflash2-mxfp8` (MXFP8, 1.20 GiB,
  CC BY-NC-ND — never redistribute the bytes).
- Cache root: `/home/sdrew/.cache/glm53-entrpi`, mounted at `/cache`
  (`HF_HOME=/cache/huggingface`) + jit subdirs. Created at first boot.

## The profile system (the main thing to know)

This lane's value is Entrpi's **validated configuration matrix** — the
README "Choosing a configuration" table. Every row is a set of `.env`
values (all drive through the compose command block into `vllm serve`
args). Defaults = their validated production row:

| row | key values |
|---|---|
| **Default** (long docs, few users) | MAX_LEN=524288, MAX_SEQS=4, MNBT=8192, KV fp8_ds_mla, SPEC=dflash, KV_CACHE_MEMORY=14.4e9 |
| **Snappy chat + uploads** | + MIXED_PREFILL_DECODE_WEIGHT=1.0, MIXED_PREFILL_CAP=512 |
| **1M requests** | MAX_LEN=1048576, KV_DTYPE=nvfp4_ds_mla, VLLM_NVFP4_MLA_DYNAMIC_SCALE=1, MNBT=4096 |
| **Agentic (12–16 streams)** | MAX_LEN=131072, MAX_SEQS=12, MNBT=4096, SPEC=none, MIXED_PREFILL_DECODE_WEIGHT=1.0, MIXED_PREFILL_CAP=512 |
| **Unquantized KV** | MAX_LEN=131072, KV_DTYPE=, ATTN_BACKEND=, SKIP_MM_PROFILING=0, MAX_SEQS=6 |
| **MTP fallback** | MTP=4 (drafter not loaded; ~20% slower decode) |

Rules:
- Change ONE row at a time; a profile change is a `.env` edit + full
  redeploy (worker down, leader down, worker up, leader up ~35 s later).
- `KV_CACHE_MEMORY` empty → 14.4e9 (the launcher's conversion is
  reproduced in the compose block). "auto" → vLLM budgeting. Lower =
  smaller pool (~90k tokens/GB); raising past 14.4e9 without re-measuring
  floors collapses memory headroom (v2.1 evidence: 13.4e9 → 2.26 GiB
  floor).
- `BLOCK_SIZE=2304` auto-bumps to 4608 with fp8 KV (boot log "Setting
  attention block size to 4608") — expected.
- 1M-row uses `nvfp4_ds_mla` (not our default fp8_ds_mla) AND requires
  `VLLM_NVFP4_MLA_DYNAMIC_SCALE=1` or the engine refuses to boot.

## Known-benign boot/log lines (verified upstream; re-verify on our boot)

| line | verdict |
|---|---|
| `Setting attention block size to 4608` | fp8 KV auto-bump from 2304 |
| `GID auto-detect: NCCL_IB_GID_INDEX=N` | our compose's sysfs detect; the ERROR dump (no RoCE v2 GID) instead = fail |
| `hybrid APC groups: ...` (v2.3 image) | fine-grained prefix reuse (512-token hash) engaged — normal |
| `kit update available` | N/A in compose port (no install.sh/launcher) — ignore |

## Upstream watchlist (research on every update pass)

**Entrpi/glm-5.3-flash-exl3-2x-spark** (kit) — HEAD 63f254f (09-02),
v2.3-tier1. Reviewed 09-05: no new commits, no release. Watch:
- **LATEST / VERSION / new tags** — an eventual "final release" is
  pending (referenced 09-03 on the forum). A new tag = re-run the whole
  adoption: digest-pin, verify the serve args still match, re-check
  FINDINGS §8 ("what does NOT help") for the new release.
- **Issue #4 (09-04, open)** — NVIDIA **driver generation** vs UMA
  headroom: 610.43.02-open costs ~4 GiB more unified memory at init and
  is nondeterministic (0/4 launches); 580.173.02 passes first try.
  Both our sparks are on **580.173.02** — the good side. If a DGX
  update lands 610, expect boot nondeterminism + less KV headroom (same
  root cause as MiaAI #114). Verify `nvidia-smi --query-gpu=driver_version`
  at boot.
- **install.sh features we deliberately don't use** (update-check,
  rollback, MEM_USED_MAX_GB preflight, hotfix overlay dirs): keep out of
  the compose port unless a knob is needed; the memory preflight
  (`<6 GB system memory in use`) is the one worth replicating by hand:
  `free -m` before launch.

**Entrpi/vllm-glm-5.3-flash-spark** (fork) — HEAD f223ff9 (09-02). The
per-M dispatch (`VLLM_B12X_MXFP8_MAX_M`), rowwise-fp8 draft head, and
breakable-graph machinery are baked in the v2.3-tier1 image. Any fork
commit that changes kernel contracts will surface as an image update,
not a patch.

**Forum watch items (09-05 sweep; mostly shared with the miaai lane):**

- **Spec-decode TTFT alternation (382099)** — long-prefill TTFT
  alternates ~2× (14.6s ↔ 19-27s) after a mixed-shape workload, strictly
  every other run, on EXL3 + DFlash2 k=7 + fp8_ds_mla + 2×GB10 (measured
  on entrpi **v2-glmnext**, sibling tag). Spec-generic (MTP too; SPEC=none
  clean). Restart clears it. **Benchmark discipline: restart between
  configs and record cumulative `vllm:prompt_tokens_total`; GPU util%
  lies on stall-bound steps (96% @36W = parked on syncs; power is the
  honest signal).**
- **Vision+text concurrent crash (381350 post 273, 09-04)** —
  `CUDA_ERROR_NOT_PERMITTED` at DeepGEMM mhc_pre_tilelang when an image
  request co-schedules with a cached text request; config dump in the
  trace is entrpi-style (exl3, dflash2 k=7, fp8_ds_mla, instanttensor).
  Fatal, not recoverable in-place. Real risk for concurrent vision+text
  on this lane (vision tower loaded). Test this combo before relying on
  it; "newer image" was the reporter's next step — unresolved in thread.
- **Degenerate output / acceptance 0.00 (381350 post 253)** — draft
  acceptance collapsing to 0.00 + '!!!!!' thinking streams; sglang
  issue 36669 reference is for the NVFP4 lane's marlin fix, does NOT
  carry to exl3. Watch item.
- **1 tok/s stutter quirk (381350 posts 333/334)** — GLM-5.3 quant
  stacks stutter to ~1 tok/s for 10-20s periodically (real 600k session:
  20-25 tok/s baseline). Seen on NVFP4 + Intel W4A16; may appear here.
  Informational.

**Parallel/community lanes (info only — different images):**
- `Plaaasma/glm53-flash-dual-dgx-spark` (forum 382120) — NVFP4 KV at
  288 B/token (2.2M pool), and the **EXL3_TEMP_ROWS_FUSED=192 +**
  fixed-shape-gather mixed-step fix (~87 host round-trips/step killed;
  830→480ms mixed step). That fix targets the MIA-AI kit's stock
  `apply_exl3_fused_moe` (hard-coded `TEMP_ROWS_FUSED = 128` in exl3.py,
  no env var). **Our entrpi v2.3-tier1 exl3.py is a different file**
  (105 KB, Sep 2; uses `_load_b12x_fused_moe`/`standard_fused_moe` —
  b12x path, not the stock scratch-cache path) — the temp-rows knob does
  NOT apply here. Leave it.
- `Reederey87/glm53-flash-exl3-2x-dgx-spark` — same day-0 base image
  lineage, LONG_PREFILL_TOKEN_THRESHOLD=1792 (their MNBT 3584/2 — the
  MNBT/2 rule), unconditional Reasoning-Effort template. Prefix-cache
  fork patches are not vendorable to the public image.
- `punkjazz-labs/glm-5.3-flash-exl3-4x-dgx-spark` — TP4; #86-rightsize
  stall bisect (we ship stock on miaai; entrpi has no indexer-workspace
  knob — N/A).

## Revision-pin procedure (for any bump)

Pins live in `.env` (`IMAGE@sha256`, `DFLASH_REVISION`; weights are a
host dir, not a pin). Steps:
1. Image: `docker manifest inspect` (or pull) the new tag, replace the
   digest in `.env`. THEN verify the four serve-arg knobs still exist
   (a new image may retire a flag → boot fails vs silently ignores).
2. Drafter: if Entrpi changes `DFLASH_REPO` (the MXFP8 copy only has
   `62f758c0` and a 09-02 HEAD `610aa967` that DROPS lil.yaml — all
   serving files byte-identical, so either rev serves; pin `62f758c0`
   as shipped), update `.env` + re-download on both nodes.
3. Weights: only if upstream switches quant/checkpoint — currently
   brandonmusic = Mia-AiLab bytes; re-point only the hardlink source.
4. TFTR: `--load-format instanttensor` requires the flat dir; do NOT
   point at an HF hub snapshot (symlink layout) — materialize hardlinks
   as the README does.

## When to use this vs glm-v53-flash-miaai

- **entrpi**: self-contained image (fixes baked), ring draft-KV (+32%
  pool vs miaai slot-share), fine-grained 512-token prefix reuse, b12x
  fused MoE. The "permanent fix" lane. Default config maxes at 524k;
  1M needs the nvfp4 row.
- **miaai**: hotfix-audited (two+one fail-closed boot patches), 1M
  native default, same-image measured tuning (threshold 3584, gate
  2048). Both same weights/served name/port. Switch = tear down one
  recipe, start the other. Practical rule: agentic/multi-stream → entrpi
  agentic row or miaai; 1M-context-first → miaai.
