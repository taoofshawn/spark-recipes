# research.md — glm-v53-flash-miaai: maintenance notes

Working notes for future update/maintenance sessions on this recipe. The
README is the deploy doc; this file is the memory. Anything not needed to
maintain or re-derive the recipe was pruned — git history retains the full
adoption narrative (see `git log glm-v53-flash-miaai/`).

## Current state (2026-09-02)

- Deployed and validated on the 2-node cluster (spark-0f0b / spark-6d14).
  Serves as `glm-5.3-flash` on :4000, 1M ctx, DFlash2 k=7. Boot-log health
  markers and what they mean: see README "Boot-log health markers".
- Single-stream ~24-28 tok/s decode, 4-way aggregate ~59 tok/s (upstream
  documents 33-74 on their kit). Prefill ~600-750 tok/s.
- The DS4-vision recipe (`deepseek-v4-flash-vision-miaai`) shares this
  cluster and port; only one may serve at a time.

## Environment facts (do not rediscover these)

- Each DGX Spark exposes ONE GPU; TP=2 always spans the two nodes. A
  single-node TP=2 diagnostic boot is impossible (`local_rank <
  device_count` assert).
- Sparks have no `hf` CLI, no host `huggingface_hub`, pip is PEP-668-locked:
  `python3 -m venv /tmp/hfvenv && /tmp/hfvenv/bin/pip install huggingface_hub`
  for any cache work.
- The GLM image bakes `HF_HOME=/root/.cache/huggingface` — the host cache
  mounts there (the DS4/Anemll image uses `/cache/huggingface` instead).
- `docker-compose.override.yml` is AUTO-LOADED when no `-f` is given. If a
  boot behaves strangely, check for a leftover override file on BOTH nodes.
- Worker first, leader ~35 s later. During the leader's ~6 min weight load
  the worker spams `TCPStore.cpp sendBytes failed` / NCCL heartbeat warnings
  — wait-window noise. Judge boot health by the LEADER log.
- GPU contention: `docker compose down` the other recipe on BOTH nodes
  before starting one (a surviving old leader container causes GPU
  allocation crash-loops).
- Worker-local debugging trick: `docker exec glm53-exl3-miaai ps -eo` —
  `VLLM::EngineCore` + `VLLM::Worker_TP0` alive = healthy; a couple of
  defunct `python3` zombies (ppid 1) are cosmetic either way.

## The KV hotfix — `hotfix_kv_check_glm5.py`

The GHCR image's baked vLLM (`vllm/v1/core/kv_cache_utils.py`) predates the
overlay's DFlash2 KV work. Three defects, all inside the image, none
reachable from compose flags:

1. **Stale builder branch**: `_get_kv_cache_groups_glm5_next`'s else branch
   is STANDALONE — leaves the drafter at block_size=16 / window 16384, so a
   1M request needs ~1306 block ids vs a ~621-block pool. Symptom: boot
   ValueError (34.15 GiB) or silent capacity starvation
   (`waiting{reason="capacity"}` forever, GPU ~0%, small requests fine).
2. **Detector/builder mismatch**: `_glm5_next_tensor_layout` rejects
   `page_size_padded` specs the builder emits.
3. **No glm5 branch in the capacity check**
   (`_max_memory_usage_bytes_from_groups`) — the stock "GPU KV cache size"
   line is window-unaware and not an honest capacity figure (upstream
   issue #94).

The hotfix edits `kv_cache_utils.py` in place at boot (patch loop in
docker-compose.yml, before `vllm serve`):

- **A**: detector accepts drafter specs with `page_size_padded == mla_page`.
- **B**: early glm5 capacity branch — allocator-consistent per-block math,
  window-bounded cdiv, mamba groups excluded (length-independent SSM state).
- **E** (load-bearing): replaces the STANDALONE builder branch with the
  overlay's padded slot-share rescale (block=64, page padded to mla_page →
  313 blocks/req → 1M schedulable).
- **D**: one-line debug dump when the detector still returns None.

Fail-closed by design: it preflights every anchor and aborts the boot
(nonzero exit) if the image text no longer matches. That failure is the
signal the image changed — do not serve until the hotfix is re-derived or
retired.

Why upstream doesn't hit this: their validated boots use locally built
images with builder/detector/accounting in sync (their issue #97 documents
GHCR-vs-rebuild drift).

## Known-benign boot/log lines (all verified on this cluster)

| line | verdict |
|---|---|
| `SymmMemCommunicator: Device capability 12.1 not supported` | expected on GB10; PYNCCL fallback |
| `Custom collectives are disabled because this multi-node ...` | expected for TP across 2 nodes |
| `Sparse MLA impl has no dense-MHA prefill path; using the top-k MQA path only` | by design; only SM12x kernel |
| `Draft model DFlash2Qwen3ForCausalLM does not support external multimodal embeddings` | by design; drafter is text-only |
| `Disabling fine-grained prefix-cache hits ... KpoolTailManager` | upstream default; 3584-token block-aligned hits still work |
| sampling defaults overridden by `generation_config.json` (temp 1.0, top_p 0.95) | checkpoint ships tuned defaults; clients may override |
| one-time Triton JIT `_topk_topp_kernel` latency spike | first sampled request only; cached after |
| worker TCPStore/NCCL spam during leader load | rendezvous noise; judge by leader log |
| defunct `python3` zombies (ppid 1) | cosmetic; verify workers alive via `ps -eo` |

## Revision-pin procedure (use for any model/drafter bump)

Pins live in `.env` (`MODEL_REVISION`, `DFLASH_REVISION`) — both nodes'
caches must hold exactly those revisions (`HF_HUB_OFFLINE=1` means vLLM
uses whatever snapshot dir resolves first).

1. Diff old vs new revision trees via the HF API
   (`https://huggingface.co/api/models/<id>/tree/<rev>`; compare LFS oids).
   The DFlash2 repo has a pattern of same-day "Checkpoint update" commits
   with byte-identical serving files — often only the README differs and
   re-pinning is a no-op.
2. If weights differ: proper `snapshot_download` in the /tmp venv on BOTH
   nodes (a `cp -a` of an existing snapshot only works for byte-identical
   trees; symlinks to ../../blobs must survive — verify none broken).
3. If the tree matches but the snapshot dir for the pinned rev doesn't
   exist, materialize it by `cp -a` of the cached snapshot, then
   rglob-verify no broken symlinks.

## Upstream watchlist (research these on every update pass)

**MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks** (source of image + profile;
HEAD eb0469fb 2026-09-02, reviewed 2026-09-02):

- **#110 + PR #112** (`LONG_PREFILL_TOKEN_THRESHOLD`) — an unset threshold
  lets one long chunked prefill claim the whole per-step token budget and
  FREEZE every other session for its duration (measured: 440 s at 325k,
  MNBT=2048). PR: `--long-prefill-token-threshold 1024` fixes it for ~10%
  on the long prefill alone. **Directly relevant to this cluster's two-
  session usage** (our 2026-09-02 "compact crawls, other session stalls"
  observation is the same mechanism at ~75k). ADOPT CANDIDATE — but NOT in
  the image's serve path yet (start.sh has no flag plumbing at review
  time); pass it via `GLM53_EXTRA_ENV`-style launch arg or wait for
  upstream merge; measure TTFT of the other session before/after.
- **#108** — the `padded slot-share ... exact-fit page mismatch
  draft_bytes/token=2048` line we treat as a HEALTH MARKER is upstream
  issue: ~5.2% structural padding per co-owned page, reproduced every
  boot. Cosmetic-but-real; if upstream lands a rescale, re-check our
  marker text.
- **#106** — prefix cache stops hitting GLOBALLY after ~50-60 min of
  concurrent agentic load; only a restart restores hits. Watch for it
  here (symptom: `vllm:prefix_cache_hits_total` frozen + full re-prefills
  on byte-identical prompts at low KV usage).
- **#113** — 4-Spark DCP=4 data: on LONG agentic contexts k=3 beat k=7
  (81.4 vs 64.0 aggregate x4, accept 50% vs 27%); k=7 remains right for
  structured/code output (high-accept regime). Our k=7 default targets
  coding-agent traffic — if long-context multi-stream becomes the norm
  here, A/B k=3-4 on OUR kit (their numbers are a 4-node shape).
- **#111** — changing reasoning_effort still invalidates the whole prefix
  cache (effort line sits at ~token 8; #63 fixed only the thinking toggle).
  Clients cycling effort levels pay cold re-prefills.
- **#97** — GHCR vs local-rebuild drift. A re-synced GHCR image is the
  retirement path for our hotfix: re-test WITHOUT the hotfix first.
- **#94** — honest KV-capacity boot line. If merged into the image, it
  supersedes our fix B.
- **#102** — `EXL3_FAT_KERNEL=1` + MNBT=7168 head-rank silent death at
  multimodal warmup (did NOT reproduce here). If a future boot dies
  without a traceback right after CUDA-graph capture: set
  `EXL3_FAT_KERNEL=0` and `MAX_NUM_BATCHED_TOKENS=2048`.
- **#86** — `GLM53_INDEXER_WORKSPACE=rightsize` reclaims ~4.5 GiB for KV
  (+26-28% capacity). We ship `stock`. Flip if KV pressure appears.
- **#88 / #85 / PR #100 / #99 (KV disk tier)** — read before touching
  SPEC_METHOD / max_num_seqs / KV-tier knobs.
- New image tags: re-run the revision-pin procedure and expect hotfix
  anchor drift (fail-closed boot = re-derive time).

**Entrpi** (https://github.com/Entrpi — parallel DGX-Spark serving line;
v2.3-tier1 as of 2026-09-02):

- **`vllm-glm-5.3-flash-spark`** — vLLM fork with glm5_next port + sm121
  fixes; active (2026-09-02: b12x MXFP8 drafter GEMM per-M dispatch,
  mamba spec-state ring). The most likely PERMANENT fix for the
  builder/detector/accounting drift; if self-consistent, prefer rebasing
  on it over maintaining the hotfix.
- **`glm-5.3-flash-exl3-2x-spark`** — independent implementation of this
  same lane; v2.3 ships their OWN image (`ghcr.io/entrpi/...`,
  measured 74.2 tok/s single-stream in their FINDINGS.md, plus a math_500
  n=50 gate). Their `docs/FINDINGS.md` is the best single reference for
  what does NOT help on this lane (section 8) — read before spending time
  on speculative tuning here.
- **`ds4` / `ds4-on-spark`** — Blackwell perf forks of antirez/ds4
  (~4x prefill, ~1.5x decode); relevant to the DS4 recipes.
- **`dgx-spark-serving-mode`** — frees UMA memory by paring the desktop
  stack; relevant if more KV headroom is needed.
- Adjacent: `sparkinfer-glmrt`, `qwen3.5-122B-A10B-on-spark` (DFlash).

**Community**: `vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe` (kpool-tail
mechanism credited inside the MiaAI overlay).

## Research playbook for a fresh session

1. Snapshot this repo's state for the recipe (`git log -5 -- .`), confirm
   the running container's image digest still equals GHCR `:exl3`'s
   manifest digest (inspect + registry HEAD), and confirm all three boot
   markers on the leader.
2. MiaAI-Lab pass: new commits / open PRs / issues (the watchlist above);
   note anything touching the overlay, KV code, or serve flags.
3. Entrpi pass: same for their repos; diff their KV fixes against ours.
4. Image pass: GHCR tag list; if `:exl3` moved, pull on one node, boot in a
   scratch project name, and run the marker checks before touching the
   serving setup.
5. Adopt only what (a) applies to the running image/revision, (b) doesn't
   break the validated invariants (worker-first, port 4000, offline
   serving, GMU/hotfix wiring), and (c) carries a measured claim. Document
   the why in the commit, and update this file + README markers when the
   hotfix or profile changes.
