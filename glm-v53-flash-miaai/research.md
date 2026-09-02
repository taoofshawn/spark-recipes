# research.md — glm-v53-flash-miaai: long-term maintenance notes

Dated audit trail of every problem hit while adopting and deploying this
recipe, with root causes, workarounds, and what a future update session must
re-visit. This file is the long-term memory; README.md is deliberately kept
minimal (live info only).

---

## 2026-09-01/02 — first deployment (2x GB10: spark-0f0b / spark-6d14)

### Status at time of writing

- DS4-vision port (`deepseek-v4-flash-vision-miaai`): deployed, healthy,
  vision validated (correct 4-shape description; multimodal_tokens=117 in
  usage). Temporarily torn down to make room for GLM — bring back with
  `docker compose --env-file .env --env-file .env.node{0,1} up -d` on branch
  `deepseek-v4-flash-vision-miaai`.
- GLM (this recipe): boot repeatedly killed by a KV-capacity `ValueError`
  (34.15 GiB needed vs 16.16 available). Root-caused (below) and patched via
  a vendored boot hotfix (`hotfix_kv_check_glm5.py`); final boot validation
  was still in progress when this file was written. One earlier boot WITH the
  check fully no-op'd reached health 200 — allocation itself is sound.

### Environment facts that shaped everything

- Each DGX Spark exposes **ONE GPU** (`nvidia-smi` shows a single GPU).
  TP=2 always spans the two nodes. A single-node TP=2 diagnostic boot is
  impossible (worker assert `local_rank < device_count` fails).
- No `hf` CLI, no `huggingface_hub` in the host python, and pip is
  PEP-668-locked on the Sparks. Use `python3 -m venv /tmp/hfvenv` +
  `/tmp/hfvenv/bin/pip install huggingface_hub` for any cache work.
- Host HF cache mounts into the GLM image at **`/root/.cache/huggingface`**
  (image bakes that path) — NOT `/cache/huggingface` like the DS4/Anemll
  image. Wrong mount path = offline model-not-found.
- `docker-compose.override.yml` is AUTO-LOADED by compose when no `-f` is
  given. We used one for diagnostics (`/tmp/kvdiag/patched.py` mount +
  `GLM53_KV_DIAG=1`) — **it was removed after diagnosis**. If a future boot
  behaves strangely, check for a leftover override file on BOTH nodes.
- Worker-first start order + ~35 s stagger. The worker (headless) sits in
  the torch.distributed rendezvous while the leader loads ~6 min of weights;
  during that window the worker log spams non-fatal
  `TCPStore.cpp sendBytes failed` / `ProcessGroupNCCL "should dump"` warnings
  every second. **These are wait-window noise, not the failure.** Judge boot
  health by the LEADER log.
- GPU contention: before starting one recipe, `docker compose down` the
  other on BOTH nodes. (Once the leader's old DS4 container survived a
  switch because `down` was only run on the worker — the new container then
  crash-looped on GPU allocation.)

### Problem 1 — boot script YAML folding (BOTH miaai recipes)

`command: bash -lc - >` (folded scalar) joins consecutive script lines into
one; plain statements merge (`fi echo ...`) → bash syntax error at boot, and
worse, statements that happen to still parse get silently corrupted.

Fix: `- |` (literal scalar). Commits 5be01c3 (ds4v) and ede5ead (glm).
Lesson: always extract the rendered script and `bash -n` it after compose
edits.

### Problem 2 — HF hub-dir name mapping (GLM)

`tr '/' '--'` is char-to-char: it produced `models--Mia-AiLab-GLM-...`
(single dash) while HF uses `--` per slash. Boot died with
`FATAL: EXL3 weights not found` even though weights were cached. Fix: `sed
's|/|--|g'` (cbd502b). The DS4 recipes always used sed.

### Problem 3 — weight-revision pinning vs cache (BOTH recipes)

The recipe pins upstream's revisions; the Sparks' caches held different
(same-day) revisions:
- Vision-Exp: pin `86f746b3`, cache had `e46e16bf` (newer commit).
- GLM EXL3: pin `25a44fd`, cache had `024db9f7`.
- DFlash2 drafter: cache had TWO revs (7d74cdd8 "Release", dc77ff1c
  "Checkpoint update"), tip bf582e4e not cached; the compose resolved the
  drafter with an un-pinned fallback (nondeterministic glob order).

Resolution procedure that worked:
1. Diff the two revisions' trees via the HF API
   (`/api/models/<id>/tree/<rev>`, compare LFS oids). In both cases the
   trees were byte-identical except README — only the pin was stale-era.
2. Materialize the pinned snapshot dir by `cp -a` of the cached snapshot
   (relative symlinks to ../../blobs survive; verify no broken links with a
   rglob is_symlink()/exists() check).
3. For the drafter, downloaded tip bf582e4e (5 files, ~40 s) on both nodes
   and PINNED it: `DFLASH_REVISION` in .env + compose passthrough (05540ab).

Watch out: if a future pinned rev has REAL weight diffs, do a proper
`snapshot_download` in the /tmp venv on both nodes instead of the cp -a
trick, and expect a long download.

### Problem 4 — THE BIG ONE: GLM KV-capacity ValueError (34.15 GiB)

Symptom: boot dies after weight load at

    ValueError: To serve at least one request with the model's max seq len
    (1000000), (34.15 GiB KV cache is needed, which is larger than the
    available KV cache memory (16.16 GiB) ...)

34.15 GiB = ceil(1_000_000 / 64) * 2_351_104 — the drafter group's PADDED
page charged for every block of a 1M request.

Root cause chain (all inside the GHCR image's
`vllm/v1/core/kv_cache_utils.py`, none of it reachable from compose flags):

1. The baked DFLASH2-DRAFTER-GROUP **builder**
   (`_get_kv_cache_groups_glm5_next`) emits the drafter's 5 SWA layers as
   PADDED slot-share: block_size=64, page_size_padded=mla_page. Its own
   comment blesses this ("Manager 64 matches the SWA kernel ... safe strided
   view").
2. The baked **layout detector** (`_glm5_next_tensor_layout`) REJECTS any
   drafter spec with `page_size_padded is not None` — a stale precondition
   ("NEVER page_size_padded") from the pre-padded era. Builder and detector
   disagree => detector returns None => every glm5-aware consumer falls back
   to generic paths.
3. The KV-capacity check (`_max_memory_usage_bytes_from_groups`, called from
   `get_kv_cache_configs` per-worker loop) has NO glm5 branch at all in this
   image. The all-uniform "DeepseekV4" branch captures GLM's groups and
   charges length-scaled pages per block for the drafter AND (in honest
   accounting) the mamba groups.
4. Mamba extra: MambaSpec groups are padded to the MLA page; their naive
   `max_memory_usage_bytes` scales with max_model_len (~280 blocks each at
   1M), but mamba SSM state is length-INDEPENDENT and, in the slot-share
   layout, mamba layers parasitize the MLA block ids (shared_by in
   `get_kv_cache_config_from_groups`). True 1M demand is ~313 block ids
   (280 MLA + ~32 draft-window + 1 tail) ≈ 8.4 GiB ≤ 16.16 available —
   consistent with upstream's documented 1.75x concurrency at 1M on a 690-
   block pool.

Why upstream doesn't hit this: their validated boots run LOCALLY BUILT
images (recipe-stamped, issue #102 env) where builder, detector and
accounting are in sync. The GHCR prebuilt image lags — their open issue
#97 "Local rebuild gets overwritten by GHCR" is exactly this class of
drift. With their own start.sh + this same GHCR image you would hit the
identical error; it is not a compose-conversion artifact.

Fix: vendored boot hotfix `hotfix_kv_check_glm5.py` (mounted
`/opt/glm53/hotfix_kv_check_glm5.py`, run in the compose patch loop before
`vllm serve`), which edits `kv_cache_utils.py` in place:
- A: detector accepts drafter specs whose page_size_padded == mla_page
  (still rejects any other padding);
- B: an early glm5 branch in `_max_memory_usage_bytes_from_groups` computes
  needed memory allocator-consistently: per_block = len(mla)*mla_page +
  len(idx)*idx_page (+ standalone draft pages), blocks/req = cdiv sum over
  MLA + draft + tail groups, mamba groups EXCLUDED (length-independent,
  slot-shared);
- D: diagnostic wrapper dumping the group structure when the detector
  still returns None (leave in; it is one line per boot, or strip when the
  hotfix is retired).
Commits 9583658 -> e080b64 -> bc15903 -> 01ec852. The script is fail-closed:
it preflights every anchor and REFUSES to write (nonzero exit, boot aborts)
if the image changes under it.

Empirical backing: a diagnostic boot with the check fully no-op'd reached
health 200 — the allocator's pool sizing (num_blocks = available //
per_block, ~648 blocks) covers the real ~313-block demand of a 1M request.

### Entrpi's repositories — review for tips and alternative solutions

**https://github.com/Entrpi** maintains a parallel line of DGX Spark
serving work that overlaps both of our recipes. Review on every update
pass:

- **`Entrpi/glm-5.3-flash-exl3-2x-spark`** — GLM-5.3-Flash EXL3 4bpw +
  DFlash2 on 2x Spark, one-shot installer, measured 33-74 tok/s. An
  INDEPENDENT implementation of the same lane as this recipe — compare
  serve flags, KV accounting fixes, and measured numbers before trusting
  our workarounds.
- **`Entrpi/vllm-glm-5.3-flash-spark`** — a vLLM fork branch with the
  glm5_next port + **sm121 fixes**. This is the most likely source of a
  PERMANENT fix for the builder/detector/accounting drift we hotfixed
  (Problem 4). If that fork's kv_cache_utils is self-consistent, prefer
  rebasing our image on it (or adopting its patches) over maintaining
  `hotfix_kv_check_glm5.py`.
- **`Entrpi/ds4` / `Entrpi/ds4-on-spark`** — CUDA Blackwell perf fork of
  antirez/ds4 (DeepSeek-V4-Flash): ~4x prefill, ~1.5x decode. Relevant to
  the DS4 recipes (aiden/tonyd2wild/vision) — a possible prefill
  acceleration to evaluate.
- **`Entrpi/ds4-spark-vllm`** — DS4 hybrid quant on a single Spark via
  vLLM (single-node lane).
- **`Entrpi/dgx-spark-serving-mode`** — headless / multi-user serving mode:
  free unified memory for vLLM by paring down the desktop stack. Relevant
  if we need more KV headroom on the UMA.
- **`Entrpi/sparkinfer-glmrt`**, **`Entrpi/qwen3.5-122B-A10B-on-spark`**
  (DFlash speculative decode on Spark) — adjacent DFlash work.

Also in the same community ecosystem: `vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-
Spark-recipe` (credited inside the MiaAI overlay for the kpool-tail
mechanism, docs/KPOOL_TAIL_BUG.md).

### Things a future session MUST re-visit (GLM)

1. **When upstream re-bases the GHCR `:exl3` image** (or if you switch to a
   local `BUILD=1` image with a current overlay, or adopt Entrpi's
   `vllm-glm-5.3-flash-spark` fork), RETIRE the hotfix: delete
   `hotfix_kv_check_glm5.py`, its compose volume mount, and its entry in
   the patch loop. The hotfix's anchor preflight is designed to FAIL the
   boot loudly when the image no longer matches — that is your signal.
2. **Upstream (MiaAI-Lab) issues to watch:**
   - #97 (GHCR vs local rebuild drift) — if they publish a re-synced GHCR
     image, re-test without the hotfix first.
   - #94 (honest KV-capacity boot log, GLM53_KV_CAPACITY_LOG) — if merged
     into the image, their capacity line supersedes our accounting patch.
   - #102 (EXL3_FAT_KERNEL=1 + MNBT=7168: head TP rank dies SILENTLY at
     multimodal warmup on 2x GB10 — open as of 2026-09-01). Our boots have
     not yet reached that stage. If a boot dies WITHOUT a traceback right
     after CUDA-graph capture / multimodal warmup: set `EXL3_FAT_KERNEL=0`
     and `MAX_NUM_BATCHED_TOKENS=2048` (their documented clean-boot combo)
     and record it here.
   - #86 (GLM53_INDEXER_WORKSPACE=rightsize) — opt-in, reclaims ~4.5 GiB of
     indexer prefill workspace for KV (+26-28% capacity). We kept
     `stock`. If KV pressure shows up, flip to `rightsize` in .env.
   - #88 (SPEC_METHOD=mtp rollback), #85 (concurrency) — read before
     touching SPEC/MNS knobs.
3. **Drafter/model revision bumps:** re-run the HF tree-oid diff before
   swapping pins (see Problem 3 procedure). The DFlash2 repo has a pattern
   of same-day "Checkpoint update" commits with byte-identical serving
   files.
4. **If the 34.15-style ValueError ever returns with a DIFFERENT number:**
   the hotfix math section (fix B) is the place to re-derive; the debug
   wrapper (fix D) prints the exact group layout to correlate.

### DS4-vision recipe notes (same session, for symmetry)

- Image digest pin `sha256:a839484...` VERIFIED against GHCR
  (Docker-Content-Digest match). Tag updates upstream should be re-pinned
  the same way (token + manifest HEAD).
- Its hotfix chain runs `hotfix-dsv4-vision-exp.py` etc. at boot from
  `./patches/`; the bias_vl routing fix (upstream #175/#179) is baked into
  that vendored file — keep it in sync with upstream main.
- Entrpi's ds4 forks (see above) are the perf-improvement lane to watch.

### omp configuration (pending task from the operator)

After GLM validation: update the omp model config — add image input support
for the deepseek endpoint, and update GLM's context length to 1M
(/v1/models reports max_model_len 1000000).
