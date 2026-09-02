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
- GLM (this recipe): DEPLOYED, HEALTHY, SERVING VALIDATED on 2026-09-02
  (~02:45Z) after a vendored KV-accounting hotfix (Problem 4 below).
  `/v1/models` -> `GLM-5.3-Flash-EXL3`, max_model_len 1000000; image+text
  chat completion returns a correct 4-shape description.

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

Root cause chain (all inside the GHCR image's
`vllm/v1/core/kv_cache_utils.py`, none of it reachable from compose flags),
in the order we disproved candidates:

1. The baked DFLASH2-DRAFTER-GROUP **builder**
   (`_get_kv_cache_groups_glm5_next`) emits the drafter's SWA layers as a
   group whose specs lose their sliding-window bound once wrapped in
   `UniformTypeKVCacheSpecs`.
2. The baked **layout detector** (`_glm5_next_tensor_layout`) rejects any
   drafter spec with `page_size_padded is not None` — out of sync with the
   builder's padded geometry (an older/other builder variant). Detector
   returns None => every glm5-aware consumer falls back to generic paths.
3. The KV-capacity check (`_max_memory_usage_bytes_from_groups`, called from
   `get_kv_cache_configs` per-worker loop) has NO glm5 branch at all in this
   image.
4. Final root cause (confirmed by a per-group accounting breakdown logged at
   boot): the mamba groups were NOT the problem (their naive demand is only
   ~9 blocks each). The inflator was the **DFlash2 draft group**:
   block=16, page=163,840, contributing **1025 blocks** at 1M. The builder
   wraps the drafter's SlidingWindowSpec layers in `UniformTypeKVCacheSpecs`,
   and THAT WRAPPER LOSES THE SLIDING WINDOW — its demand scales with
   max_model_len instead of the window-bounded live demand
   ceil(2048/16) = 128 blocks. A windowed SWA group never holds more than
   ceil(window/block) live blocks; the stock wrapper forgets this. The same
   window-unaware formula produces the misleading stock capacity line (see
   upstream issue #94 — "GPU KV cache size" is not an honest capacity figure
   for this hybrid).

Why upstream doesn't hit this: their validated boots run LOCALLY BUILT
images (recipe-stamped, issue #102 env) where builder, detector and
accounting are in sync. The GHCR prebuilt image lags — their open issue
#97 "Local rebuild gets overwritten by GHCR" is exactly this class of
drift. With their own start.sh + this same GHCR image you would hit the
identical error; it is not a compose-conversion artifact.

2026-09-02: raised `GPU_MEM_UTIL` 0.87 -> 0.8848 (commit 3c6b8bd) per the
boot-log's own hint (CUDA-graph memory profiling makes 0.87 behave like
0.8552) to recover an upstream-sized pool (~690 blocks / ~1.75M tokens) on
the next boot. Verify on first boot after the node reboot: "Available KV
cache memory" should be ~17-18.7 GiB and blocks/req headroom larger; if the
engine OOMs at graph capture, back off to 0.87.

2026-09-02 (post-reboot boot with 0.8848, then hotfix v6): VERIFIED —
`Available KV cache memory: 16.21 GiB`, health 200, serving as
`glm-5.3-flash` (renamed from GLM-5.3-Flash-EXL3, commit bc79124, to match
the other glm recipe and the omp config id).

### FINAL root cause (Problem 4, confirmed 2026-09-02 ~04:15Z)

The image's baked builder `_get_kv_cache_groups_glm5_next` is an OLDER
patch revision: its else branch is "STANDALONE" — `new_draft_specs =
dict(draft_specs)` — leaving the drafter at block_size=16 with its
16384-token window. Live demand of that group = 1025 block ids, so a full
1M request needs ~1306 block ids vs a ~621-block pool => the scheduler
correctly REFUSES every large-max_tokens request forever
(`num_requests_waiting{reason="capacity"}` stuck, GPU 0%, small requests
still run at ~18 tok/s — which is why the server "looked" healthy).
The current overlay's builder instead emits PADDED SLOT-SHARE
(block_size=64, page_size_padded=mla_page) -> draft demand 32 blocks ->
total 313 blocks/req -> fits.

The boot check ValueError (34.15 GiB) was the same arithmetic surfacing at
boot; the 16,384-token window (not the config's 2048) is what the runtime
class qwen3_dflash2.py assigns; do not confuse the two when re-deriving.

Fix E in `hotfix_kv_check_glm5.py` replaces the stale STANDALONE branch
with the current overlay's padded slot-share code (fail-closed anchor on
the exact baked text). After it, the boot logs upstream's exact expected
line:

    DFlash2 drafter KV: padded slot-share block=64 mla_page=2351104
    (was block=16); exact-fit page mismatch draft_bytes/token=2048

and the stock capacity line jumps to `GPU KV cache size: 1,054,006
tokens, Maximum concurrency ... 1.05x` (from 436,661 / 0.42x).

### Load tests (2026-09-02 ~04:45Z, post-fix E) — PASS

| test | result |
|---|---|
| single large request (max_tokens=6000) | completed, finish=length, ~24 tok/s sustained |
| admission during large request | running=1, waiting=0, capacity=0 (was stuck forever pre-fix) |
| GPU during decode | 95-96% util, 51-60 W (was 0%, 12.9 W) |
| 4x concurrent large requests (max_num_seqs=4) | all 4 admitted simultaneously, 0 stuck |
| aggregate throughput under 4-way load | 59.1 tok/s (~14.8 tok/s/req) |
| 4x3000-token completions | all finish=length, correct usage counts |
| post-burst state | running=0, waiting=0, clean idle — no recurrence |

Historic numbers for comparison: upstream documents 33-74 tok/s (their
kit/env); our single-stream ~24-28 tok/s, 4-way aggregate ~59 tok/s.

### Boot warning review (2026-09-02 post-reboot boot) — all expected

| warning | verdict |
|---|---|
| `SymmMemCommunicator: Device capability 12.1 not supported` | expected on GB10; falls back to PYNCCL (also in repo AGENTS.md table) |
| `Custom collectives are disabled because this multi-node ...` | expected for TP over 2 nodes; PYNCCL all-reduce is the working path |
| `Sparse MLA impl has no dense-MHA prefill path; using the top-k MQA path only` | by design — packed fp8_ds_mla sparse path is the only SM12x kernel; upstream serves the same way |
| `Draft model DFlash2Qwen3ForCausalLM does not support external multimodal embeddings ... text-only draft inputs` | by design; the drafter drafts text tokens only |
| `Disabling fine-grained prefix-cache hits ... KpoolTailManager requires block-aligned lookups` | upstream default; block-aligned (3584-token) hits still work. Upstream PR #84 (GLM53_FINEGRAINED_APC) re-enables 64-token hits opt-in — evaluate later |
| `Default vLLM sampling parameters have been overridden by the model's generation_config.json (temperature 1.0, top_p 0.95)` | intended: the checkpoint ships its tuned sampling defaults; omp clients may override per-request |
| `Triton kernel JIT compilation during inference: _topk_topp_kernel ... latency spike` | one-time compile on first top-k/top-p shape; cached afterwards. Only the very first sampled request pays it |
| worker-side `TCPStore ... sendBytes failed` / NCCL heartbeat spam during leader weight load | rendezvous wait-window noise, non-fatal (judge boot by the LEADER log) |
| two defunct `python3` zombies per node (ppid 1) | dead multiprocessing helpers; PID1 (vllm) never reaps them. Cosmetic, no resource held. Note: in an earlier boot the SAME symptom pattern (zombies + idle GPU) meant dead TP workers — distinguish by `ps -eo` inside the container: if `VLLM::EngineCore` and `VLLM::Worker_TP0` are alive, zombies are cosmetic |

Fix: vendored boot hotfix `hotfix_kv_check_glm5.py` (mounted
`/opt/glm53/hotfix_kv_check_glm5.py`, run in the compose patch loop before
`vllm serve`), which edits `kv_cache_utils.py` in place:
- A: detector accepts drafter specs whose page_size_padded == mla_page
  (still rejects any other padding) — future-proofs the detector against
  the builder's padded geometry;
- B: an early glm5 branch in `_max_memory_usage_bytes_from_groups`
  computes needed memory allocator-consistently: per_block =
  len(mla)*mla_page + len(idx)*idx_page (+ standalone draft page),
  blocks/req = window-bounded cdiv over MLA + draft + tail groups,
  mamba groups EXCLUDED (length-independent SSM state, slot-shared with
  MLA block ids);
- E: replaces the stale STANDALONE builder else branch with the current
  overlay's padded slot-share rescale (THE load-bearing fix);
- D: diagnostic wrapper dumping the group structure when the detector
  still returns None (leave in; it is one line per boot, or strip when
  the hotfix is retired).
Commits 9583658 -> ... -> 6768d54. The script is fail-closed: it
preflights every anchor and REFUSES to write (nonzero exit, boot aborts)
if the image changes under it.

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

Also in the same community ecosystem:
`vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe` (credited inside the
MiaAI overlay for the kpool-tail mechanism, docs/KPOOL_TAIL_BUG.md).

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
     multimodal warmup on 2x GB10 — open as of 2026-09-01). Our validated
     boot DID pass warmup with the defaults (fat=1, MNBT=7168), so this
     did not reproduce on our kit — but if a FUTURE boot dies WITHOUT a
     traceback right after CUDA-graph capture / multimodal warmup: set
     `EXL3_FAT_KERNEL=0` and `MAX_NUM_BATCHED_TOKENS=2048` (their
     documented clean-boot combo) and record it here.
   - #86 (GLM53_INDEXER_WORKSPACE=rightsize) — opt-in, reclaims ~4.5 GiB of
     indexer prefill workspace for KV (+26-28% capacity). We kept
     `stock`. If KV pressure shows up (e.g. want >1 concurrent 1M
     request), flip to `rightsize` in .env first.
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
