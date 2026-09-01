# Handoff — deepseek-v4-flash-vision (continue here)

**Date:** 2026-08-31 / 2026-09-01 02:17 UTC-4
**State:** Image BUILT and booting; blocked on a config validator. All fixes are
identified; **next session starts at "redeploy the fixed image"** below.

---

## Goal

Serve `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` (the multimodal DSV4) on the
2-node DGX Spark cluster via a from-scratch vLLM build (vLLM main pinned at
`07ea9350baf84e33fd696d36fec9b9f24735a733` + open vision PRs #54566/#54631 +
community `nvfp4_ds_mla` KV patch), port 4000, 1M context, DSpark speculative
decoding. Everything is staged on the cluster; the image builds and boots to
the point of weight loading.

## Cluster (fixed, from AGENTS.md)

- Leader / node 0 / rank 0 / API: `spark-0f0b.shawndo.intra` (RoCE `192.168.0.170`)
- Worker / node 1 / rank 1 / headless: `spark-6d14.shawndo.intra` (RoCE `192.168.0.171`)
- Login from this dev host via `ssh spark-0f0b.shawndo.intra` etc.
- Repo checked out at `~/spark-recipes-vision-build/` on both sparks (recipe dir synced, **not** the git repo).

## Current state (last verified)

- **Image `vllm-vision-dspark:main-07ea9350ba` (26.9GB) is ON THE LEADER, PATCHED** for the nvfp4+MLA validator issue (derived image built directly on the leader, verified in-container).
- **The WORKER still has the OLD (unpatched) image** — the docker-save/load transfer of the patched image was never re-run after the validator fix.
- Both `ds4v-dspark` containers are crash-looping (`Restarting (1)`) with:
  `Value error, nvfp4 KV cache is not supported with MLA (Multi-head Latent Attention) backends` from `vllm/config/vllm.py` `VllmConfig.validate_nvfp4_kv_cache_with_mla`.
- This is EXPECTED: it is the exact bug that was fixed in the leader image (see Fixes §6). Restart the containers AFTER redeploying the fixed image to both nodes.

## NEXT ACTIONS (in order)

1. **Transfer the patched leader image to the worker:**
   - On leader: `docker save vllm-vision-tag | ssh spark-6d14.shawndo.intra "docker load"` (takes ~5–8 min over the 10 GbE link; use `nohup` + a marker file — I repeatedly lost sessions with plain foreground ssh). Prefer `rsync /tmp/vllm-vision.tar` if the tar on disk is the patched one (re-save from the patched image first, as the tar predates the fix).
2. **Recreate containers (worker first, leader ~35 s later):**
   ```
   ssh spark-6d14: cd ~/spark-recipes-vision-build && docker compose --env-file .env --env-file .env.node1 up -d --force-recreate
   sleep 35
   ssh spark-0f0b:  cd ~/spark-recipes-vision-build && docker compose --env-file .env --env-file .env.node0 up -d --force-recreate
   ```
3. **Watch boot logs** on the leader (`docker logs ds4v-dspark -f`). Milestones: `Resolved architecture: DeepseekV4ForConditionalGeneration`, `Using nvfp4_ds_mla data type to store kv cache`, `Starting vLLM server`, `Uvicorn running on http://0.0.0.0:4000`.
4. **Verify serving + vision request:**
   - `curl http://127.0.0.1:4000/v1/models` → expect `deepseek-v4-flash-vision-exp`, `max_model_len` 1048576.
   - Send a chat completion with an `image_url` content block per README (images allowed only in `user` messages; cap 8).
5. **Clean up / finish:**
   - Run `docker compose down` on both when done; remove `/tmp/vllm-vision.tar` (~27GB ×2).
   - Commit the recipe package (see Repo state below); copy the recipe to both nodes' final `~/spark-recipes/deepseek-v4-flash-vision/` if you want that path.

## Fixes that had to be discovered historically (all IN the working recipe now)

1. **Editable-install overlay mounts must target `/workspace/vllm`, NOT site-packages.** The image pip-installs vLLM as an editable (`/workspace`); mounting overlays into `/usr/local/lib/python3.12/dist-packages/vllm/...` breaks the import finder at serve time (`ModuleNotFoundError: vllm.v1.attention`). The compose now mounts `./encoding_dsv4.py`, `./deepseek_v4_wrapper.py`, `./detokenizer.py` over `/workspace/vllm/...` (see `docker-compose.yml`).
2. **Base image ENTRYPOINT is `vllm serve`.** The official `vllm/vllm-openai:v0.28.0` has `ENTRYPOINT=["vllm","serve"]`, which swallowed the old `command:` as server args. The compose now uses `entrypoint: ["/bin/bash", "/boot.sh"]` and mounts `./boot.sh` (the GID-detect + `vllm serve` script, rewritten from the compose command block with compose-escape fixups `$${→${`).
3. **Compose mangles multiline `command:` strings** (word-splits them). Hence boot.sh as a mounted file.
4. **DSpark `num_speculative_tokens` must be divisible by 5** (drafter emits 5 tokens/pass). The Vision-Exp checkpoint has MTP depth 3, but validation demands k divisible by **5**. Set `MTP_NUM_TOKENS: 5` in `docker-compose.yml` (k=6 was rejected: `num_speculative_tokens:6 must be divisible by n_predict=5`).
5. **`nvfp4 KV` + MLA validator** (current blocker): `vllm/config/vllm.py` `validate_nvfp4_kv_cache_with_mla` rejects any `nvfp4*` KV when the model uses MLA. The community `nvfp4_dsmmla` KV is exactly an MLA cache format — the guard is wrong for it. **Fixed in the leader image only** by patching the condition to allow dtype ending `_ds_mla`:
   ```python
   if (
       self.cache_config.cache_dtype.startswith("nvfp4")
       and not self.cache_config.cache_dtype.endswith("_ds_mla")
       and self.model_config.use_mla
   ):
   ```
   **TODO: bake this into the build script as a proper patch file** — the derived leader image already has it, but the build script (`build/build-vllm-vision.sh`) does NOT (someone re-running from scratch loses the fix). Add `patches/0005-nvfp4-mla-validator.patch` and append it to `PATCHES` in the script, then rebuild once to prove reproducibility.
6. **Build resource knobs:** `MAX_JOBS` default lowered to `8` (a full 20-job build thrashes the GB10 (load >40) and starves sshd; 8 keeps the node usable). Full from-scratch build ~26 min at MAX_JOBS=20, ~50+ min at 8. Must run on an arm64 spark (host is amd64 — no local build).
7. Build deps in the image: added `setuptools-rust`, `libcusparse-dev-13-0`, `libcusolver-dev-13-0`, `cuda-nvrtc-dev-13-0` (CUDA 13 images drop unversioned libnvrtc + dev headers; cmake + DeepGEMM need them).

## Repo / file state — LOCAL (dev host: /home/sdrew/code/github.com/taoofshawn/spark-recipes)

- Branch: `deepseek-v4-flash-vision` (checkout).
- Untracked new dirs: `deepseek-v4-flash-vision/` (the recipe) and `.research-dsv4-vision/` (HF downloads: reference `inference/` + `encoding/` reference implementation + tests).
- Recipe contents (all UNCOMMITTED):
  - `docker-compose.yml` (entrypoint /boot.sh, MTP 5, overlay mounts /workspace, `MODEL_TON` rev `e46e16bf…` = what's cached on both nodes; weights identical to pinned 86f746b3…)
  - `.env`, `.env.node0`, `.env.node1` (cluster IPs/NICs correct)
  - `boot.sh` (mounted; GID autodetect + exec vllm serve)
  - `encoding_dsv4.py`, `deepseek_v4_wrapper.py`, `detokenizer.py` (merged vision+tool hardened encoders; extended from aiden recipe)
  - `build/build-vllm-vision.sh`, `patches/0001-0004`
  - `README.md`, `research.md` (research includes the full PR/community/place checkpoint audit + test-plan)
- **Encoding tests pass:** `encoding/test_encoding_dsv4.py` → 12/12 against the merged `encoding_dsv4.py` (ran via a pytest shim; no pytest on the dev host).

## Remote state (leader/worker)

- Both nodes: `~/spark-recipes-vision-build/` has compose + `.env*` + overlays + `boot.sh` (synced, current) ✓
- Model cache BOTH nodes: `~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-Vision-Exp/snapshots/e46e16bf…` (157GB each) ✓ — offline serving needs this exact snapshot (`MODEL_REVISION` matches).
- Leader: image patched (validator fix). Worker: image OLD → must re-transfer + `docker load` before recreate.
- State of `/tmp/vllm-vllm.tar` on worker (26.9G) is NOT the patched image (patched image was built after). Re-save/transfer.
- Stale artifacts on leader: `/tmp/vllm-transfer.done`, `/tmp/vllm-load.done`, logs; safe to delete.
- The two `ds4v-dspark` containers are crash-looping — expected; whatever you do, `docker compose down` them first if you change anything, then bring them up per the boot order.

## Gotchas learned (avoid re-debugging)

- ssh sessions from dev host DIE when the local `timeout` kills the foreground job — for long transfers use `nohup ... &` on the leader with a marker file, then poll.
- `docker compose run` reproduces compose env+mounts better than `docker run` when bisecting container issues.
- The site-packages/editable split (fix #1) is the #1 trap for anyone touching overlays.
- The base image already contains `vllm/vllm-openai:v0.28.0`; we build FROM it (CUDA 13 toolchain present) — the `pip install -e` in the image recompiles kernels for SM121.

## Verification recap (evidence)

- Image builds: `vllm-vision-dspark:main-07ea9350ba` built on the leader; in-image checks pass: `DeepseekV4VLProcessor`, `DeepseekV4ForConditionalGeneration` (registry), `nvfp4_ds_mla` in `CacheDType`.
- Overlay exports: `vllm serve --help` in a container WITH the /workspace mounts gets past all imports (device-inference error only outside GPU context = healthy).
- The current blocker is the nvfp4+MLA validator, fixed on the leader image; the boot previously progressed to `Resolved architecture: DeepseekV4ForConditionalGeneration` + `nvfp4_ds_mla` KV + then the validation crash.

---

**Good luck. The remaining work is: re-transfer the patched image to the worker, recreate both containers, and run the first real vision request — then commit the recipe per the repo conventions (see AGENTS.md).**
---

# UPDATE 2026-09-01 (post-handoff session — full constraint matrix mapped)

## What was fixed / changed since the handoff

1. **Validator patch formalized as `patches/0005-nvfp4-ds-mla-mla-validator.patch`** and wired into `build/build-vllm-vision.sh` (PATCHES + verify guard). ✅
2. **Worker got the patched image** (multiple wired-IP transfers — see below). ✅
3. **b12x package added** (`b12x==1.3.0` pip; the official base image does NOT ship it — without it the mxfp4 oracle rejects B12X at boot). In build script + in-image verify. ✅ Boot marker reached: `Using 'B12X_MXFP4_MXFP8' Mxfp4 MoE backend`.
4. **flashinfer-python upgraded to 0.6.18** (adds DSV4 sparse-MLA `(32, topk=192)` dispatch from PR #4380). Companion `flashinfer-cubin` 0.6.18 was never published to PyPI → compose sets `FLASHINFER_DISABLE_VERSION_CHECK=1`. In build script. ✅
5. **`MODEL_REVISION` → `e46e16bf…`** (what's actually cached offline on both nodes; weights identical to the 86f746b3 pin). ✅
6. **Compose reworked**: `entrypoint: ["/bin/bash","/boot.sh"]` + mounted `boot.sh` (base image ENTRYPOINT `vllm serve` swallows the command; compose word-splits multiline command strings). ✅
7. **DSpark `MTP_NUM_TOKENS=5`** (validation: k must be divisible by the drafter's n_predict=5). ✅
8. **RE-PINNED the build to `lucamotz/vllm@71165e0528…`** (`codex/deepseek-v4-vl-streaming-loader`): upstream main at the flashinfer-0.6.18 bump + vision layer + streaming + DSpark trained-block-width + FIVE follow-up fixes our vendored 0001/0002 never had (vision gate, OOV tokens, MTP enable, breakable-CG registration, fixes). Build script rewritten (uniform patch loop). 0001/0002 dropped; **0004/0005/0006 kept**. ✅ Image `vllm-vision-dspark:71165e0528` builds; ALL in-image checks pass.
9. **New patches**: `0006-kv-spec-block64-clamp.patch` (vLLM `get_num_kernel_states` clamps to ≥1 state/page) and `patches/flashinfer-decode-dsv4-pagesize.cu` (kernel launcher accepting 256-token pages via template instantiation). Both in the repo, 0006 in PATCHES, both synced to the leader.
10. **AGENTS.md updated**: node-to-node bulk transfers MUST use the wired 192.168.0.x IPs (wifi ≈58 MB/s vs wired; transfer drops 12 min → ~4 min).
11. **`--enable-flashinfer-autotune` must stay OFF**: the 0.6.18 autotuner crashes the first decode launch (unsupported tactic → hard Check-fail instead of tactic skip).

## THE REMAINING BLOCKER (root cause fully mapped)

`FLASHINFER_MLA_SPARSE_DSV4` decode on GB10: `decode-dsv4 launch failed (unsupported shape or kernel error)` from `SparseMlaSm120DecodeDsv4` — happens during the TARGET model's memory-profile decode (not spec-decode specific). Constraint matrix, all empirically tested:

| --block-size | vLLM DSV4 KV spec | flashinfer decode-dsv4 kernel | result |
|---|---|---|---|
| 256 | ✓ (compress groups: 256//128=2, 256//4=64 states) | ✗ hard guard `page_block_size != 64` → `return false` | unsupported shape |
| 64 | ✗ `get_num_kernel_states`: 64//128 = 0 states → page_size_bytes=0 → ZeroDivisionError (patch 0006 clamps it) → then `No common block size for 64` in hybrid KV group merge | ✓ (64 is the kernel's template param) | no common block size |
| 256 + kernel launcher patched to instantiate `<…,256>` pages | ✓ | ✗ STILL fails inside `launch_decode_dsv4_impl` (deeper shape/traits check — KVCacheTraits<MT> vs actual head dims, or SM121 occupancy) | unsupported shape |

Also tested: PIECEWISE vs FULL cudagraph mode (irrelevant — the profile decode launches the kernel regardless); autotune on/off (off required).

**Conclusion**: flashinfer 0.6.18's SM120 DSV4 sparse-MLA decode kernel does not currently launch on GB10 (SM121) for this model's shape set (num_q_heads=32 padded, topk=192, 584B/token fp8_ds_mla KV). The kernel was validated upstream only on RTX PRO 6000 (SM120). The model itself is 1 day old. This is an upstream kernel-support gap, NOT a recipe bug.

## Next paths (in order of likelihood)

1. **Watch flashinfer upstream** for GB10/SM121 DSV4 support: issues #3937 (DSV4 NVFP4 KV on SM120), #4802 (SM120 sparse MLA refactor), PR #4752 (short-query tile heuristic). Any flashinfer release >0.6.18 or nightly with `sparse_mla_sm120` changes is worth an immediate retest (image rebuild not needed — just `pip install` the new flashinfer + wired retransfer).
2. **Ask lucamotz** (PR #54631 author, claims a real 2-node DGX Spark image request) which flashinfer build/flags his test used — his branch pins 0.6.18, so either he used additional patches or a different attention path.
3. **tonyd2wild's runtime image** ships a patched flashinfer for the DSV4 text model on this cluster — diffing his flashinfer against 0.6.18 upstream may reveal the missing GB10 fixes.
4. Only after the kernel launches: revisit `nvfp4_ds_mla` KV (patch 0004) — the fp8_ds_mla fallback works at the config level and block 256 satisfies the spec.

## Current cluster state (as of session end)

- Both nodes: containers DOWN (`docker compose down` — clean).
- Both nodes: image `vllm-vision-dspark:71165e0528` present (leader built it 2026-09-01 01:39; worker loaded the same via wired transfer).
- Both nodes: `~/spark-recipes-vision-build/` has current compose/boot.sh/env/patches/0001-0006 + flashinfer .cu.
- Leader: `/tmp/vllm-vision-build/vllm` = pristine lucamotz pin (patch 0006 diffable); stale tars in /tmp on both nodes (~27 GB each ×3) — safe to delete.
- Boot command when the kernel gap closes: worker first (`docker compose --env-file .env --env-file .env.node1 up -d`), leader 35 s later (`…node0`), then `curl http://127.0.0.1:4000/v1/models` → expect `deepseek-v4-flash-vision-exp`, then a chat completion with an `image_url` block.
