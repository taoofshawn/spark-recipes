# glm-v53-flash-intel-w4a16-sparkrun — research & provenance

Created 2026-09-20. Sparkrun-native port of `glm-v53-flash-intel-w4a16`
(mtp3 lane only). Every value traces to the parent recipe on `main`
(`5d107ba`, which carries the HF-ID serving flow) or to sparkrun's documented
semantics. No new upstream review was performed — this port changes the
*launch mechanism*, not the serving stack.

## Provenance

| value | source |
|---|---|
| model + revision `5eee1846…`, served name `glm-5.3-flash`, port 8000 | parent `.env` `MODEL`/`MODEL_REVISION`/`SERVED_MODEL_NAME` |
| `--revision gptq-surgery` HF-ID serving + `prepare-model.sh` surgery | parent `.env` `SURGERY_REVISION`, compose preflight block; parent README "the GPTQ surgery"; forum 382041 (@miken post 5) |
| mtp3 lane only | parent `.env` `LANE=mtp3` default; parent README "Why mtp3 is the default" (2026-09-18 A/B: mtp3 ≥ dflash2pmu at c4 aggregate in every round, 46.1–52.2 vs 39.8–45.6 tok/s) |
| spec config `{"method":"mtp","num_speculative_tokens":3,"disable_eagle_block_drop":true}` | parent compose mtp3 branch (baked patch 0001 = upstream vllm#53388); florianbrede-ayet `tp2_glm53flash_autoround_mtp3_pmu128` (forum 382632), vendored in parent `mtp3-pmu128/` |
| GMU 0.85, MAX_SEQS 6, MNBT 8192, block 2304, KV fp8_e4m3, KV pin 13.5 GB (upstream 14.0 GB rejected — ~12% c4 cost, 2026-09-19 bench), marlin MoE, `--no-enable-flashinfer-autotune`, ASYNC off, prefix-match-unit 128 + prompt-tokens-details, glm47/glm45 parsers, temp 1.0/top_p 0.95/thinking/effort-high pins | parent `.env` + compose `ARGS` mtp3 branch (all measured on this stack; not transferable between images) |
| baked patch series (0001 #53388, 0002 #53906, 0003 scheduler LCM, SM121 kpool indexer overlay `8a3ecfb0…`) + fail-closed installer | parent `mtp3-pmu128/{Dockerfile,apply_runtime_patches.py,patches/}` (upstream = florianbrede-ayet repo pinned 2026-09-07/re-synced `d528afe`) — baked into the published image, see the image build note below |
| chat template at `/models/chat_template_mm.jinja` | parent compose volume mount + `--chat-template` flag; `patches/chat_template_mm.jinja` baked into the published image at the same path (sparkrun has no per-file bind mounts; baking keeps the recipe mod-free) |
| NCCL/RoCE env block | parent compose `environment:` (NICs `enp1s0f0np0`, HCAs `rocep1s0f0,roceP2p1s0f0` = THIS cluster's, per AGENTS.md) |
| executor rootful bake (privileged/user:null/etc.) | parent compose runs the image as root; pattern from `.archived/deepseek-v4-flash-aiden-sparkrun` executor_config |
| `builder: docker-pull`, JSON flags as single `defaults` values, plain-invocation command ending without trailing backslash so sparkrun appends rendezvous flags | sparkrun docs (sparkrun.dev `/recipes/format/`, `/runtimes/vllm/`, `/developer-reference/builders/`) + `deepseek-v4-flash-aiden-sparkrun` precedent |

## Port decisions (why the sparkrun version differs from compose)

1. **Self-hosted published image** `ghcr.io/taoofshawn/vllm-glm53-intel-mtp3-pmu128:20260920@sha256:f1aaeafbc77d0173b3551f86c93f06a09debe979b5bbf079c9cc754b1391b481`
   (public, anonymous-pullable — verified 2026-09-20) instead of a node-local
   `glm53-intel-mtp3-pmu128:20260907`: the compose image isn't pullable, and
   sparkrun's docker-pull path wants a registry ref. Content is functionally
   identical to the compose mtp3 image (same base digest, same patch series,
   same installer) plus the chat template baked at the path the compose mounts
   it. Image-config labels (`org.opencontainers.image.description`,
   `base.digest`) confirm the build source.

   **Image build recipe (for future rebuilds — the build directory was
   removed from this recipe after publish):** `FROM
   ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6`;
   copy `apply_runtime_patches.py` + patches `0001-vllm-53388…`,
   `0002-vllm-53906…`, `0003-vllm-scheduler-lcm…`,
   `sparse_attn_indexer_kpool_sm121.py` from
   `../glm-v53-flash-intel-w4a16/mtp3-pmu128/`, run the installer twice
   (apply + `--verify-only`) against
   `/usr/local/lib/python3.12/dist-packages`; `COPY chat_template_mm.jinja`
   (from `../glm-v53-flash-intel-w4a16/patches/`) to
   `/models/chat_template_mm.jinja`.
2. **No GID auto-detect in the recipe.** sparkrun's per-host IB detection
   (`src/sparkrun/scripts/ib_detect.sh`) finds the RoCE v2 GID index
   (`show_gids`, sysfs fallback, default 3) and passes `NCCL_IB_GID_INDEX`
   per host via comm env; recipe env wins over comm env, so setting it in the
   recipe would clobber the detected per-host value. The compose recipe's
   boot-time detect loop is therefore dropped, not ported.
3. **No `--nnodes/--node-rank/--master-addr/--master-port/--headless`.**
   sparkrun's vllm-distributed runtime appends these per node itself
   (`vllm_distributed.py :: _generate_parallel_command`); hardcoding them
   triggers sparkrun's `check_hardcoded_rendezvous_flags` warning and would
   go stale. `--distributed-executor-backend mp` IS kept (compose-validated;
   not a sparkrun-managed flag).
4. **JIT caches repointed under the HF-cache mount**
   (`TORCHINDUCTOR_CACHE_DIR=/cache/huggingface/torchinductor`, same for
   tilelang): compose mounts a dedicated `CACHE_HOST_PATH` at `/cache`;
   sparkrun only mounts the host HF cache at `/cache/huggingface`, so this
   keeps the caches persistent without extra mounts.
5. **`NCCL_IB_ADDR_FAMILY`/`NCCL_IB_ADDR_RANGE` dropped**: they steer GID
   selection, which is now done by sparkrun's detector (explicit index).
   Everything else in the NCCL block is compose-verbatim.
6. **DFlash2 lane excluded by design** (user request): no drafter download,
   no `DFLASH_*` vars, no `dflash2-pmu128/` files.

## First bring-up (pending)

- Image is published and pinned by digest; first `sparkrun run` not yet
  executed — the recipe is unverified until a boot log shows the markers in
  the README. Watch items on first boot:
  - `SpeculativeConfig(method='mtp', ...)` present
  - KV pool ~1.92M tokens (13.5 GB pin)
  - rendezvous across nodes (sparkrun head-first gating should make the
    compose start-order failure mode impossible)
  - PMU128 cached_tokens behavior (README verify section)

## Changelog

- **2026-09-20** — created the sparkrun port (mtp3 lane only) on branch
  `glm-v53-flash-intel-w4a16-sparkrun`, off `main` @ `5d107ba`. Values traced
  in the table above; no serving-stack changes relative to the parent recipe's
  default lane.
- **2026-09-20** — image `ghcr.io/taoofshawn/vllm-glm53-intel-mtp3-pmu128:20260920`
  published (public; manifest digest
  `sha256:f1aaeafbc77d0173b3551f86c93f06a09debe979b5bbf079c9cc754b1391b481`,
  config labels confirm the build source). `container:` pinned by digest;
  removed the local `sparkrun-image/` build directory and the README
  build/push steps (build recipe preserved above).
