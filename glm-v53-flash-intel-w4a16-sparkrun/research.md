# glm-v53-flash-intel-w4a16-sparkrun — research & provenance

Sparkrun-native port of `glm-v53-flash-intel-w4a16` on `main` @ `e539498`
(the glm53flash-pmu128-mtp3 image lineage). Every value traces to the parent
recipe or to sparkrun's documented semantics. No new upstream review was
performed — this port changes the *launch mechanism*, not the serving stack.

## Provenance

| value | source |
|---|---|
| model `Intel/GLM-5.3-Flash-W4A16-AutoRound` + revision `5eee1846…`, served name `glm-5.3-flash`, port 8000 | parent `.env` `MODEL`/`MODEL_REVISION`/`SERVED_MODEL_NAME` |
| serving the raw auto-round snapshot via `INCConfig` (no weight preparation) | parent compose preflight (`raw auto-round snapshot OK`) + README "Image" section |
| spec config `{"method":"mtp","num_speculative_tokens":3,"disable_eagle_block_drop":true}` | parent compose `ARGS` (mtp3, native MTP head) |
| GMU 0.85, MAX_SEQS 6, MNBT 8192, block 2304, KV fp8_e4m3, KV pin 13.5 GB, `--kv-cache-memory-bytes`, oracle MoE selection (no `--moe-backend`), `--no-enable-flashinfer-autotune`, prefix-match-unit 128 + prompt-tokens-details, glm47/glm45 parsers, temp 1.0/top_p 0.95/thinking/effort-high pins | parent `.env` + compose `ARGS` (all measured on this stack; not transferable between images) |
| chat template at `/models/chat_template_mm.jinja` | baked into the published image at that path (sparkrun has no per-file bind mounts; baking keeps the recipe mod-free) |
| NCCL/RoCE env block | parent compose `environment:` (NICs `enp1s0f0np0`, HCAs `rocep1s0f0,roceP2p1s0f0` = THIS cluster's, per AGENTS.md) |
| executor rootful bake (privileged/user:null/etc.) | parent compose runs the image as root; pattern from `.archived/deepseek-v4-flash-aiden-sparkrun` executor_config |
| `builder: docker-pull`, JSON flags as single `defaults` values, plain-invocation command ending without trailing backslash so sparkrun appends rendezvous flags, `model_revision` in BOTH top-level and `defaults` | sparkrun docs (sparkrun.dev `/recipes/format/`, `/runtimes/vllm/`, `/developer-reference/builders/`) + `deepseek-v4-flash-aiden-sparkrun` precedent |

## Image

`ghcr.io/taoofshawn/vllm-glm53-intel-w4a16:glm53flash-pmu128-mtp3`
(digest-pinned in the recipe; public, anonymous-pullable — verified
2026-09-23 via the ghcr manifest API). Full lineage, patch provenance, and
the validation record live in the parent recipe's `research.md`; summary:

- base: `vllm/vllm-openai:glm53-flash-arm64-cu130` — the official
  model-recipe image, digest-pinned; the only image carrying GLM-5.3-Flash
  native model + MTP (vLLM `0.28.1rc1.dev580+g385dce36b`, CUDA 13.0 aarch64,
  torch 2.13.0+cu130)
- flashinfer `0.6.18.dev20260819` in-image (the base's release flashinfer is
  not Blackwell-native for fa2/fa3 MLA)
- SM121 patch stack baked at build by a fail-closed, SHA-gated installer:
  0003 (scheduler LCM block size — PMU128 core), 0011 (SM120 sparse-MLA nope
  topk), 0012 (PDL gate), 0013 (NoPE sparse-MLA backend on cap-12), 0014
  (kpool indexer top-k gate), 0015 (INCConfig nextn/MTP expert-name
  re-resolve), 0016 (fp8-KV plan dtype), 0018 (fa2 on non-SM90)
- chat template baked at `/models/chat_template_mm.jinja`

## Port decisions (why the sparkrun version differs from compose)

1. **Digest-pinned registry ref.** sparkrun's docker-pull path wants a
   registry ref and the compose image must match exactly; the recipe pins
   tag@digest so both nodes pull identical bytes.
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
6. **Per-node vars dropped** (`NODE_RANK`/`HEADLESS`/`ROCE_IP`/
   `MASTER_ADDR`/`PORT`/`MPORT`/`VLLM_HOST_IP`): sparkrun manages node
   identity and rendezvous itself.

## Watch items (first `sparkrun run` of this image)

- `SpeculativeConfig(method='mtp', ...)` present
- KV pool ~1.92M tokens (13.5 GB pin)
- rendezvous across nodes (sparkrun head-first gating should make the
  compose start-order failure mode impossible)
- PMU128 cached_tokens behavior (README verify section)

## First bring-up (2026-09-23) — PASS after entrypoint fix

Launched under sparkrun 0.2.38, which has no `executor_config.entrypoint`
support: the image's `ENTRYPOINT ["vllm","serve"]` consumed sparkrun's
keep-alive wrapper as extra `vllm serve` args (`-c` = `--compilation-config`
alias → `Invalid JSON`). Worked around at launch time with
`--executor-args "--entrypoint \"\""`, then upgraded the leader to sparkrun
0.3.9 (entrypoint support landed in 0.3.0) and fixed it in the recipe yaml
(`executor_config: entrypoint: ""`). All boot gates green:

- `quantization=inc` (native auto-round load), `SpeculativeConfig(method='mtp',
  num_spec_tokens=3)`, `Using 'MARLIN' WNA16 MoE backend` (oracle-selected),
  `Using FLASHINFER_MLA_SPARSE_SM90 attention backend`
- `GPU KV cache size: 1,920,956 tokens` (exact 13.5 GB-pin pool)
- `/health` 200; `/v1/models` → `glm-5.3-flash` /
  `Intel/GLM-5.3-Flash-W4A16-AutoRound` / `max_model_len` 1048576
- chat smoke: coherent reply, 14 completion tokens

## Changelog

- **2026-09-23** — added `executor_config.entrypoint: ""` (requires sparkrun
  >= 0.3.0; the image inherits a consuming `ENTRYPOINT ["vllm","serve"]`
  which otherwise breaks sparkrun's keep-alive launch — see the first
  bring-up note). Leader upgraded 0.2.38 → 0.3.9.
- **2026-09-23** — updated the sparkrun port to the parent recipe's current
  image lineage (`glm53flash-pmu128-mtp3`, rebase off `main` @ `e539498`):
  new digest pin, `INCConfig` native auto-round load path (raw HF snapshot
  served directly), `--kv-cache-memory-bytes` flag, oracle MoE selection (no
  forced `--moe-backend`), and `VLLM_ENGINE_READY_TIMEOUT_S`. Recipe left
  uncommitted for review.
