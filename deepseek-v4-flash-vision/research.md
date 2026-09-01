# DeepSeek-V4-Flash-Vision-Exp — Research & Handoff Notes

**Status: DRAFT — built but NOT tested.** Gathered 2026-08-31 (release day) from
the NVIDIA Developer Forum, GitHub, Docker Hub, and HuggingFace. This file is
the context for `README.md` plus the watch-list for future passes. The recipe
was designed from first principles against the DeepSeek-ai reference; no
community recipe was used as an upstream — every repo was researched and only
the pieces that fit this cluster and this model's physics were adopted.

---

## 1. The model (DeepSeek-ai reference is the source of truth)

`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`, released 2026-08-31, MIT license,
public. It is the DeepSeek-V4-Flash-0731 text backbone **plus** a native vision
tower and aligner, further trained for multimodal agentic use. The HF repo
ships the authoritative reference implementation in `inference/` and a
vision-capable encoder in `encoding/`.

### Checkpoint facts (HF API)

| field | value |
|---|---|
| repo / revision | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` / community pin `86f746b3…` (weights identical to HEAD `e46e16bf…`; later commits are README/eval only) |
| size | 167.82 GB (156.29 GiB), 48 shards |
| arch | `DeepseekV4ForCausalLM` (same string as the text model — vision class cannot be resolved from the checkpoint alone) |
| weights | FP8 e4m3 (block 128×128, ue8m0 scales) non-expert **+ FP4 experts** (`expert_dtype: fp4`) |
| text backbone | 43 layers, 256 routed experts / 6 active, 1 shared, Hyper-Connections, MLA (head_dim 512, qk_rope 64, qkv_lora 1024), sliding window 128, C4 compression |
| DSpark | `dspark_block_size: 5`, `dspark_noise_token_id: 128799`, target layers [40,41,42], `dspark_markov_rank: 256`, MTP `num_nextn_predict_layers: 3` |
| vision | 32-layer ViT (`vision_dim` 1024, 16 heads, inter 2816, patch 14, 2D RoPE θ 10000), aligner (3×3 space-to-depth + 2-layer GELU MLP), `vision_max_n_token: 384`, `vision_min_pixels: 147456`, `vision_max_wh_ratio: 8`, `vision_downsample_ratio: 3` |
| image tokens | sentinels `vocab_size + {0..4}` (start/pad/image/newline/end); OOV to the embedding table |
| rope | YaRN factor 16 from `original_max_position_embeddings: 65536`; `max_position_embeddings: 1048576` |
| vocab | 129280 (`dspark_noise_token_id` 128799, image placeholder `<｜deepseek_image｜>` = token 129264) |
| reference runtime | torch ≥2.10, transformers>5, tilelang 0.1.8, fast_hadamard_transform (readable ref, not a serving engine) |

### Vision coupling into the LM (the four hard parts — per vLLM PR #54561 + verified against the reference)

1. **OOV sentinel ids** — each image expands to a block of `vocab_size + {0..4}`;
   they never index the embedding table but the LM reads them.
2. **Hash-routing guard** — the 3 hash-routed layers index `tid2eid[input_ids]`
   sized `[vocab_size, k]`; sentinels must be substituted before lookup or it
   reads out of bounds.
3. **Modality-forked expert selection** — every MoE gate carries a second bias
   `bias_vl [n_experts]` used instead of `bias` for image positions in top-k
   selection only (routing weights stay from unbiased scores).
4. **Input-id validation** — `v1/engine/input_processor.py` caps ids at
   `max(tokenizer.max_token_id, vocab_size - 1)` and rejects the sentinels;
   the wrapper must expose `max_token_id >= vocab_size + 4`.

Plus the reference makes image spans **bidirectionally visible** within
`[IMAGE_START, IMAGE_END]` across all layers (`get_image_visible` /
`get_window_topk_idxs_visible`) — the sparse index matrix is widened to
`window_size + vision_max_n_token` for prefill rows and per-token left/right
bounds are threaded into the index kernel.

### Reference `encoding_dsv4.py` vs the recipe's existing encoder

The DeepSeek-ai reference encoder is vision-capable (image content blocks →
`<｜deepseek_image｜>` placeholders, `<image>path</image>` tagged-text, image
validation) **but lacks the tool-argument repair** the aiden/tonyd2wild recipes
carry. This recipe's `encoding_dsv4.py` is the reference + the aiden tool-arg
hardening block (`normalize_tool_arguments`, `repair_tool_arguments_json`,
`parse_tool_arguments`, `dsml_param_to_python`,
`normalize_parsed_dsml_tool_args`, `prepare_openai_tool_call_for_execution`) so
the vision recipe keeps both capabilities. This merge must be re-done whenever
either the reference encoder or the hardening block is updated upstream.

---

## 2. vLLM state on release day (2026-08-31)

**The latest release v0.28.0 (2026-08-26) serves the text model fully but has
ZERO vision support for DeepSeek-V4.** Vision lives only on open PRs based on
vLLM **main**. The SM120 sparse-MLA decode fix `eidx must be contiguous`
(#53574) is merged into main only, not v0.28.0 — another reason the from-scratch
build pins main, not the release.

### Relevant vLLM PRs (all `vllm-project/vllm`)

| PR | state | what it does |
|---|---|---|
| **#54566** (Isotr0py) | OPEN, draft | canonical vision implementation: ViT+aligner (`common/vision.py`), multimodal preprocessor (`common/mm_preprocess.py`), VL wrapper (`nvidia/vl_model.py`), OOV sentinels, `tid2eid` guard, `bias_vl` expert routing, **bidirectional image attention** via widened sparse index rows + per-token visibility, arch-convertor routing. Also touches tokenizer (`max_token_id`, `flatten_content_blocks`) and fused-topk CUDA kernels (`topk_softplus_sqrt_kernels.cu`). Tested on 2×RTX PRO 6000 SM120 TP2 with `--attention-backend FLASHINFER_MLA_SPARSE_DSV4`; image token counts match the reference exactly. |
| **#54631** (lucamotz) | OPEN | one-pass checkpoint streaming — defers MegaMoE/MHC finalization to the model-level post-load hook so interleaved vision/LM weights stream instead of materializing the whole map; **tested on a 2-node DGX Spark with a real image request**. Also enables DSpark for the vision wrapper (`allow_deepseek_v4_vision` plumbing, `_normalize_deepseek_v4_dspark_hf_config`). |
| **#53574** (lucifer1004) | MERGED to main | SM120: keep C128A decode topk indices contiguous (fixes the `eidx must be contiguous` boot blocker; pre-existing main bug, not the vision PR's fault). |
| #52292 | OPEN | opt-out for synchronized FlashInfer autotune on multi-node (GB10 deadlock without GPUDirect RDMA). Not adopted — the recipe uses `--enable-flashinfer-autotune` and, if the 2-node deadlock appears at boot, the first knob is `--no-enable-flashinfer-autotune`. |
| #51538, #43477, #52018, #52502, #52035 | MERGED in v0.28.0 | DSV4 sparse MLA end-to-end, SM120 route, B12X FP4 MoE, GB10 fused-MoE tuning, DeepGEMM nv_dev pin — the text-side foundation this build inherits. |

### From-scratch build decision

- **Base: vLLM main pinned at `07ea9350baf84e33fd696d36fec9b9f24735a733`** —
  the vision PRs are main-based and #53574 is only in main. A v0.28.0 + cherry-
  pick attempt fails (patches do not apply cleanly).
- Patches applied in order: `0001-vision-54566.patch` (git apply),
  `0002-streaming-54631.patch` (patch with fuzz — follow-on to 0001, context
  drifts), `0004-nvfp4-ds-mla-kv.patch` (git apply). `0003` (#53574) is already
  in the pinned main and is kept in the dir for reference/re-bases.
- Built with the **official vLLM Dockerfile** (`docker build --target
  vllm-openai`, `TORCH_CUDA_ARCH_LIST=12.1a`), which compiles `csrc/` from
  source — required because the vision PR modifies CUDA kernels.

---

## 3. NVFP4 KV (`nvfp4_ds_mla`) — the community patch

`nvfp4_ds_mla` is **not** an upstream vLLM KV dtype (upstream has `fp8_ds_mla`
only; `nvfp4`/`nvfp4_4over6` are generic, not DS-MLA). All three 2x DGX Spark
vision recipes use it and report the largest KV pools. It comes from the
tonyd2wild/Anemll lineage.

This recipe's patch 0004 ports the tonyd2wild stage-A/B/C plumbing to main's
structure: the KV is stored in the same packed uint8 DS-MLA layout as
`fp8_ds_mla` (584B envelope: 448 NoPE + 128 RoPE + 8 fp8 scale) with
`KVQuantMode.NVFP4`. Files touched: `config/cache.py`, `utils/torch_utils.py`,
`models/deepseek_v4/attention.py`, `nvidia/flashinfer_sparse.py`,
`v1/attention/backends/mla/sparse_swa.py`. **This patch is the highest-risk
item in the build** — the FlashInfer sparse MLA kernel must accept the NVFP4
quant mode on the SM121 path. If it fails at boot, the compose default flips to
`fp8_ds_mla` (upstream, validated with the vision PR on SM120) with no other
change.

### KV pool expectations (measured on the working vision recipes, for reference)

| recipe | KV dtype | pool tokens | GMU |
|---|---|---|---|
| MiaAI (Anemll 0.1.1, vLLM 0.25.2) | nvfp4_ds_mla | 2,331,430 (17.04 GiB) | 0.83 |
| tonyd2wild vision (0.21.1rc1) | nvfp4_ds_mla | 2,790,000 (19.12 GiB) | 0.85 |
| sfxnz (eugr B12X) | fp8 | 1,250,741 | 12 GiB pin |

The Vision-Exp ViT takes more weight RAM than 0731 (~few GiB/rank), which is
why MiaAI runs GMU 0.83 (vs 0.85 on the text model).

---

## 4. Community recipe landscape (2026-08-31) — directions, not upstreams

Three repos ship working 2x DGX Spark vision servers on release day; all pin
revision `86f746b3…` and all confirm native vision works. The directions below
were evaluated against this recipe's priorities (coding quality > vision > KV
pool > speed) and this cluster's validated invariants (start order, port 4000,
offline serving, GID auto-detect, node-local JIT caches).

| | MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark | tonyd2wild/…-Vision-Exp-DSpark-… | sfxnz/…-Vision-Exp-vLLM-2x-DGX-Spark |
|---|---|---|---|
| runtime | Anemll `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` (vLLM 0.25.2.dev0) | `vllm-dspark-runtime:…-probe-c-…` (vLLM 0.21.1rc1, B12X) | `eugr/spark-vllm-b12x:latest` + Dockerfile |
| vision method | `patches/vision_exp/` monkey-patch (ViT+Aligner into `DeepseekV4ForCausalLM`) | `ds4v_model/vision/mm/registry.py` bind-mounts | vLLM plugin `VLLM_PLUGINS=dsv4_vision` (wrapper subclass) |
| KV | nvfp4_ds_mla | nvfp4_ds_mla | fp8 |
| DSpark | k=6, `flashinfer_b12x` | k=5, Patch 4 required | k=6 |
| GMU / seqs | 0.83 / 6 | 0.85 / 12 | — / 2 |
| vision specifics | `--limit-mm-per-prompt image=8`; images in user messages only; single-chunk prefill; `VLLM_USE_B12X_MHC` off (rms_norm_eps 1e-20) | same vision constraints; **Patch 4 mandatory** (draft shared-expert loader, silently halves decode if dropped) | `VLLM_USE_B12X_MHC` off; `bias_vl` gate mapping missing upstream (text path) |
| measured | 62–83 tok/s; tool-eval 91/100 | count 80/code 52/prose 33 tok/s | 87/27 tok/s (structured/prose) |
| provenance | monkey-patch — not a clean upstream path | patch-on-old-runtime | plugin — closest to a clean vLLM integration but uses `--hf-overrides`-free arch routing |

**Why this recipe goes its own way:** the monkey-patch (MiaAI) and bind-mount
(tonyd2wild) approaches patch a prebuilt community image rather than building
from source; sfxnz's plugin approach is closest to upstream but wraps the model
class in a way that loses DSpark draft attributes unless carefully structured.
The clean path for a from-scratch build is the actual upstream PRs (#54566 +
#54631) which implement vision natively in vLLM main. The community recipes
contributed the *operational* knowledge adopted here: KV dtype choice, GMU
0.83, `MAX_NUM_SEQS=6`, `MAX_NUM_BATCHED_TOKENS=8192`, `--limit-mm-per-prompt
{"image":8}`, `VLLM_USE_B12X_MHC=0`, and the k-divisibility constraint.

### Facts adopted from the community (with sources)

- **`VLLM_USE_B12X_MHC` must stay OFF.** Vision-Exp `rms_norm_eps=1e-20` breaks
  the B12X fused Gram mHC kernel (accepts only `1e-6`) — sfxnz README and
  tonyd2wild both document it. Default is off; this recipe never sets it.
- **Images belong in `user` messages only** — `system`/`assistant` images return
  HTTP 400 (MiaAI README). Cap 8 images/request. GIF = still frame; no video.
- **Image spans need single-chunk prefill** — the reference asserts image spans
  are prefilled in one chunk (`forward`, `start_pos == 0` branch).
- **`k` divisibility is runtime/image-specific.** The checkpoint has both
  `dspark_block_size=5` and `num_nextn_predict_layers=3`. On this upstream build
  (main-based), DSpark uses `dspark_block_size` as the trained draft width
  (#54631 `_normalize_deepseek_v4_dspark_hf_config`) and the vision recipes boot
  k=6; tonyd2wild's older 0.21.1 image instead requires k divisible by 5. If the
  upstream validator rejects k=6 at boot, try k=3 first (smallest divisor), then
  k=5.
- **Draft acceptance on prose is modest (~25%)** — inherent to this vision
  variant; Patch-4-style shared-expert loader fix is already in the base here
  (it's a main-era fix, see DSPARK-SHARED-EXPERT-FIX.md in the vendored
  tonyd2wild upstream).
- **Boot time** — cold start compiles Triton/DeepGEMM kernels (~7-8 min+) and
  AOT; warm-up penalty until a few hundred tokens flow. Never benchmark right
  after boot.
- **`--tokenizer-mode deepseek_v4` + `--tool-call-parser deepseek_v4` +
  `--reasoning-parser deepseek_v4`** is the native path; the vision PR extends
  the same wrapper with image handling. Do not switch to HF tokenizer mode.

---

## 5. Overlay files (this recipe)

All bind-mounted `:ro` over the official image's vLLM install
(`/opt/venv/lib/python3.12/site-packages/vllm/...`).

| file | target | what it does |
|---|---|---|
| `encoding_dsv4.py` | `tokenizers/deepseek_v4_encoding.py` | DeepSeek-ai reference encoder (vision placeholders, `<image>` tags, image validation) **+** tool-argument hardening from the aiden/tonyd2wild recipes (merged; see §1) |
| `deepseek_v4_wrapper.py` | `tokenizers/deepseek_v4.py` | PR #54566 base (`max_token_id` for sentinels, wrap-after-cache, `TokenizersBackend`) + corrected reasoning-effort aliases (none/off/low/high/max/xhigh) |
| `detokenizer.py` | `v1/engine/detokenizer.py` | stop-suppression backport: client `stop` strings stay dormant inside ` thinking` (prevents silent `content:null`) |

---

## 6. Testing plan (when you switch over)

1. **Build gate** — `docker images` shows `vllm-vision-dspark:main-07ea9350ba`
   on both nodes; same digest.
2. **Boot gate** — worker first, then head ~30s later; `/v1/models` shows
   `"id": "deepseek-v4-flash-vision-exp"`, `"max_model_len": 1048576`.
3. **Log markers** — `Using DeepSeek's ... nvfp4_ds_mla KV cache format.`,
   `Available KV cache memory: N GiB`, `GPU KV cache size: N tokens`,
   `Application startup complete`.
4. **Vision gate** — image prompt (base64 `image_url`) → correct description;
   text-only prompt → correct text; both with `thinking` on and off.
5. **KV gate** — record pool size; compare vs MiaAI 2.33M @ 0.83.
6. **Agent/tool gate** — tool-call JSON validity (the hardening overlay), stop
   strings not firing mid-reasoning.
7. **Speed gate (informational)** — `stream:false`, read
   `usage.completion_tokens`; expect ~60-80 tok/s single stream warmed.

### Known failure modes to check first

| symptom | likely cause / fix |
|---|---|
| `eidx must be contiguous` at boot | #53574 not in the base → verify pin `07ea9350…`; if a newer pin drops it, re-apply `0003`. |
| `kv_cache_dtype not supported` for nvfp4_ds_mla | 0004 didn't apply to a re-pinned base → re-check all patches apply. |
| DSpark k=6 rejected at boot | try k=3 (or 5); update `MTP_NUM_TOKENS`. |
| vision 400 / blank image | images in `system`/`assistant`, >8 images, or image span not single-chunk prefilled. |
| decode half-speed, correct output | draft shared-expert loader — should be fixed in base; verify no `Skipping unknown DSpark weight` in DEBUG logs. |
| 2-node deadlock during FlashInfer autotune | add `--no-enable-flashinfer-autotune` (or backport #52292). |
| `HF_HUB_OFFLINE` model not found | vision snapshot missing on worker cache; `hf download` it there. |

---

## 7. Watch-list (future improvement passes)

1. **Upstream vision PR merge.** #54566 is a draft and #54631 is stacked on it;
   when the vision PR merges, re-base this recipe onto the next vLLM release and
   drop patches 0001/0002 (keep 0004 — nvfp4_ds_mla is still community-only).
2. **Bidirectional image attention.** The PR's version is in; verify on real
   screenshots whether the causal-only fallback in tacos8me's #54561 branch
   would change anything (it shouldn't be adopted — lower fidelity).
3. **Video support.** The checkpoint has no video tower; the model card says GIF
   is a still frame. Watch for a video variant.
4. **`bias_vl` on the B12X text path** — sfxnz notes upstream only maps
   `ffn.gate.bias` to `e_score_correction_bias`; image-token expert routing on
   the B12X fused path may need the same `bias_vl` mapping as the PR's router
   fix. Check at boot under image traffic.
5. **InstantTensor/runai_streamer load formats** — a faster boot (178 GB in ~3
   min) is field-validated on the GLM stack; the vision checkpoint is ~168 GB.
   Gate-test before adopting (multi-node-unstable on GLM without caps).
6. **nvfp4 KV for `fp8_ds_mla` fallback** — keep the 584B envelope decision
   documented; if the FlashInfer SM120 kernel rejects NVFP4 quant mode, the
   fallback is `fp8_ds_mla` with a ~40% smaller pool (still 1M-capable).
7. **k tuning** — per-position acceptance on vision content; k=6 vs k=3 vs k=5.
8. **vLLM releases** — v0.29+ may ship DSV4 vision natively; re-evaluate the
   from-scratch build vs the official image when it does.

## Sources

- Model: [deepseek-ai/DeepSeek-V4-Flash-Vision-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp)
  (reference `inference/` + `encoding/` vendored; weights NOT stored in this
  repo — they live in the HF cache on the sparks).
- vLLM PRs: [#54566](https://github.com/vllm-project/vllm/pull/54566),
  [#54631](https://github.com/vllm-project/vllm/pull/54631),
  [#54561](https://github.com/vllm-project/vllm/issues/54561),
  [#53574](https://github.com/vllm-project/vllm/pull/53574),
  [#52292](https://github.com/vllm-project/vllm/pull/52292),
  [v0.28.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.28.0).
- Community recipes (directions only, not upstreams):
  [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark),
  [tonyd2wild/DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark),
  [sfxnz/DeepSeek-V4-Flash-Vision-Exp-vLLM-2x-DGX-Spark](https://github.com/sfxnz/DeepSeek-V4-Flash-Vision-Exp-vLLM-2x-DGX-Spark).
- Forum: [381911 release thread](https://forums.developer.nvidia.com/t/deepseek-v4-flash-vision-exp-is-released-as-open-weights/381911)
  (0rand, technigma.ai, DColt), 372268 (aiden thread), 378824 (tonyd2wild),
  380257 (GB10 vision+DSpark benchmark).
