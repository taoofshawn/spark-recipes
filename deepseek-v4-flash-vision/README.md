# deepseek-v4-flash-vision — 1M token context, native vision, DSpark

A self-contained recipe for running `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`
(the first multimodal DeepSeek-V4 model) on a 2-node DGX Spark cluster, served
through a **from-scratch vLLM build** (vLLM main + open vision PRs) with DSpark
speculative decoding and the community `nvfp4_ds_mla` KV cache.

The model shares DeepSeek-V4-Flash-0731's text backbone (43 layers, 256 routed
experts, DSpark MTP) and adds a 32-layer ViT + aligner for native image input.
Weights: FP8 non-experts + FP4 experts, ~168 GB, 1M native context.

> ⚠️ **Status: built, NOT yet tested on this cluster.** The image build
> compiles on a spark; boot/vision/KV verification is pending your switch-over
> (do not stop a running recipe while this one is being validated). See
> `research.md` §6 for the test plan and known failure modes.

## Configuration overview

| knob | value |
|---|---|
| Image | `vllm-vision-dspark:main-07ea9350ba` (built from source, see below) |
| vLLM base | vLLM **main** pinned `07ea9350b…` + PRs #54566 (vision) + #54631 (streaming) + #53574 (SM120 eidx fix, in main) + `nvfp4_ds_mla` KV patch |
| Vision | native ViT+aligner via upstream PR (bidirectional image attention); `--limit-mm-per-prompt '{"image":8}'` |
| KV cache | `nvfp4_ds_mla` (community patch on the `fp8_ds_mla` packed layout) |
| DSpark spec tokens | 6 (`MTP_NUM_TOKENS=6`) |
| `max_num_seqs` | 6 |
| `max_num_batched_tokens` | 8192 |
| `gpu_memory_utilization` | 0.83 (Vision-Exp ViT takes more weight RAM than 0731) |
| attention / MoE | `FLASHINFER_MLA_SPARSE_DSV4` / `b12x` (`VLLM_USE_B12X_MOE=1`) |
| sampling | `--generation-config vllm`; thinking `true`, `reasoning_effort=high` defaults |
| context length | 1M (1048576) |
| serve | port `4000`, served model `deepseek-v4-flash-vision-exp` |

> `VLLM_USE_B12X_MHC` must stay **off** (never set it): the Vision-Exp
> `rms_norm_eps=1e-20` breaks the B12X fused Gram mHC kernel (accepts only
> `1e-6`) — documented by sfxnz and tonyd2wild. Default is off; this recipe
> does not set it.

## Prep (both nodes)

```bash
cd ~/spark-recipes/deepseek-v4-flash-vision/
HF_MODEL=deepseek-ai/DeepSeek-V4-Flash-Vision-Exp
HF_REVISION=e46e16bf6035c6f317eb2ac7458eb0362926d402
hf download $HF_MODEL --revision $HF_REVISION
```

> The revision in `docker-compose.yml`'s `MODEL_REVISION` must match the
> snapshot on disk on BOTH nodes — serving is offline (`HF_HUB_OFFLINE=1`),
> so vLLM can only load what is already cached. `e46e16bf…` is what this
> cluster has cached (weights identical to the community pin `86f746b3…`).

## Build the image (from scratch — on a spark)

vLLM has no release with DeepSeek-V4 vision yet; the vision layer exists only
on open PRs based on vLLM **main** (see `research.md` §2). This recipe pins
main, applies the vendored patches, and builds the image with the official
`vllm/vllm-openai:v0.28.0` ARM64 image as base (it carries the CUDA 13.0
toolchain) — the modified `csrc/` kernels are recompiled for SM121.

```bash
# on one spark (ARM64); does not touch running containers
cd ~/spark-recipes/deepseek-v4-flash-vision/build
./build-vllm-vision.sh          # ~30-90 min first build
# repeat on the worker node (or push to a registry and pull both)
```

Result: `vllm-vision-dspark:main-07ea9350ba` on both nodes, same digest.

## Run

```bash
# Node 1 (follower): start first
docker compose --env-file .env --env-file .env.node1 up -d

# Node 0 (leader): start about 30s after
docker compose --env-file .env --env-file .env.node0 up -d
```

Verify:

```bash
curl -s http://127.0.0.1:4000/v1/models
# expect "id": "deepseek-v4-flash-vision-exp", "max_model_len": 1048576
```

## Vision usage

Images use OpenAI `image_url` content blocks; the placeholder is
`<｜deepseek_image｜>`. Images belong in **`user` messages only** (system or
assistant images → HTTP 400). Cap 8 images per request. GIF is a still frame;
there is no video encoder in the weights.

```bash
curl -s http://127.0.0.1:4000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-v4-flash-vision-exp",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
    {"type": "text", "text": "What is in this image? One sentence."}
  ]}],
  "max_tokens": 128,
  "temperature": 0,
  "chat_template_kwargs": {"thinking": false, "reasoning_effort": "low"}
}'
```

## Overlays (mounted over the image's vLLM install)

| file | target (editable tree) |
|---|---|
| `encoding_dsv4.py` | `/workspace/vllm/tokenizers/deepseek_v4_encoding.py` |
| `deepseek_v4_wrapper.py` | `/workspace/vllm/tokenizers/deepseek_v4.py` |
| `detokenizer.py` | `/workspace/vllm/v1/engine/detokenizer.py` |

> The image pip-installs vLLM **editable**, so the runtime resolves `vllm.*`
> from `/workspace/vllm` — overlays must mount there, NEVER into
> site-packages (`/usr/local/lib/python3.12/dist-packages/vllm`). Mounting
> into site-packages breaks the editable import finder at serve time
> (`ModuleNotFoundError: vllm.v1.attention`).

- `encoding_dsv4.py` = DeepSeek-ai reference encoder (vision placeholders,
  `<image>` tagged-text, image validation) **merged with** the tool-argument
  hardening the aiden/tonyd2wild recipes carry (`normalize_tool_arguments`,
  `repair_tool_arguments_json`, `parse_tool_arguments`…).
- `deepseek_v4_wrapper.py` = PR #54566 base (`max_token_id` for image sentinels,
  wrap-after-cache, `TokenizersBackend`) + corrected reasoning-effort aliases
  (none/off/low/high/max/xhigh).
- `detokenizer.py` = stop-suppression backport: client `stop` strings stay
  dormant inside ` thinking` (prevents silent `content:null` answers).

All three must be staged on **both** nodes at the same path (the compose
bind-mounts them; a missing file on one node silently runs unpatched).

## Updating the vLLM pin / patches

`build/build-vllm-vision.sh` pins vLLM main at `07ea9350b…`. When the upstream
vision PR (#54566) merges, re-pin to the merge commit (or the next release),
verify `patches/0001`/`0002` apply (or drop them), and re-verify `0004`
(nvfp4 KV) still applies. See `research.md` §7 for the watch-list.

## Reference

- [DeepSeek-V4-Flash-Vision-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp)
  — model card + reference `inference/` + `encoding/`.
- `research.md` — full audit: model facts, vLLM PR status, community recipe
  landscape, NVFP4 KV provenance, testing plan.
