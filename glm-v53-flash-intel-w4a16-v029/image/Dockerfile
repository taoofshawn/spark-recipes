# GLM-5.3-Flash Intel W4A16 AutoRound serving image (spark-recipes
# glm-v53-flash-intel-w4a16-v029). Base: the OFFICIAL model-recipe image
# vllm/vllm-openai:glm53-flash-arm64-cu130 (re-pushed 2026-09-09, main
# snapshot 385dce36b) — the only official image carrying GLM-5.3-Flash native
# model + MTP support (v0.29.0 stable predates the #53906 merge; verified
# 2026-09-20). Digest-pinned. Everything SM121-specific is baked here; the
# recipe runs with no runtime overlays and no GPTQ model surgery (the Intel
# AutoRound checkpoint loads natively via vLLM's INCConfig).
FROM vllm/vllm-openai@sha256:b0501f99fec5136f248f78d5850977a2ec32d55cd9a665f4a9ffef24cbdf7fe5

ARG VLLM_DIST=/usr/local/lib/python3.12/dist-packages

COPY apply_runtime_patches.py /tmp/spark-recipe/apply_runtime_patches.py
COPY patches/ /tmp/spark-recipe/patches/

# Fail-closed patch application (exact before/after SHA gates + syntax check
# of every patched file), then an independent second verification pass.
RUN set -eu; \
    python3 /tmp/spark-recipe/apply_runtime_patches.py --root "$VLLM_DIST"; \
    python3 /tmp/spark-recipe/apply_runtime_patches.py --root "$VLLM_DIST" --verify-only; \
    rm -rf /tmp/spark-recipe

# Baked chat template (rodman80's chat_template_mm.jinja — honors
# enable_thinking + reasoning_effort + image/video macros; byte-identical
# SHA-256 918b0d8c… to the copy validated on the legacy lane).
COPY chat_template_mm.jinja /models/chat_template_mm.jinja

# SM121 runtime env from the validated lineage (tonyd2wild launcher env):
# FlashInfer JIT compiles for sm_121a at runtime; expandable segments avoid
# the GB10 unified-memory swap-wedge.
ENV TORCH_CUDA_ARCH_LIST="12.1a" \
    FLASHINFER_CUDA_ARCH_LIST="12.1a" \
    FLASHINFER_DISABLE_VERSION_CHECK="1" \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

LABEL org.opencontainers.image.description="GLM-5.3-Flash Intel W4A16 AutoRound, official glm53-flash recipe base, SM121 patch stack baked (spark-recipes)" \
      org.opencontainers.image.base="vllm/vllm-openai:glm53-flash-arm64-cu130" \
      org.opencontainers.image.base.digest="sha256:b0501f99fec5136f248f78d5850977a2ec32d55cd9a665f4a9ffef24cbdf7fe5" \
      org.opencontainers.image.revision.vllm="0.28.1rc1.dev580+g385dce36b" \
      org.opencontainers.image.revision.patch53969="open PR d1e5e66-base, applied 2026-09-20"
