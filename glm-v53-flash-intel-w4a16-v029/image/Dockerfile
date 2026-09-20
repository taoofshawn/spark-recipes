# GLM-5.3-Flash Intel W4A16 AutoRound serving image (spark-recipes
# glm-v53-flash-intel-w4a16-v029). Base: official stable vLLM v0.29.0, arm64,
# CUDA 13.0 — digest-pinned. Everything SM121-specific is baked here; the
# recipe runs with no runtime overlays and no GPTQ model surgery (the Intel
# AutoRound checkpoint loads natively via vLLM's INCConfig on v0.29.0).
FROM vllm/vllm-openai@sha256:18372a7224938643461b846fb64c5c9d3d6e9727e82caf2dc3043e620c9d4d7a

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

LABEL org.opencontainers.image.description="GLM-5.3-Flash Intel W4A16 AutoRound, stock vLLM v0.29.0, SM121 patch stack baked (spark-recipes)" \
      org.opencontainers.image.base="vllm/vllm-openai:v0.29.0-aarch64" \
      org.opencontainers.image.base.digest="sha256:18372a7224938643461b846fb64c5c9d3d6e9727e82caf2dc3043e620c9d4d7a" \
      org.opencontainers.image.revision.vllm="v0.29.0" \
      org.opencontainers.image.revision.patch53969="open PR d1e5e66-base, applied 2026-09-20"
