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

# flashinfer: the release 0.6.18 bundled in this base is NOT Blackwell-native
# for the fa2/fa3 MLA paths — the fp8-MLA device gate excludes major==12 and
# the fa3 fp8 MLA kernels carry no SM121 SASS ("no kernel image is available
# for execution on the device", observed 2026-09-21 round-2 bring-up at the
# SM90 sparse-MLA warmup). The legacy serving image
# glm53-intel-mtp3-pmu128:20260907 served THIS EXACT model/hardware/config
# with flashinfer 0.6.18.dev20260819: torch 2.13.0+cu130 (ABI-identical),
# ckv_scale_arr API identical (19 refs both sides), SM121-capable. Adopt its
# flashinfer + cubin dists wholesale. Applied BEFORE the patch chain, so the
# installer's SHA gates see the adopted tree (the legacy flashinfer already
# carries the (9,12) fp8-MLA gate; patch 0017 is therefore not needed).
COPY --from=glm53-intel-mtp3-pmu128:20260907 /usr/local/lib/python3.12/dist-packages/flashinfer /usr/local/lib/python3.12/dist-packages/flashinfer
COPY --from=glm53-intel-mtp3-pmu128:20260907 /usr/local/lib/python3.12/dist-packages/flashinfer_cubin /usr/local/lib/python3.12/dist-packages/flashinfer_cubin
COPY --from=glm53-intel-mtp3-pmu128:20260907 /usr/local/lib/python3.12/dist-packages/flashinfer_cubin-0.6.18.dev20260819.dist-info /usr/local/lib/python3.12/dist-packages/flashinfer_cubin-0.6.18.dev20260819.dist-info
COPY --from=glm53-intel-mtp3-pmu128:20260907 /usr/local/lib/python3.12/dist-packages/flashinfer_python-0.6.18.dev20260819.dist-info /usr/local/lib/python3.12/dist-packages/flashinfer_python-0.6.18.dev20260819.dist-info
RUN rm -rf ${VLLM_DIST}/flashinfer_python-0.6.18.dist-info

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
