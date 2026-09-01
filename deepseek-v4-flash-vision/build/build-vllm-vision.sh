#!/usr/bin/env bash
# =============================================================================
# build-vllm-vision.sh — build a from-scratch vLLM image for
# deepseek-ai/DeepSeek-V4-Flash-Vision-Exp on 2x DGX Spark (GB10, SM121).
#
# Strategy
# --------
# vLLM has NO release containing the DeepSeek-V4-Flash-Vision-Exp vision layer
# (released 2026-08-31; v0.28.0 is the newest release and is text-only for
# DSV4). The vision implementation exists only on open PRs based on vLLM main:
#   #54566 (Isotr0py) vision tower+aligner+bidirectional image attention
#   #54631 (lucamotz)  one-pass checkpoint streaming (tested on 2-node DGX
#                      Spark with a real image request)
# and the SM120 sparse-MLA decode fix #53574 is already merged into main.
# Therefore this build pins vLLM main at VLLM_PIN and applies the PRs as
# patches (vendored in ../patches/).
#
# The build base is the official `vllm/vllm-openai:v0.28.0` ARM64 image, which
# ships the CUDA 13.0 toolchain (nvcc, gcc) plus python 3.12 / torch 2.13.
# We pip-install the patched vLLM source into a fresh image FROM it, so the
# modified csrc kernels (topk_softplus_sqrt.cu etc.) are compiled for SM121.
# (The official vLLM Dockerfile's manylinux build-builder image is amd64-only,
# so the full from-scratch Dockerfile route cannot run on ARM64.)
#
# NVFP4 KV note: `nvfp4_ds_mla` is a community KV dtype (tonyd2wild/Anemll
# lineage), NOT in upstream vLLM. Patch 0004 adds it on top of the DSV4
# fp8_ds_mla packed layout (584B envelope, NVFP4 quant mode). If the vision
# path ever rejects it at boot, the recipe's compose default can fall back to
# `fp8_ds_mla` (already upstream).
#
# Build host: run on one spark (ARM64); the image is architecture-specific.
# Does NOT touch running containers. Run:  ./build/build-vllm-vision.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_DIR="$(dirname "$SCRIPT_DIR")"

# --- Tunables ---------------------------------------------------------------
# lucamotz/vllm @ codex/deepseek-v4-vl-streaming-loader = upstream main at the
# flashinfer-0.6.18 bump (#54313) + the DSV4 vision layer (#54566) + streaming
# loader + DSpark trained-block-width + the five follow-up vision/DSpark fixes
# (vision gate, OOV token usage, MTP enable, breakable-CG registration) that
# the open PRs #54566/#54631 never carried individually. This exact tree is
# what lucamotz validated with a real image request on 2-node DGX Spark.
VLLM_REPO="${VLLM_REPO:-https://github.com/lucamotz/vllm.git}"
# Pinned vLLM main commit — the base the vision PRs were developed against.
# Update deliberately, then re-verify ../patches/0001-0004 still apply.
VLLM_PIN="${VLLM_PIN:-71165e052868a3949ccce0be117ab56aff541d7d}"
# Official ARM64 runtime image carrying the CUDA toolchain.
BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai:v0.28.0}"
# GB10 = SM121. The recipes run TORCH_CUDA_ARCH_LIST=12.1a; keep kernel
# compilation targeted at the actual GPU.
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
# GB10 (Grace-Blackwell) has only 16-20 physical cores; a full vLLM build with
# MAX_JOBS=20 thrashes the node (load >40) and starves sshd. 8 keeps the
# compile near-saturated without wedging the box.
MAX_JOBS="${MAX_JOBS:-8}"
IMAGE_NAME="${IMAGE_NAME:-vllm-vision-dspark:${VLLM_PIN:0:10}}"
WORKDIR="${WORKDIR:-/tmp/vllm-vision-build}"

# 0005: vLLM main's VllmConfig validator rejects ALL nvfp4* KV dtypes when the
# model uses MLA. nvfp4_ds_mla (community patch 0004) is exactly an MLA cache
# format (fp8_ds_mla packed layout), so exempt the *_ds_mla suffix there.
# The lucamotz branch supersedes 0001/0002 (vision + streaming + dspark width
# + follow-up fixes are all in the pinned tree). 0004/0005 remain: the
# community nvfp4_ds_mla KV dtype and the nvfp4+MLA validator exemption are
# not upstream.
PATCHES=(0004-nvfp4-ds-mla-kv.patch 0005-nvfp4-ds-mla-mla-validator.patch 0006-kv-spec-block64-clamp.patch)

echo "==> vLLM vision image build"
echo "    pin:      $VLLM_PIN"
echo "    base:     $BASE_IMAGE (arm64)"
echo "    arch:     $TORCH_CUDA_ARCH_LIST"
echo "    image:    $IMAGE_NAME"
echo "    workdir:  $WORKDIR"

mkdir -p "$WORKDIR"
cd "$WORKDIR"

if [ ! -d vllm/.git ]; then
  echo "==> cloning vLLM main"
  git clone --quiet --depth 1 "$VLLM_REPO" vllm
fi
cd vllm
git fetch --quiet --depth 1 origin "$VLLM_PIN"
git checkout --quiet "$VLLM_PIN"
git log --oneline -1

echo "==> applying recipe patches"
for PATCH in "${PATCHES[@]}"; do
  if git apply --check "$RECIPE_DIR/patches/$PATCH" 2>/dev/null; then
    git apply "$RECIPE_DIR/patches/$PATCH"
  else
    # context can drift against a newer tree; fall back to patch(1) with fuzz
    patch -p1 --fuzz=3 < "$RECIPE_DIR/patches/$PATCH"
  fi
done

echo "==> verifying patched tree"
grep -q "DeepseekV4ForConditionalGeneration" vllm/model_executor/models/registry.py \
  || { echo "FATAL: vision class not registered"; exit 1; }
grep -q "nvfp4_ds_mla" vllm/config/cache.py \
  || { echo "FATAL: nvfp4_ds_mla KV missing"; exit 1; }
grep -q "_ds_mla" vllm/config/vllm.py \
  || { echo "FATAL: nvfp4+MLA validator exemption missing (patch 0005)"; exit 1; }
test -f vllm/models/deepseek_v4/common/mm_preprocess.py \
  || { echo "FATAL: vision preprocessor missing"; exit 1; }
grep -q "DeepseekV4VLProcessingInfo" vllm/models/deepseek_v4/common/mm_preprocess.py \
  || { echo "FATAL: vision processor class missing"; exit 1; }
find . -name '*.rej' -delete
if grep -rEl '^(<<<<<<<|>>>>>>>)' vllm/ 2>/dev/null; then
  echo "FATAL: merge conflict markers remain"; exit 1
fi

echo "==> docker build (pip-install patched source over $BASE_IMAGE)"
cat > Dockerfile.vision <<EOF
FROM $BASE_IMAGE
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      git rsync cmake ninja-build patchelf \
      libcusparse-dev-13-0 libcusolver-dev-13-0 cuda-nvrtc-dev-13-0 \
     && rm -rf /var/lib/apt/lists/*
# Official CUDA 13 images ship only versioned libnvrtc.so.13; CMake looks for
# an unversioned CUDA_nvrtc_LIBRARY. Create the symlinks CMake expects.
RUN ln -sf /usr/local/cuda/lib64/libnvrtc.so.13 /usr/local/cuda/lib64/libnvrtc.so \
 && ln -sf /usr/local/cuda/lib64/libnvrtc-builtins.so.13.0 /usr/local/cuda/lib64/libnvrtc-builtins.so
WORKDIR /workspace
COPY . /workspace/
# Rebuild vLLM from the patched source: compiles the modified CUDA kernels
# (topk_softplus_sqrt etc.) for SM121. --no-build-isolation keeps the
# toolchain from the base image.
# b12x: NVIDIA CuTe DSL kernels (NVFP4 GEMM/MoE) that vLLM's --moe-backend
# b12x requires. The official base image does NOT ship it; without it the
# mxfp4 oracle rejects B12X at boot ("kernel does not support current device").
# vLLM main tracks the current b12x API — pin the latest release.
RUN python3 -m pip install --no-cache-dir b12x==1.3.0
# flashinfer-python 0.6.18 adds the DSV4 sparse MLA decode specializations for
# top-k 192/256 (flashinfer-ai/flashinfer#4380, merged 2026-08-08) that the
# FLASHINFER_MLA_SPARSE_DSV4 backend requires with DSpark (k=5 widens the
# index width to 192). The base image ships 0.6.16.post3 (cut the same day the
# PR merged — one release too early). Companion flashinfer-cubin 0.6.18 was
# never published to PyPI; the runtime sets FLASHINFER_DISABLE_VERSION_CHECK=1
# (see docker-compose.yml) and the kernel JIT-compiles on first use.
RUN python3 -m pip install --no-cache-dir flashinfer-python==0.6.18
RUN python3 -m pip install --no-cache-dir "cmake>=3.26.1" "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8" "setuptools-rust>=1.9.0" wheel packaging ninja jinja2 regex protobuf \
 && SETUPTOOLS_SCM_PRETEND_VERSION=0.28.0.post1.dev0+vision \
    TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST MAX_JOBS=$MAX_JOBS \
     VLLM_TARGET_DEVICE=cuda \
    python3 -m pip install --no-build-isolation --no-deps -e /workspace
RUN python3 -c "import vllm; print('vllm', vllm.__version__)" \
    && python3 -c "from vllm.models.deepseek_v4.common.mm_preprocess import DeepseekV4VLProcessingInfo; print('vision processor OK')" \
    && python3 -c "from vllm.config.cache import CacheDType; assert 'nvfp4_ds_mla' in __import__('typing').get_args(CacheDType); print('nvfp4_ds_mla OK')" \
    && python3 -c "from vllm.utils.b12x import has_b12x, get_b12x_fused_moe; assert has_b12x() and get_b12x_fused_moe() is not None; print('b12x kernels OK')"
EOF

docker build -f Dockerfile.vision -t "$IMAGE_NAME" .
docker run --rm --entrypoint python3 "$IMAGE_NAME" -c \
  "import vllm; print('final image OK', vllm.__version__)"

echo "==> done: $IMAGE_NAME"
echo "    Repeat on the worker node (or push to a registry), then set"
echo "    docker-compose.yml image: to it."
