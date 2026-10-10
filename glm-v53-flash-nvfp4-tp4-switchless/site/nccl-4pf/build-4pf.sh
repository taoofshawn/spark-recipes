#!/usr/bin/env bash
# build-4pf.sh — Phase E site build of the four-PF switchless NCCL for the 4-node ring.
#
# Runs ON A NODE (AArch64, docker). Self-contained: this directory (scp'd whole to
# $DEST/patches) carries all four patches + the builder Dockerfile; the only network
# fetch is the pinned NCCL source. Mirrors the FP8 recipe's node NCCL build pattern
# (glm-v53-flash-fp8-tp4-switchless/upstream/scripts/node/nccl/build.sh) adapted to the
# othexmr four-patch stack.
#
# Usage: DOCKER=docker ./build-4pf.sh [DEST] [JOBS]
#   DEST   scratch dir on the node (default $HOME/nccl-4pf-build) — NOT the repo
#   JOBS   make parallelism (default 20)
#   DOCKER use 'sudo -n docker' if the node user is outside the docker group
#
# Pinned inputs (see research.md "NCCL four-device verdict" entry):
#   NCCL v2.30.7-1 @ 73cf112295c33aee2b895f329f592f2a9b4b0f97 (base tree 3e7de6f9…)
#   patch order: skip-tree-pat → advertise-all-listener-gids → hardened-switchless
#                → dual-pf.site (the ONLY site-modified patch: tp4ValidateListener
#                GID range 10.100.224–227 → 10.10.1–4 for this ring)
#   site final patched tree: cf67789549df2f90ecd3309c87b9bd9b8b45c79b
# Output: $DEST/nccl/build/lib/libnccl.so.2.30.7 (+ sha256/size/symbols on stdout)

set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DEST=${1:-$HOME/nccl-4pf-build}
JOBS=${2:-20}
DOCKER=${DOCKER:-docker}

NCCL_REPO=https://github.com/NVIDIA/nccl.git
NCCL_TAG=v2.30.7-1
NCCL_COMMIT=73cf112295c33aee2b895f329f592f2a9b4b0f97
BASE_TREE=3e7de6f92f0190d1afe9f05642e634cbf43ae4c9
SITE_TREE=cf67789549df2f90ecd3309c87b9bd9b8b45c79b
IMAGE=nccl-build:cuda13.0.2-u24
GENCODE='-gencode=arch=compute_121,code=sm_121'

mkdir -p "$DEST"

# Patch inventory: the three upstream patches must byte-match othexmr's published
# SHAs; the site patch must byte-match the committed site copy. (cd $HERE: the
# manifest lists bare filenames and the script may be launched from anywhere.)
(
  cd "$HERE"
  sha256sum -c <<'SUMS'
097656d07a5774919f0d51558b51ec05de8168c0097ed6cb7764c33230ba6eb2  nccl-2.30.7-skip-tree-pat.patch
dccfce86d14c15c39f0e0a742863960205a3d9823c464b31a7f7389354844178  nccl-2.30.7-advertise-all-listener-gids.patch
e2dd39eaefc022f99d5a3d3195e81947da20a4c7acbf1d688b4df8f6691c210c  nccl-2.30.7-hardened-switchless.patch
334664f9ac4b432c4b34f2b1301804d59be7b966bb806d4c833e7288a3be5f3b  nccl-2.30.7-dual-pf.site.patch
SUMS
)

# Fresh clone at the pinned tag, renamed only on success (half-clone safety, as FP8 build.sh).
test -d "$DEST/nccl" || {
  rm -rf "$DEST/.nccl.tmp"
  git clone --quiet --branch "$NCCL_TAG" --depth 1 "$NCCL_REPO" "$DEST/.nccl.tmp"
  mv "$DEST/.nccl.tmp" "$DEST/nccl"
}
test "$(git -C "$DEST/nccl" rev-parse HEAD)" = "$NCCL_COMMIT" || { echo "ERROR: NCCL checkout is not $NCCL_COMMIT" >&2; exit 1; }
test "$(git -C "$DEST/nccl" write-tree)" = "$BASE_TREE" || { echo "ERROR: NCCL base tree does not match the provenance pin" >&2; exit 1; }

# A rerun after a successful patch must not double-apply: verify the site tree is already
# in place (idempotent) instead of blindly re-applying.
if [ "$(git -C "$DEST/nccl" write-tree)" != "$SITE_TREE" ]; then
  for p in skip-tree-pat advertise-all-listener-gids hardened-switchless dual-pf.site; do
    git -C "$DEST/nccl" apply --check "$HERE/nccl-2.30.7-$p.patch"
    git -C "$DEST/nccl" apply --index "$HERE/nccl-2.30.7-$p.patch"
  done
fi
test "$(git -C "$DEST/nccl" write-tree)" = "$SITE_TREE" || { echo "ERROR: site patched tree mismatch" >&2; exit 1; }
echo "source tree verified: $SITE_TREE"

# Builder image: same proven Dockerfile as the FP8 recipe's NCCL build (empty context —
# the Dockerfile has no COPY/ADD).
if ! $DOCKER images -q "$IMAGE" | grep -q .; then
  mkdir -p "$DEST/.ctx"
  $DOCKER build -t "$IMAGE" -f "$HERE/Dockerfile.nccl-build" "$DEST/.ctx"
fi

rm -rf "$DEST/nccl/build"
# --user: keep build/ user-owned (the FP8 build's root-owned-tree lesson). HOME redirect:
# the non-root uid has no passwd entry inside the image.
$DOCKER run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$DEST/nccl:/nccl" -w /nccl "$IMAGE" \
  bash -lc "make -j$JOBS src.build NVCC_GENCODE=\"$GENCODE\""

LIB="$DEST/nccl/build/lib/libnccl.so.2.30.7"
test -f "$LIB" || { echo "ERROR: built library missing: $LIB" >&2; exit 1; }
echo "BUILD DONE"
sha256sum "$LIB"
stat -Lc 'size=%s' "$LIB"
echo "symbols=$(nm -D --defined-only "$LIB" | wc -l)"
