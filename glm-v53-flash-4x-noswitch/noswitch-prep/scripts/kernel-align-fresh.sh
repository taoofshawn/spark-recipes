#!/usr/bin/env bash
# glm-v53-flash-4x-noswitch — Phase 1b (v2, idempotent): align a fresh DGX Spark
# (ships kernel 7.0.0-1019-nvidia, which breaks switchless-ring NCCL/RoCE per forum
# thread 383023) onto the site kernel 6.17.0-1032-nvidia, driver 580.178.04.
#
# Run as root:  ssh <node> 'sudo -n bash -s' < kernel-align-fresh.sh
# Node must be idle. A reboot is required afterwards (run it separately).
#
# Lessons baked in (learned on spark-6d90, 2026-09-26):
#   * The hwe-24.04 metas have NO 6.17.0-1032 versions in the fresh node's apt index —
#     skip them entirely and purge the surviving 7.x-tracking meta, so nothing pulls 7.x.
#   * Purging the 7.x module packages cascade-removes nvidia-driver-580-open and the
#     metas; that is INTENDED here. The driver components (nvidia-utils-580,
#     nvidia-kernel-common-580, libnvidia-*, nvidia-firmware-580-580.178.04) survive and
#     constitute a complete working 178.04 userspace — verified against spark-0f0b.
#     Do NOT reinstall the metapackage: it depends on the module meta, whose candidate
#     is the 7.x build and would drag the regression kernel back in.
#   * Purging the RUNNING kernel prompts a guard ("Abort kernel removal?") that reads
#     stdin; we answer "no" (do not abort) via a pipe. Safe: the 6.17.0-1032 kernel is
#     fully installed first and is the only kernel left afterwards, so it is the
#     unambiguous GRUB default.
set -euo pipefail

K=6.17.0-1032
echo "== target kernel: ${K}-nvidia (module build .32+1 paired with driver 580.178.04)"

echo "== apt update"
apt-get update -qq

echo "== install/complete the ${K} kernel set (no-op if already present)"
apt-get install -y \
  linux-image-${K}-nvidia linux-modules-${K}-nvidia \
  linux-modules-nvidia-580-open-${K}-nvidia linux-modules-nvidia-fs-${K}-nvidia \
  linux-headers-${K}-nvidia linux-tools-${K}-nvidia \
  linux-nvidia-6.17-headers-${K} linux-nvidia-6.17-tools-${K}

echo "== purge non-target kernels (7.0.0-1019, 6.17.0-1014) incl. their module packages"
echo "   and the 7.x-tracking meta linux-tools-nvidia-hwe-24.04"
OLDS=$(dpkg -l | awk '$1 ~ /^(ii|rc)/ && $2 ~ /^linux-/ && $2 ~ /-(7\.0\.0-1019|6\.17\.0-1014)/ {print $2}')
OLDS+=" linux-tools-nvidia-hwe-24.04"
echo "removing:${OLDS}"
if [ -n "$OLDS" ]; then
  export DEBIAN_FRONTEND=noninteractive
  printf 'no\n%.0s' $(seq 1 40) | apt-get purge -y $OLDS
fi

echo "== hold the versioned kernel packages (blocks any reintroduction of 7.x)"
apt-mark hold linux-image-${K}-nvidia linux-modules-${K}-nvidia \
  linux-modules-nvidia-580-open-${K}-nvidia linux-modules-nvidia-fs-${K}-nvidia \
  linux-headers-${K}-nvidia linux-tools-${K}-nvidia \
  linux-nvidia-6.17-headers-${K} linux-nvidia-6.17-tools-${K}

echo "== consistency checks"
dpkg --audit
apt-get check
echo "== kernels left on disk:"; ls /boot | grep vmlinuz || true
echo "== driver components:"
nvidia-smi --query-gpu=driver_version --format=csv,noheader || echo "nvidia-smi unavailable (expected until reboot into ${K})"
echo "== done — reboot the node now (6.17.0-1032 will be the only/default kernel)"
