#!/usr/bin/env bash
# glm-v53-flash-4x-noswitch — Phase 1b step 3 (existing nodes spark-0f0b / spark-6d14):
# they already run kernel 6.17.0-1032-nvidia + driver 580.173.02 (module .32) — add the
# site freeze holds so no future apt upgrade can move them (7.x regression kernel or a
# driver/module split).
#
# Run as root:  ssh <node> 'sudo -n bash -s' < hold-existing.sh
set -euo pipefail
K=6.17.0-1032

apt-mark hold \
  linux-image-${K}-nvidia linux-modules-${K}-nvidia \
  linux-modules-nvidia-580-open-${K}-nvidia linux-modules-nvidia-fs-${K}-nvidia \
  linux-headers-${K}-nvidia linux-tools-${K}-nvidia \
  linux-nvidia-6.17-headers-${K} linux-nvidia-6.17-tools-${K} \
  linux-nvidia-hwe-24.04 linux-image-nvidia-hwe-24.04 linux-headers-nvidia-hwe-24.04 \
  linux-tools-nvidia-hwe-24.04 linux-modules-nvidia-580-open-nvidia-hwe-24.04 \
  linux-modules-nvidia-fs-nvidia-hwe-24.04 2>/dev/null || true

# hold all installed 173.02 driver components
dpkg -l | awk '$1=="ii" && $3=="580.173.02-0ubuntu0.24.04.1" && $2 ~ /nvidia/ {print $2}' | xargs -r apt-mark hold

echo "== state"
uname -r
nvidia-smi --query-gpu=driver_version --format=csv,noheader
dpkg --audit; apt-get check
echo "== held:"; apt-mark showhold | sort | head -40
