#!/usr/bin/env bash
# glm-v53-flash-4x-noswitch — Phase 1b step 2 (fresh nodes): unify the driver on
# 580.173.02 with the plain kernel-module build 6.17.0-1032.32, matching spark-0f0b
# and spark-6d14 exactly.
#
# Why: the hwe module meta at the 6.17 line exists only as the .32 build (exact-pinned
# to driver 580.173.02); the .32+1 build pairs with 580.178.04 but has no 6.17-era meta,
# so upgrading the older nodes to 178.04 would force the 7.x kernel back in. The archive
# still carries 580.173.02 in noble-security, so the fresh nodes downgrade instead.
#
# Run as root:  ssh <node> 'sudo -n bash -s' < driver-align-173.sh
# A reboot is required afterwards (loaded module is still the +1 build until then).
set -euo pipefail

K=6.17.0-1032
V_OLD=580.173.02-0ubuntu0.24.04.1
V_NEW=580.178.04-0ubuntu0.24.04.1

echo "== apt update"
apt-get update -qq

echo "== unhold the module build so it can be downgraded in the same transaction"
apt-mark unhold linux-modules-nvidia-580-open-${K}-nvidia || true

echo "== downgrade driver components AND swap the module build in one transaction"
PINS=" linux-modules-nvidia-580-open-${K}-nvidia=6.17.0-1032.32"
for p in $(dpkg -l | awk -v v="$V_NEW" '$1=="ii" && $3==v {print $2}'); do
  case "$p" in
    nvidia-firmware-580-*) echo "skip firmware (name-versioned, handled below): $p" ;;
    *) PINS+=" $p=$V_OLD" ;;
  esac
done
echo "pins:${PINS}"
apt-get install -y --allow-downgrades $PINS

echo "== install the 173.02 firmware, purge the 178.04 firmware"
apt-get install -y --allow-downgrades nvidia-firmware-580-580.173.02=${V_OLD}
apt-get purge -y nvidia-firmware-580-580.178.04 || true

echo "== hold all driver components and the module set (site freeze until the 7.x"
echo "   kernel regression is fixed upstream; see build.md Phase 1b)"
dpkg -l | awk -v v="$V_OLD" '$1=="ii" && $3==v && $2 ~ /nvidia/ {print $2}' | xargs -r apt-mark hold
apt-mark hold linux-modules-nvidia-580-open-${K}-nvidia

echo "== consistency"
dpkg --audit
apt-get check
dpkg -l | grep -E '^ii  (nvidia-|libnvidia-)' | awk '{print $2, $3}' | grep 580 | head -12
echo "== done — reboot the node now (loaded module is still the +1 build until then)"
