#!/usr/bin/env bash
# glm-v53-flash-4x-noswitch — read cable EEPROM serials on the two ADDRESSED CX-7 ports
# (enp1s0f0np0 = left, enp1s0f1np1 = right) to identify which neighbor each port faces.
# Usage: ssh <node> 'sudo -n bash -s' < cable-serials.sh
set -euo pipefail
for p in enp1s0f0np0 enp1s0f1np1; do
  sn=$(ethtool -m "$p" 2>/dev/null | awk '/Vendor SN|Serial/ {print $NF; exit}')
  carrier=$(cat "/sys/class/net/$p/carrier" 2>/dev/null || echo '?')
  echo "$p carrier=$carrier sn=${sn:-unreadable}"
done
