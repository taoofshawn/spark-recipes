#!/usr/bin/env bash
# glm-v53-flash-4x-noswitch — verify that the short-name DNS aliases resolve to the same
# host keys already pinned (FQDN) in the workstation's known_hosts, then append the
# short-name entries so the recipe's strict ssh (StrictHostKeyChecking=yes) works with
# short aliases. Fingerprints are compared before anything is written.
set -euo pipefail
KH=~/.ssh/known_hosts

for h in spark-0f0b spark-6d14 spark-6d90 spark-6d24; do
  short=$(ssh-keyscan -t ed25519 "$h" 2>/dev/null | ssh-keygen -lf - 2>/dev/null | awk 'NR==1 {print $2; exit}')
  fqdn=$(ssh-keygen -lf "$KH" -F "$h.shawndo.intra" 2>/dev/null | awk '$2=="ED25519" || $2=="ssh-ed25519" {print $3; exit}')
  if [ -n "$short" ] && [ "$short" = "$fqdn" ]; then
    echo "$h MATCH $short"
    if ! ssh-keygen -lf "$KH" -F "$h" >/dev/null 2>&1; then
      ssh-keyscan -t ed25519 "$h" 2>/dev/null >> "$KH"
      echo "  appended short-name entry"
    else
      echo "  short-name entry already present"
    fi
  else
    echo "$h MISMATCH short=[$short] fqdn=[$fqdn] — NOT appended, stop here"
    exit 1
  fi
done
echo "all four verified"
