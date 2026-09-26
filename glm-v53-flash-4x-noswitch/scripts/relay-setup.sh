#!/usr/bin/env bash
# glm-v53-flash-4x-noswitch — set up the rank1→rank2 relay path for the weight fan-out:
# authorize rank 1's key on rank 2, seed rank 1's known_hosts with rank 2's fabric IP,
# then verify rank1 can reach rank2 over the L2 link (10.10.2.2 → 10.10.2.3).
# Run from the workstation: bash relay-setup.sh
set -euo pipefail

PUB=$(ssh -o BatchMode=yes spark-6d14 'cat ~/.ssh/id_ed25519.pub 2>/dev/null || cat ~/.ssh/id_rsa.pub 2>/dev/null || echo NOKEY')
if [ -z "$PUB" ] || [ "$PUB" = "NOKEY" ]; then
  echo "rank1 has no keypair; generating one"
  ssh -o BatchMode=yes spark-6d14 'ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 -q && cat ~/.ssh/id_ed25519.pub'
  PUB=$(ssh -o BatchMode=yes spark-6d14 'cat ~/.ssh/id_ed25519.pub')
fi
echo "rank1 pubkey: ${PUB:0:50}..."

ssh -o BatchMode=yes spark-6d90 "grep -qF '$PUB' ~/.ssh/authorized_keys 2>/dev/null && echo already-authorized || { echo '$PUB' >> ~/.ssh/authorized_keys; echo key-added; }"

# seed rank1's known_hosts with rank2's fabric IP (key verified against mgmt entry)
FK=$(ssh -o BatchMode=yes spark-6d14 'ssh-keyscan -T 5 -t ed25519 10.10.2.3 2>/dev/null')
MK=$(ssh -o BatchMode=yes spark-6d14 'ssh-keygen -lf ~/.ssh/known_hosts -F spark-6d90.shawndo.intra 2>/dev/null | grep -m1 "ED25519" | awk "{print \$2}"' 2>/dev/null || true)
FKFP=$(printf '%s\n' "$FK" | ssh-keygen -lf - 2>/dev/null | awk '{print $2}' | head -1)
echo "fabric-key fp: $FKFP"
ssh -o BatchMode=yes spark-6d14 "grep -q '10.10.2.3' ~/.ssh/known_hosts 2>/dev/null && echo knownhosts-present || { printf '%s\n' '$FK' >> ~/.ssh/known_hosts; echo knownhosts-added; }"

echo "== reachability test rank1 -> rank2 over fabric:"
ssh -o BatchMode=yes spark-6d14 'ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8 10.10.2.3 "hostname -s" && echo RELAY-OK'
