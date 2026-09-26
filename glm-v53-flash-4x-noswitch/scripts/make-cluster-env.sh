#!/usr/bin/env bash
# glm-v53-flash-4x-noswitch — create the site cluster.env from the recipe template.
# Run from anywhere; paths are fixed for this workstation setup.
set -euo pipefail
REPO=/mnt/c/Users/sdrew/code/glm-4x-noswitch
cd "$REPO"
cp cluster.env.example cluster.env
sed -i \
  -e 's/^NODES="gx10-a gx10-b gx10-c gx10-d"/NODES="spark-0f0b spark-6d14 spark-6d90 spark-6d24"/' \
  -e 's/^MGMT_IPS="192.0.2.11 192.0.2.12 192.0.2.13 192.0.2.14"/MGMT_IPS="10.69.42.170 10.69.42.171 10.69.42.172 10.69.42.173"/' \
  -e 's/^NODE_HOSTNAMES="gx10-a gx10-b gx10-c gx10-d"/NODE_HOSTNAMES="spark-0f0b spark-6d14 spark-6d90 spark-6d24"/' \
  -e 's/^MASTER_IP="192.0.2.11"/MASTER_IP="10.69.42.170"/' \
  -e "s|^RELAY_DEST=.*|RELAY_DEST='sdrew@10.10.2.3'|" \
  cluster.env
bash -n cluster.env && echo SYNTAX_OK
grep -E '^(NODES|MGMT_IPS|NODE_HOSTNAMES|MASTER_IP|RELAY_DEST)' cluster.env
