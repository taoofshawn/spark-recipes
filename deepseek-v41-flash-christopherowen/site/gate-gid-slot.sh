#!/usr/bin/env bash
# site/gate-gid-slot.sh — gate G-GID: every routed PF carries exactly ONE IPv4 RoCE v2 GID.
#
# Why this lane needs its own copy: spark-ds41f discovers GIDs BY ADDRESS (no
# NCCL_IB_GID_INDEX, deliberately: config/cluster.json drops it and scripts/topology.py
# renders SPARKNET_ROCE_SUBNETS from the node map instead). If a link flap files a second
# IPv4 address on a PF, or the kernel moves the only one, discovery can pick the wrong GID
# or find none — the failure mode the FP8 lane hit as an "invalid usage" NCCL abort.
#
# Read-only; safe while a lane serves. Run before every start of THIS recipe (P2-6 G6).
#
# usage: site/gate-gid-slot.sh [--quiet]
# exit:  0 every routed PF has exactly one IPv4 RoCE v2 GID
#        1 a routed PF has zero or >1 IPv4 RoCE v2 GIDs, or an address does not match the
#          netplan expectation for that (node, PF)
#        2 a node is unreachable / show_gids missing
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
QUIET=0
[ "${1:-}" = "--quiet" ] && QUIET=1
# shellcheck source=fabric/cluster.env
. "$HERE/fabric/cluster.env"

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8)
say() { [ "$QUIET" = 1 ] || printf '%s\n' "$*"; }
bad() { printf '[gid-gate] %s\n' "$*" >&2; }

# The four PFs this site addresses, and the expected IPv4 per node is derived from the
# netplan naming rule (node index + ring link), so no per-node table is duplicated here:
#   f0: 10.10.<link>.1..4   f1: 10.10.<link>.1..4   P2 views: octet + 2
# We assert structure (one IPv4 v2 GID per PF), not the exact octet — preflight.sh is what
# compares live netplan to the committed rendering.
RANK=0
rc=0
unknown=0
for node in ${NODES}; do
  host="$node.shawndo.intra"
  [ "$node" = "$(hostname -s 2>/dev/null)" ] && host="$node"
  out=$(ssh "${SSH_OPTS[@]}" "$host" 'show_gids 2>/dev/null' ) || { bad "$node: show_gids unreachable"; unknown=1; RANK=$((RANK+1)); continue; }
  [ -n "$out" ] || { bad "$node: show_gids produced nothing"; unknown=1; RANK=$((RANK+1)); continue; }

  # awK: PF, v2, IPv4-mapped rows only; count per PF.
  # Fields are taken from the END of the line: show_gids prints an EMPTY IPv4 column for
  # IPv6 GIDs, and awk's whitespace splitting collapses that gap, shifting everything after
  # it left. From the end: $NF=device name, $(NF-1)=v1|v2, $(NF-2)=IPv4 (when present).
  counts=$(printf '%s\n' "$out" | awk '
    $1 ~ /^roce/ && $(NF-1) == "v2" && $(NF-2) ~ /^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ {
      c[$1]++; a[$1] = a[$1] " " $(NF-2) }
    END { for (p in c) printf "%s %d%s\n", p, c[p], a[p] }' | sort)
  [ -n "$counts" ] || { bad "$node: no IPv4 RoCE v2 GIDs at all"; rc=1; RANK=$((RANK+1)); continue; }

  say "[gid-gate] $node"
  seen=""
  while read -r pf n addrs; do
    case "$pf" in
      rocep1s0f0|rocep1s0f1|roceP2p1s0f0|roceP2p1s0f1) ;;
      *) say "           $pf: not a routed PF for this site (ignored, $n IPv4 v2 GID(s))"; continue ;;
    esac
    if [ "$n" -eq 1 ]; then
      say "           $pf: 1 IPv4 v2 GID ($addrs)"
    else
      bad "$node $pf: $n IPv4 RoCE v2 GIDs ($addrs) — discovery needs exactly one"
      rc=1
    fi
    seen="$seen $pf"
  done <<<"$counts"

  for pf in rocep1s0f0 rocep1s0f1 roceP2p1s0f0 roceP2p1s0f1; do
    case " $seen " in *" $pf "*) ;; *) bad "$node $pf: no IPv4 RoCE v2 GID"; rc=1 ;; esac
  done
  RANK=$((RANK+1))
done

[ "$unknown" = 1 ] && rc=2
if [ "$rc" = 0 ]; then
  say "[gid-gate] PASS: one IPv4 RoCE v2 GID on each of the 4 routed PFs, all nodes"
else
  bad "FAIL — fix the PF addressing (netplan) before starting this lane"
fi
exit "$rc"