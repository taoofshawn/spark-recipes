#!/usr/bin/env bash
set -euo pipefail

# Cross-lane fabric preflight (shawndo, 2026-10-10) — read-only, safe to run while a lane
# is serving. Run BEFORE any lane bring-up or lane switch:
#   bash glm-v53-flash-fp8-tp4-switchless/noswitch-prep/scripts/verify-fabric-p2.sh
#
# Why this exists: both GLM lanes share ONE fabric and ONE netplan. The NVFP4 lane's
# Phase E addressed the second PCIe views (P2) of every cable for its four-PF NCCL
# contract; the FP8 lane uses the f-port subset. A stale netplan regeneration or a
# swapped NCCL staging dir breaks the OTHER lane silently (the breaking lane still
# boots). This script catches that drift before a boot wastes the attempt.
#
# Checks, per node (SSH, read-only):
#   1. live /etc/netplan/40-cx7.yaml == the generated files
#      (upstream/scripts/node/etc/<node>/40-cx7.yaml) — catches both a stale host
#      netplan and a stale generator. Needs passwordless sudo on the nodes.
#   2. every fabric interface in the generated netplan is UP and owns its address.
#   3. the 16-ping mesh across BOTH PCIe views of every cable (negligible ICMP load).
#   4. both NCCL staging dirs exist with the pinned SHA-256:
#        ~/.local/lib/nccl-patched      2-PF SparkRing  afe5f486…  (FP8 proven)
#        ~/.local/lib/nccl-patched-4pf  four-PF         8de68def…  (NVFP4 Phase E)
#      so a lane whose NCCL_DIR points at one of them cannot boot on a swapped lib.
#
# Expectations are NOT duplicated here: they are the generated netplan files, kept
# consistent with cluster.env by `render-netplan.sh --check` (run first, below).

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)                 # glm-v53-flash-fp8-tp4-switchless/noswitch-prep/scripts
LANE_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)                 # glm-v53-flash-fp8-tp4-switchless
REPO=$(cd "$LANE_DIR/.." && pwd)                          # repo root
GEN_DIR="$LANE_DIR/upstream/scripts/node/etc"

NCCL_2PF_SHA256=afe5f48626284eae89988516e450c7f20cc303904ba4b7083b88aa3ca1e9b85f
NCCL_4PF_SHA256=8de68def30bcde20fabded468bd60f0df82ea12d3ead02d5120c7feda7a41a87

log() { printf '[verify-fabric] %s\n' "$*"; }
fail() { printf '[verify-fabric] FAIL: %s\n' "$*" >&2; EXIT=1; }

( cd "$LANE_DIR/upstream" && bash scripts/render-netplan.sh --check >/dev/null ) \
  || { echo "render-netplan.sh --check FAILED — fix generator/generated files first" >&2; exit 1; }
log "generator --check: PASS (generated files consistent with cluster.env)"

# shellcheck disable=SC1090
. "$LANE_DIR/noswitch-prep/cluster.env"
read -r -a ALIASES <<<"$NODES"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8)

EXIT=0
r=0
for alias in "${ALIASES[@]}"; do
  host="$alias.shawndo.intra"

  # --- 1. live netplan == generated expectations ---
  live=$(ssh "${SSH_OPTS[@]}" "$host" "sudo -n grep -vE '^[[:space:]]*#|^[[:space:]]*\$' /etc/netplan/40-cx7.yaml" 2>/dev/null) \
    || { fail "$alias: cannot read /etc/netplan/40-cx7.yaml (passwordless sudo?)"; r=$((r + 1)); continue; }
  if diff <(printf '%s\n' "$live") <(grep -vE '^#|^$' "$GEN_DIR/$alias/40-cx7.yaml") >/dev/null; then
    log "$alias: live netplan == generated"
  else
    fail "$alias: live /etc/netplan DIFFERS from generated — resolve before booting any lane (re-deploy host config from the generated files)"
  fi

  # --- parse generated yaml: ordered stanza -> first address ---
  STANZAS=(); ADDRS=()
  while IFS= read -r line; do
    case "$line" in
      "    "[A-Za-z0-9_.-]*":")
        s=${line%:}                  # stanza line: 4-space indent + name + ':'
        STANZAS+=("${s#"    "}") ;;   # (6-space 'addresses:' lines don't match: they
                                     # start with spaces, outside the class)
      *"        - "*"/24")
        a=${line#*- }                # address line: 8-space indent + '- x.y.z.w/24'
        ADDRS+=("${a%/24}") ;;
    esac
  done < "$GEN_DIR/$alias/40-cx7.yaml"
  [ "${#STANZAS[@]}" -eq "${#ADDRS[@]}" ] && [ "${#STANZAS[@]}" -ge 2 ] \
    || { fail "$alias: cannot parse generated netplan stanzas/addresses"; r=$((r + 1)); continue; }

  # --- 2. interfaces UP with their expected address ---
  state=$(ssh "${SSH_OPTS[@]}" "$host" "ip -br a" 2>/dev/null) || state=""
  i=0
  while [ "$i" -lt "${#STANZAS[@]}" ]; do
    iface=${STANZAS[$i]}; want=${ADDRS[$i]}
    if printf '%s\n' "$state" | grep -F "$iface" | grep -qF "$want"; then
      log "$alias: $iface UP @ $want"
    else
      fail "$alias: $iface not UP @ $want (expected by generated netplan)"
    fi
    i=$((i + 1))
  done

  # --- 3. ping mesh across both views ---
  # View pairing in the generated yaml: stanza 2 (P2) belongs to stanza 0 (f0),
  # stanza 3 (P2) to stanza 1 (f1). Peer f address on each link comes from
  # FABRIC_TARGETS[r] (same /24 prefix); peer P2 = peer f octet + 2.
  # Each test binds the local source address (-I) of the SAME view, so both PCIe
  # views of every cable are genuinely exercised.
  ok=0; total=0
  # View pairing in the generated yaml is ADJACENT: (0,1) = f0 + its P2 view, (2,3) =
  # f1 + its P2 view. Peer f address on each link comes from FABRIC_TARGETS[r] (same
  # /24 prefix); peer P2 = peer f octet + 2. Each test binds the local source address
  # (-I) of the SAME view, so both PCIe views of every cable are genuinely exercised.
  # On a 2-PF-only netplan (no P2 stanzas) the second ping of each pair is skipped:
  # the P2 test only runs when the paired stanza is in the SAME /24 (i.e. it is the
  # second PCIe view of the same link).
  pair_i=0
  while [ "$pair_i" -lt "${#STANZAS[@]}" ]; do
    f_i=$pair_i; p2_i=$((pair_i + 1))
    [ "$p2_i" -lt "${#STANZAS[@]}" ] || break
    f_addr=${ADDRS[$f_i]}; p2_addr=${ADDRS[$p2_i]}
    prefix=${f_addr%.*}
    peer_f=""
    for peer in ${FABRIC_TARGETS[$r]}; do
      [ "${peer%.*}" = "$prefix" ] && peer_f=$peer
    done
    if [ -z "$peer_f" ]; then
      fail "$alias: no FABRIC_TARGETS peer on $prefix — cluster.env/generated files disagree"
    else
      total=$((total + 1))
      if ssh "${SSH_OPTS[@]}" "$host" "ping -c1 -W2 -I $f_addr $peer_f >/dev/null 2>&1" 2>/dev/null; then
        ok=$((ok + 1))
      else
        fail "$alias: no ping $f_addr -> $peer_f"
      fi
      if [ "${p2_addr%.*}" = "$prefix" ]; then
        total=$((total + 1))
        if ssh "${SSH_OPTS[@]}" "$host" "ping -c1 -W2 -I $p2_addr ${prefix}.$(( ${peer_f##*.} + 2 )) >/dev/null 2>&1" 2>/dev/null; then
          ok=$((ok + 1))
        else
          fail "$alias: no ping $p2_addr -> ${prefix}.$(( ${peer_f##*.} + 2 ))"
        fi
      fi
    fi
    pair_i=$((pair_i + 2))
  done
  log "$alias: ping mesh $ok/$total (f + P2 views)"

  # --- 4. NCCL staging dirs ---
  for pair in "nccl-patched:$NCCL_2PF_SHA256" "nccl-patched-4pf:$NCCL_4PF_SHA256"; do
    dir=${pair%%:*}; want=${pair#*:}
    got=$(ssh "${SSH_OPTS[@]}" "$host" "sha256sum \$HOME/.local/lib/$dir/libnccl.so.2.30.7 2>/dev/null | cut -d' ' -f1" 2>/dev/null || true)
    if [ "$got" = "$want" ]; then
      log "$alias: ~/.local/lib/$dir SHA OK (${want:0:8}…)"
    else
      fail "$alias: ~/.local/lib/$dir/libnccl.so.2.30.7 missing or SHA mismatch (got: ${got:-none}, want: ${want:0:8}…)"
    fi
  done
  r=$((r + 1))
done

if [ "$EXIT" -eq 0 ]; then
  log "ALL CHECKS PASS — fabric consistent for either lane"
else
  log "FAILURES above — resolve before any lane bring-up (see the lane READMEs' cross-lane notes)"
fi
exit "$EXIT"
