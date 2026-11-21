#!/usr/bin/env bash
# preflight.sh — read-only fabric/host verification for THIS recipe only.
#
# Safe to run while a lane is serving: it reads, never writes, and never starts, stops or
# reconfigures anything. Run it before every bring-up and before any switch.
# Exit 0 = ALL CHECKS PASS.
#
# EVERY expectation comes from this recipe's own files:
#   <fabric>/cluster.env                 topology, ring peers, NCCL SHAs, pins, digest
#   <fabric>/$NETPLAN_DIR_REL/<node>/    the committed rendered expectation (two files/node)
#   <fabric>/check_switchless_nccl.py    the NCCL artifact identity check
# Nothing here reads a sibling recipe directory. This script is byte-identical in every
# recipe; the two path keys in cluster.env are what differ.
#
# Checks, per node (SSH, read-only):
#   1. the committed netplan == what the generator renders from cluster.env
#   2. live /etc/netplan/40-cx7.yaml and /etc/default/tp4-fabric-iptables == committed
#   3. every fabric interface UP with its expected address
#   4. the ping mesh across the ring's links, both PCIe views where addressed
#   5. NCCL staging: the SHA(s) this recipe loads, verified as AArch64 ELF artifacts
#   6. kernel + driver match the pins; version-locked packages are held
#   7. docker reachable, rdma-core present, /dev/infiniband populated
#   8. CONTRACT_DIGEST printed (must be identical in every switchless recipe)
#
# Environment:
#   PREFLIGHT_SKIP_HOST_PINS=1   skip check 6 (off-cluster / CI use)

set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
TP4_LOG_TAG='[preflight]'
log()  { printf '%s %s\n' "$TP4_LOG_TAG" "$*"; }
warn() { printf '%s WARN: %s\n' "$TP4_LOG_TAG" "$*" >&2; }
fail() { printf '%s FAIL: %s\n' "$TP4_LOG_TAG" "$*" >&2; EXIT=1; }

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8)
EXIT=0

case "${1:-}" in
  -h|--help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  '') ;;
  *) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2 ;;
esac

# ---------------------------------------------------------------- facts
# shellcheck source=/dev/null
. "$HERE/cluster.env" || { fail "cannot read $HERE/cluster.env"; exit 1; }
read -r -a ALIASES <<<"$NODES"
[ "${#ALIASES[@]}" -eq 4 ] || { fail "NODES must name four nodes (got ${#ALIASES[@]})"; exit 1; }

# Host pins (kernel/driver + the version-locked package list) live in this recipe's own
# vendored versions.env — one source of truth shared with bootstrap-node.sh.
PINS_FILE="$HERE/${PINS_REL:-scripts/node/bootstrap/versions.env}"
if [ -f "$PINS_FILE" ]; then
  # shellcheck source=/dev/null
  . "$PINS_FILE"
  HAVE_PINS=1
else
  HAVE_PINS=0
  warn "no pins file at $PINS_FILE — the kernel/driver check will be skipped"
fi

NETPLAN_DIR="$HERE/${NETPLAN_DIR_REL:-netplan}"
[ -d "$NETPLAN_DIR" ] || { fail "committed netplan dir missing: $NETPLAN_DIR"; exit 1; }

# Does this recipe keep a 2-PF artifact? Drives check 5's expectation set only.
#   nvfp4 : site/nccl-2pf/                      (vendored rollback tree)
#   fp8   : upstream/scripts/node/nccl/         (its own, always had it)
#   deepseek: none — its rollback is a donor-image extraction, not a build tree
has_2pf_tree() {
  [ -d "$HERE/nccl-2pf" ] || [ -d "$HERE/../nccl-2pf" ] \
    || [ -d "$HERE/../upstream/scripts/node/nccl" ]
}
if has_2pf_tree && [ -n "${NCCL_2PF_SHA256:-}" ]; then
  NCCL_ARTIFACTS=(".local/lib/nccl-patched-4pf/libnccl.so.2.30.7:${NCCL_4PF_SHA256:-}" \
                  ".local/lib/nccl-patched/libnccl.so.2.30.7:${NCCL_2PF_SHA256}")
else
  NCCL_ARTIFACTS=(".local/lib/nccl-patched-4pf/libnccl.so.2.30.7:${NCCL_4PF_SHA256:-}")
fi

# ---------------------------------------------------------------- 1. generator --check
RENDER="$HERE/${RENDER_REL:-scripts/render-netplan.sh}"
if [ -f "$RENDER" ]; then
  if out=$(cd "$HERE" && bash "$RENDER" --check --out "$NETPLAN_DIR" 2>&1); then
    log "generator --check: PASS (committed netplan == rendered from cluster.env)"
  else
    fail "generator --check FAILED — fix the committed netplan before booting anything"
    printf '%s\n' "$out" | sed 's/^/    /' >&2
  fi
else
  warn "no generator at $RENDER — skipping the render consistency check"
fi

# ---------------------------------------------------------------- per node
r=0
for alias in "${ALIASES[@]}"; do
  if [ "$alias" = "$(hostname -s 2>/dev/null)" ]; then host="$alias"; else host="$alias.shawndo.intra"; fi
  NP="$NETPLAN_DIR/$alias/40-cx7.yaml"
  FE="$NETPLAN_DIR/$alias/tp4-fabric-iptables.env"

  # --- 2. live config == committed ----------------------------------------------
  live_np=$(ssh "${SSH_OPTS[@]}" "$host" "sudo -n grep -vE '^[[:space:]]*#|^[[:space:]]*\$' /etc/netplan/40-cx7.yaml" 2>/dev/null)
  if [ -z "$live_np" ]; then
    fail "$alias: cannot read /etc/netplan/40-cx7.yaml (passwordless sudo?)"
  elif [ "$live_np" = "$(grep -vE '^[[:space:]]*#|^[[:space:]]*$' "$NP")" ]; then
    log "$alias: live netplan == committed"
  else
    fail "$alias: live netplan DIFFERS from committed — re-deploy host config from this recipe"
  fi

  live_fe=$(ssh "${SSH_OPTS[@]}" "$host" "sudo -n grep -vE '^[[:space:]]*#|^[[:space:]]*\$' /etc/default/tp4-fabric-iptables" 2>/dev/null)
  if [ -z "$live_fe" ]; then
    fail "$alias: cannot read /etc/default/tp4-fabric-iptables"
  elif [ "$live_fe" = "$(grep -vE '^[[:space:]]*#|^[[:space:]]*$' "$FE")" ]; then
    log "$alias: live iptables env == committed"
  else
    fail "$alias: live /etc/default/tp4-fabric-iptables DIFFERS from committed"
  fi

  # --- 3. interfaces UP with expected addresses ---------------------------------
  state=$(ssh "${SSH_OPTS[@]}" "$host" "ip -br a" 2>/dev/null)
  if [ -z "$state" ]; then fail "$alias: cannot read interface state"; r=$((r+1)); continue; fi
  # Interface stanza = a 4-space-indented "<name>:" line; its addresses are the
  # "- <ip>/24" entries that follow. Whitespace-tolerant on purpose: the renderer's
  # exact indent has changed before and must not silently disable this check.
  mapfile -t pairs < <(awk '
      /^[[:space:]]*[A-Za-z0-9_.-]+:[[:space:]]*$/ {
        line=$0; sub(/^[[:space:]]+/,"",line); sub(/:[[:space:]]*$/,"",line)
        if (line !~ /^(network|version|renderer|ethernets|addresses|dhcp4|dhcp6|mtu|optional|routes|nameservers|parameters|match)$/) iface=line
        next
      }
      /^[[:space:]]*-[[:space:]]*[0-9.]+\/[0-9]+/ {
        a=$2; sub(/\/.*/,"",a); if (iface!="") print iface, a
      }' "$NP")
  n_up=0
  for pair in "${pairs[@]}"; do
    set -- $pair; ifc=$1; adr=$2
    if printf '%s\n' "$state" | grep -F "$ifc" | grep -qF "$adr"; then
      n_up=$((n_up+1))
    else
      fail "$alias: $ifc not UP @ $adr"
    fi
  done
  log "$alias: fabric addresses UP $n_up/${#pairs[@]}"

  # --- 4. ping mesh -------------------------------------------------------------
  ok=0; total=0
  for pair in "${pairs[@]}"; do
    set -- $pair; adr=$2; prefix=${adr%.*}
    peer_f=""
    for peer in ${FABRIC_TARGETS[$r]}; do
      [ "${peer%.*}" = "$prefix" ] && peer_f=$peer
    done
    [ -n "$peer_f" ] || continue
    # The P2 view of the same link: local octet is f+2, so its peer is peer_f+2.
    if [ "${adr##*.}" -eq $(( ${peer_f##*.} + 2 )) ] 2>/dev/null; then
      target="${prefix}.$(( ${peer_f##*.} + 2 ))"
    else
      target="$peer_f"
    fi
    total=$((total+1))
    if ssh "${SSH_OPTS[@]}" "$host" "ping -c1 -W2 -I $adr $target >/dev/null 2>&1" 2>/dev/null; then
      ok=$((ok+1))
    else
      fail "$alias: no ping $adr -> $target"
    fi
  done
  log "$alias: ping mesh $ok/$total"

  # --- 5. NCCL staging identity -------------------------------------------------
  for pair in "${NCCL_ARTIFACTS[@]}"; do
    rel=${pair%%:*}; want=${pair#*:}
    if [ -z "$want" ]; then fail "$alias: no SHA pinned for $rel"; continue; fi
    got=$(ssh "${SSH_OPTS[@]}" "$host" "sha256sum \$HOME/$rel 2>/dev/null | cut -d' ' -f1" 2>/dev/null)
    if [ "$got" = "$want" ]; then
      log "$alias: $rel SHA OK (${want:0:8}…)"
    else
      fail "$alias: $rel missing or SHA mismatch (got ${got:-none}, want ${want:0:8}…)"
    fi
  done

  # --- 6. kernel/driver pins + holds --------------------------------------------
  if [ "${PREFLIGHT_SKIP_HOST_PINS:-0}" = 1 ]; then
    log "$alias: host pin check skipped (PREFLIGHT_SKIP_HOST_PINS=1)"
  elif [ "${HAVE_PINS:-0}" = 1 ] && [ -n "${KERNEL:-}" ]; then
    kr=$(ssh "${SSH_OPTS[@]}" "$host" "uname -r" 2>/dev/null)
    dv=$(ssh "${SSH_OPTS[@]}" "$host" "nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1" 2>/dev/null)
    [ "$kr" = "$KERNEL" ] && log "$alias: kernel $kr OK" \
      || fail "$alias: kernel '${kr:-unknown}' != pinned $KERNEL (7.x breaks ring NCCL — do not upgrade)"
    if [ -z "${DRIVER:-}" ] || [ "$dv" = "$DRIVER" ]; then
      log "$alias: driver ${dv:-unknown} OK"
    else
      fail "$alias: driver '$dv' != pinned $DRIVER"
    fi
    if [ -n "${KERNEL_PKGS:-}" ]; then
      holds=$(ssh "${SSH_OPTS[@]}" "$host" "apt-mark showhold 2>/dev/null | wc -l" 2>/dev/null)
      [ "${holds:-0}" -gt 0 ] && log "$alias: $holds package(s) held" \
        || warn "$alias: apt-mark showhold is empty — are the version holds applied?"
    fi
  else
    log "$alias: no KERNEL pin in cluster.env — host pin check skipped"
  fi

  # --- 7. docker + rdma + infiniband --------------------------------------------
  if ssh "${SSH_OPTS[@]}" "$host" "docker info >/dev/null 2>&1 || sudo -n docker info >/dev/null 2>&1" 2>/dev/null; then
    log "$alias: docker reachable"
  else
    fail "$alias: docker not reachable (daemon down or user not permitted)"
  fi
  missing=$(ssh "${SSH_OPTS[@]}" "$host" "for p in rdma ibv_devinfo; do command -v \$p >/dev/null 2>&1 || echo \$p; done" 2>/dev/null)
  [ -z "$missing" ] && log "$alias: rdma-core tooling present" \
    || fail "$alias: missing tool(s): $(printf '%s' "$missing" | tr '\n' ' ')"
  n_ib=$(ssh "${SSH_OPTS[@]}" "$host" "ls /dev/infiniband 2>/dev/null | wc -l" 2>/dev/null)
  [ "${n_ib:-0}" -ge 2 ] && log "$alias: /dev/infiniband populated ($n_ib)" \
    || fail "$alias: /dev/infiniband has ${n_ib:-0} entries (expected >= 2)"

  r=$((r+1))
done

# ---------------------------------------------------------------- 8. digest
if [ -n "${CONTRACT_DIGEST:-}" ]; then
  log "CONTRACT_DIGEST=${CONTRACT_DIGEST}"
  log "  compare against every other switchless recipe's cluster.env — one line, must match"
else
  fail "CONTRACT_DIGEST is not set in cluster.env"
fi

if [ "$EXIT" -eq 0 ]; then
  log "ALL CHECKS PASS"
else
  log "FAILURES above — resolve before any bring-up; the failure text names the fix"
fi
exit "$EXIT"