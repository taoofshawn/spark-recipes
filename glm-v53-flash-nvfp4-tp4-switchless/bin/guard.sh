#!/usr/bin/env bash
# guard.sh — "is anything serving right now?" — read-only contention check.
#
# Sourced by this recipe's bin/start.sh. It answers one question: may this recipe start?
# It never starts, stops, removes, or reconfigures anything.
#
# WHY IT EXISTS: only one lane can serve at a time (every recipe uses all reserved GPUs,
# and all of them bind :8000). Instead of a shared "lane switching" document, the
# precondition is executable and cannot go stale.
#
# FAIL-CLOSED (owner decision, 2026-11-20): if a node's state cannot be established
# (unreachable, docker wedged, ssh timeout) the answer is NO — starting blind is how a
# half-launched cluster happens, and that costs a boot plus manual cleanup on every node
# that did come up. Override deliberately with --allow-unreachable.
#
# The container names/patterns below are KNOWLEDGE for contention detection, not a path
# dependency: this file works with every sibling recipe directory deleted, and never reads
# one. Entries marked "self" are this recipe's own and are ignored unless --include-self.
#
# Requires from the caller (set before sourcing):
#   GUARD_FACTS   absolute path to this recipe's cluster.env
#   GUARD_LAUNCHER  absolute path to the recipe's launcher (for the remediation hint)
# Optional: GUARD_PORT (default 8000)

[[ -n "${GUARD_FACTS:-}" && -f "${GUARD_FACTS:-}" ]] || {
  printf '[guard] ERROR: GUARD_FACTS is unset or missing (%s)\n' "${GUARD_FACTS:-unset}" >&2
  return 2 2>/dev/null || exit 2
}

# shellcheck source=/dev/null
. "$GUARD_FACTS"

GUARD_PORT=${GUARD_PORT:-${API_PORT:-8000}}
GUARD_SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8)
GUARD_NODES=(${NODES:-})

# Known container patterns across every switchless recipe on this host set. Each entry is
# "<glob>|<recipe label>|<optional: exact names>".
# Short labels only — never a sibling DIRECTORY name, so this file stays valid when a
# sibling recipe is deleted (A1). "knowing the container name" is not "depending on that
# recipe": nothing here reads another recipe's files.
GUARD_KNOWN=(
  "glm53_fp8_dflash_tp4|fp8-lane|"
  "glm53-nvfp4-r*|nvfp4-lane|"
  "dsv41-*|deepseek-lane|dsv41-head dsv41-worker"
)

guard_log()  { printf '[guard] %s\n' "$*"; }
guard_bad()  { printf '[guard] BLOCKED: %s\n' "$*" >&2; }

# guard_check [--include-self]
#   0 = clear to start; 1 = something is running; 2 = state unknown (fail-closed)
guard_check() {
  local include_self=0 allow_unreachable=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --include-self)      include_self=1 ;;
      --allow-unreachable) allow_unreachable=1 ;;
      *) printf '[guard] unknown option %s\n' "$1" >&2; return 2 ;;
    esac
    shift
  done

  local self_pat="${CONTAINER_PATTERN:-}" self_names="${CONTAINER_NAMES:-}"
  local rc=0 node busy="" unknown="" line pat label names nm

  for node in "${GUARD_NODES[@]}"; do
    local host="$node.shawndo.intra"
    [ "$node" = "$(hostname -s 2>/dev/null)" ] && host="$node"

    # --- one round trip: running container names only (stopped ones are not serving) ---
    if ! line=$(ssh "${GUARD_SSH_OPTS[@]}" "$host" \
          "docker ps --format '{{.Names}}' 2>/dev/null || sudo -n docker ps --format '{{.Names}}' 2>/dev/null" 2>/dev/null); then
      unknown="$unknown $host"; continue
    fi

    local running; running=$(printf '%s\n' "$line" | tr -d '\r')

    # --- is :PORT bound? (the engine binds on the head; harmless to check everywhere) ---
    local portbound
    portbound=$(ssh "${GUARD_SSH_OPTS[@]}" "$host" \
      "ss -ltn 2>/dev/null | awk '\$4 ~ /:${GUARD_PORT}\$/ {print \$4; exit}'" 2>/dev/null) || portbound=""

    for entry in "${GUARD_KNOWN[@]}"; do
      pat=${entry%%|*}; rest=${entry#*|}; label=${rest%%|*}; names=${rest#*|}
      [ "$names" = "$label" ] && names=""
      # skip this recipe's own containers unless asked
      if [ "$include_self" = 0 ] && [ "$pat" = "$self_pat" ]; then continue; fi
      while IFS= read -r nm; do
        [ -n "$nm" ] || continue
        # shellcheck disable=SC2254
        case "$nm" in
          $pat) busy="$busy $host:$nm($label)" ;;
        esac
      done <<<"$running"
      # exact-name list, when the recipe has one (globs can be too broad)
      if [ -n "$names" ]; then
        for nm in $names; do
          if printf '%s\n' "$running" | grep -qxF "$nm"; then
            [ "$include_self" = 1 ] || [ "$pat" != "$self_pat" ] || continue
            case "$busy" in *"$host:$nm("* ) continue ;; esac
            busy="$busy $host:$nm($label)"
          fi
        done
      fi
    done

    # :PORT bound with no identified container = contention. Distinguish this recipe's OWN
    # still-running lane (a previous start) from an unknown owner: the remedy differs.
    if [ -n "$portbound" ] && [ -z "$(printf '%s' "$busy" | grep -F "$host:")" ]; then
      if [ "$include_self" = 0 ] && [ -n "$self_pat" ]; then
        while IFS= read -r nm; do
          [ -n "$nm" ] || continue
          # shellcheck disable=SC2254
          case "$nm" in
            $self_pat) busy="$busy $host:$nm(THIS-RECIPE_already_running)" ;;
          esac
        done <<<"$running"
      fi
      [ -n "$(printf '%s' "$busy" | grep -F "$host:")" ] \
        || busy="$busy $host::$GUARD_PORT(port in use, owner unidentified)"
    fi
  done

  if [ -n "$busy" ]; then
    guard_bad "something is already serving:"
    local b
    for b in $busy; do printf '           %s\n' "$b" >&2; done
    case "$busy" in
      *"THIS-RECIPE"*)
        printf '[guard] this recipe is ALREADY RUNNING — stop it before starting again:\n' >&2
        printf '[guard]   ./bin/stop.sh\n' >&2 ;;
      *)
        printf '[guard] stop it with that recipe'"'"'s own bin/stop.sh, e.g.\n' >&2
        printf '[guard]   cd <that recipe> && ./bin/stop.sh\n' >&2 ;;
    esac
    printf '[guard] this recipe will NOT start until nothing else is running.\n' >&2
    if [ -n "$unknown" ]; then
      printf '[guard] ALSO unknown (unreachable/docker wedged):%s — resolve before retrying.\n' "$unknown" >&2
    fi
    return 1
  fi

  if [ -n "$unknown" ]; then
    if [ "$allow_unreachable" = 1 ]; then
      guard_log "state unknown on:${unknown} — continuing because --allow-unreachable was given"
      return 0
    fi
    guard_bad "cannot determine state on:${unknown}"
    printf '[guard] fail-closed by design: starting blind risks a half-launched cluster.\n' >&2
    printf '[guard] fix reachability (ssh / docker / sudo -n), or re-run with --allow-unreachable\n' >&2
    printf '[guard] if you are deliberately starting on a subset of nodes.\n' >&2
    return 2
  fi

  guard_log "clear: no other lane serving on ${#GUARD_NODES[@]} node(s), :$GUARD_PORT free"
  return 0
}