#!/usr/bin/env bash
# check-patch3.sh — fail-closed preflight for the cold-start garble root fix.
#
# Patch 3 (credit @roady001, issue #3) lives in the vLLM scheduler. A pre-Patch-3 image
# (notably `probe-c-p2b`) boots clean, passes smoke tests and serves warm requests
# correctly — it only garbles on COLD prefill. That makes it invisible until production.
#
# Measured 2026-07-30 on 2x DGX Spark with a ~20k-token agent prompt, forced cold prefill:
#     without Patch 3 : 44/44 requests garbled  (k=3, k=5, confidence scheduler, all doc settings)
#     with Patch 3    :  0/28 requests garbled  (k=3 and k=5)
# Warm requests never failed in any configuration, which is exactly why a 5-prompt gate
# passes on a broken deployment.
#
# usage:  ./scripts/check-patch3.sh <container-name> [more containers...]
# exit 0 = Patch 3 present everywhere, exit 1 = missing somewhere.
set -uo pipefail

SCHED=/opt/env/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py
rc=0

if [ $# -eq 0 ]; then
  echo "usage: $0 <container-name> [container-name...]" >&2
  exit 2
fi

for c in "$@"; do
  if ! docker inspect "$c" >/dev/null 2>&1; then
    echo "  ?? $c — container not found"
    rc=1
    continue
  fi
  n=$(docker exec "$c" grep -c is_prefill_chunk "$SCHED" 2>/dev/null || echo 0)
  if [ "${n:-0}" -ge 1 ]; then
    echo "  OK   $c — Patch 3 present ($n guard references)"
  else
    echo "  FAIL $c — PATCH 3 MISSING. Cold prefills will garble."
    echo "       Fix on every node:"
    echo "         cp recipe/overlay/vllm/v1/core/sched/scheduler.py /var/tmp/patch3-scheduler.py"
    echo "       then add to the container run:"
    echo "         -v /var/tmp/patch3-scheduler.py:$SCHED:ro"
    echo "       (or rebuild as dspark-nvfp4-stage-c, which bakes Patch 3 in)"
    rc=1
  fi
done

exit $rc
