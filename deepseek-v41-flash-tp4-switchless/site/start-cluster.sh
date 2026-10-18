#!/usr/bin/env bash
# Cluster launcher for deepseek-v41-flash-tp4-switchless: wraps the sparkring per-rank
# launcher (upstream/deepseek_v41_cycle_serve.sh) with this site's SSH mesh so that the
# lane is controlled from the HEAD node like the GLM lanes (./start.sh serve | stop | status | logs).
#
#   ./start-cluster.sh check   — resolve + validate all four rank envs (offline, no docker)
#   ./start-cluster.sh serve   — workers 3,2,1 first, then rank 0 (API host)
#   ./start-cluster.sh stop    — rank 0 first, then workers 3,2,1; containers preserved
#   ./start-cluster.sh status  — per-rank docker ps + /health
#   ./start-cluster.sh logs [rank] [lines]
#
# Transport note: the sparkring launcher's own HCA-count preflight was relaxed to accept
# the four-PF HCA list (SITE MOD #1 in upstream/deepseek_v41_cycle_serve.sh) and the
# four-PF NCCL env lines are added by SITE MOD #2 (see upstream/VENDORED-AT.md). All
# other NCCL values in the site env already match the cycle launcher's required contract.
set -euo pipefail
cd "$(dirname "$0")"

ENV_FILE=${ENV_FILE:-env.switchless}
UPSTREAM_DIR=../upstream
LANCHER=deepseek_v41_cycle_serve.sh
REMOTE_DIR=$HOME/.local/ds41f-tp4/recipe

# shellcheck disable=SC1090
source "$ENV_FILE"
read -r -a HOSTS <<<"$HOSTS"; read -r -a IPS <<<"$IPS"
[[ ${#HOSTS[@]} == 4 && ${#IPS[@]} == 4 ]] || { echo "HOSTS/IPS must list four ranks" >&2; exit 2; }
[[ -n $IMAGE_ID ]] || { echo "IMAGE_ID empty — pin the built image's sha256:... id in $ENV_FILE first" >&2; exit 2; }
render_rank_env() {
  local r=$1
  cat <<EOF
NODE_RANK=$r
MASTER_ADDR=${IPS[0]}
VLLM_HOST_IP=${IPS[$r]}
MODEL_HOST_PATH=$MODEL_DIR
CACHE_HOST_PATH=$CACHE_DIR
PATCH_DIR=$REMOTE_DIR/upstream/patches
NCCL_SO_HOST_PATH=$NCCL_SO_HOST_PATH
IMAGE=$IMAGE
IMAGE_ID=$IMAGE_ID
API_PORT=$API_PORT
MASTER_PORT=$MASTER_PORT
SERVED_MODEL_NAME=$SERVED_MODEL_NAME
MAX_MODEL_LEN=$MAX_MODEL_LEN
MAX_NUM_SEQS=$MAX_NUM_SEQS
MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS
GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION
NUM_SPECULATIVE_TOKENS=$NUM_SPECULATIVE_TOKENS
DRAFT_SAMPLE_METHOD=$DRAFT_SAMPLE_METHOD
ENFORCE_EAGER=$ENFORCE_EAGER
TEXT_ONLY=$TEXT_ONLY
THINKING_DEFAULT=$THINKING_DEFAULT
ENGRAM_DISK_THREADS=$ENGRAM_DISK_THREADS
ENGRAM_DISK_CHUNK=$ENGRAM_DISK_CHUNK
ENGRAM_BALANCED=$ENGRAM_BALANCED
ENGRAM_PACKED_DIR=$ENGRAM_PACKED_DIR
NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME
NCCL_IB_HCA=$NCCL_IB_HCA
NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX
NCCL_IB_SUBNET_PREFIX_LEN=$NCCL_IB_SUBNET_PREFIX_LEN
NCCL_IB_SUBNET_AWARE_ROUTING=$NCCL_IB_SUBNET_AWARE_ROUTING
NCCL_IB_MERGE_NICS=$NCCL_IB_MERGE_NICS
NCCL_CROSS_NIC=$NCCL_CROSS_NIC
NCCL_ALGO=$NCCL_ALGO
NCCL_PROTO=$NCCL_PROTO
NCCL_P2P_LEVEL=$NCCL_P2P_LEVEL
NCCL_MIN_NCHANNELS=$NCCL_MIN_NCHANNELS
NCCL_MAX_NCHANNELS=$NCCL_MAX_NCHANNELS
NCCL_SKIP_TREE_CONNECT=$NCCL_SKIP_TREE_CONNECT
NCCL_SWITCHLESS_RING_ONLY=$NCCL_SWITCHLESS_RING_ONLY
NCCL_CUMEM_ENABLE=$NCCL_CUMEM_ENABLE
NCCL_IGNORE_CPU_AFFINITY=$NCCL_IGNORE_CPU_AFFINITY
NCCL_DEBUG=$NCCL_DEBUG
EOF
}

# --- remote helper ------------------------------------------------------------------------
rssh() {
  if [[ ${DRY:-0} == 1 ]]; then echo "[$1] ${*:2}"; return; fi
  if [[ $1 == local ]]; then bash -c "${*:2}"; else ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "${@:2}"; fi
}

# Stage the recipe tree to every node's overlay root (rsync; the GLM lanes' sync_overlay
# equivalent). Cheap: the vendored subset is ~350 KB (patches + launcher + tools + docs).
sync_recipe() {
  for h in "${HOSTS[@]}"; do
    if [[ $h == local ]]; then
      [[ ${DRY:-0} == 1 ]] || mkdir -p "$REMOTE_DIR"
      rsync -a --delete "$(cd "$UPSTREAM_DIR" && pwd)/" "$REMOTE_DIR/upstream/" 2>/dev/null || true
    else
      rssh "$h" "mkdir -p $REMOTE_DIR"
      if [[ ${DRY:-0} == 1 ]]; then echo "[dry-run] rsync upstream -> $h:$REMOTE_DIR/upstream"; else
        rsync -a --delete -e ssh "$(cd "$UPSTREAM_DIR" && pwd)/" "$h:$REMOTE_DIR/upstream/"
      fi
    fi
  done
}

# NCCL library sanity: the shared four-PF build must be staged on every node (same SHA).
check_nccl() {
  local h sum
  if [[ ${DRY:-0} == 1 ]]; then
    for h in "${HOSTS[@]}"; do echo "[dry-run] sha256sum $NCCL_SO_HOST_PATH on $h"; done
    return 0
  fi
  for h in "${HOSTS[@]}"; do
    sum=$(rssh "$h" "sha256sum '$NCCL_SO_HOST_PATH'") || { echo "NCCL lib unreadable on $h: $NCCL_SO_HOST_PATH" >&2; exit 2; }
    sum=${sum%% *}
    [[ $sum == "$NCCL_SHA256" ]] || { echo "NCCL SHA mismatch on $h: got $sum, want $NCCL_SHA256" >&2; exit 2; }
  done
}

case ${1:-status} in
  check)
    check_nccl
    for r in 0 1 2 3; do
      echo "== rank $r =="
      render_rank_env "$r" > "/tmp/ds41f-r$r.env"
      if [[ ${HOSTS[$r]} == local ]]; then
        bash "$UPSTREAM_DIR/$LANCHER" --check "/tmp/ds41f-r$r.env" || exit 2
      else
        scp -q -o BatchMode=yes "/tmp/ds41f-r$r.env" "${HOSTS[$r]}:/tmp/ds41f-r$r.env"
        rssh "${HOSTS[$r]}" "bash $REMOTE_DIR/upstream/$LANCHER --check /tmp/ds41f-r$r.env" || exit 2
      fi
    done
    echo "check: all four ranks validated";;

  serve)
    sync_recipe
    check_nccl
    for r in 3 2 1 0; do
      echo "== launching rank $r (${HOSTS[$r]}) =="
      render_rank_env "$r" > "/tmp/ds41f-r$r.env"
      if [[ ${HOSTS[$r]} == local ]]; then
        bash "$UPSTREAM_DIR/$LANCHER" --run "/tmp/ds41f-r$r.env"
      else
        scp -q -o BatchMode=yes "/tmp/ds41f-r$r.env" "${HOSTS[$r]}:/tmp/ds41f-r$r.env"
        rssh "${HOSTS[$r]}" "bash $REMOTE_DIR/upstream/$LANCHER --run /tmp/ds41f-r$r.env"
      fi
    done
    echo "launched deepseek-v41-flash-r0..3; watch ./start-cluster.sh logs 0 until /health is 200 (~8 min from local NVMe)";;

  stop)
    # rank 0 first (a worker joining a live old head hangs at rendezvous — upstream gotcha)
    for r in 0 1 2 3; do
      rssh "${HOSTS[$r]}" "cid=\$(docker inspect --format '{{.Id}}' deepseek-v41-flash-r$r 2>/dev/null) && docker stop --time 30 \"\$cid\" && docker rm \"\$cid\" || echo 'rank $r: not running'"
    done;;

  status)
    for r in 0 1 2 3; do
      rssh "${HOSTS[$r]}" "echo \$(hostname) \$(docker ps -a --filter name=deepseek-v41-flash-r$r --format '{{.Status}}') avail=\$(free -g | awk 'NR==2{print \$7}')G"
    done
    rssh "${HOSTS[0]}" "curl -s -m 3 -o /dev/null -w 'health %{http_code}\n' http://127.0.0.1:$API_PORT/health" || true;;

  logs)
    r=${2:-0}; rssh "${HOSTS[$r]}" "docker logs --tail ${3:-40} deepseek-v41-flash-r$r 2>&1 | cut -c1-200";;

  *) echo "usage: $0 check|serve|stop|status|logs [rank] [lines]"; exit 2;;
esac
