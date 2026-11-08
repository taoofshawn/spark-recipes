#!/bin/bash
# Which prefill-result path does the engine actually take, and is cache_unfinished_req on it?
set -u
C=dsv41-head

echo "=== who calls process_batch_result / process_batch_result_prefill ==="
docker exec "$C" grep -rn "process_batch_result_prefill\|process_batch_result(" \
  /sgl-workspace/sglang/python/sglang/srt/managers/ 2>/dev/null | grep -v '\.pyc' | head -12

echo
echo "=== the prefill result processor's head ==="
docker exec "$C" grep -n "def process_batch_result_prefill" -A 25 \
  /sgl-workspace/sglang/python/sglang/srt/managers/scheduler_components/batch_result_processor.py

echo
echo "=== does the batch result processor run at all? count its log markers ==="
docker logs "$C" 2>&1 | grep -c "Prefill batch"
