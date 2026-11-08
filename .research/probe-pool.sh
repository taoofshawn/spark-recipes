#!/bin/bash
# Compare DSV4 pool sizes across every boot in the recorded serve logs.
set -u
for f in /tmp/dsv41-serve*.log; do
  [ -f "$f" ] || continue
  line=$(grep -o 'DSV4 pool sizes:[^|]*' "$f" 2>/dev/null | head -1)
  [ -n "$line" ] && echo "$(basename "$f"): $line"
done
echo "=== from the running container ==="
docker logs dsv41-head 2>&1 | grep -o 'DSV4 pool sizes:[^|]*' | head -1
echo "=== gpu memory now ==="
docker exec dsv41-head python3 -c 'import torch; print([round(torch.cuda.mem_get_info(i)[0]/2**30,2) for i in range(torch.cuda.device_count())])' 2>/dev/null || echo "torch query unavailable"
