#!/bin/bash
# Which sglang copy does the container resolve, and are there duplicate cache modules?
set -u
C=dsv41-head

echo "=== PYTHONPATH ==="
docker exec "$C" printenv PYTHONPATH

echo "=== sglang package location ==="
docker exec "$C" python3 -c 'import sglang; print(sglang.__file__)'

echo "=== the cache module the container resolves ==="
docker exec "$C" python3 -c 'import sglang.srt.mem_cache.unified_radix_cache as m; print(m.__file__)'

echo "=== every unified_radix_cache.py on disk ==="
docker exec "$C" find / -name 'unified_radix_cache.py' -not -path '*/node_modules/*' 2>/dev/null

echo "=== receipt count and the cache class the engine built ==="
docker logs "$C" 2>&1 | grep -E 'Tree cache initialized|SWA_THIN receipt' | tail -4
