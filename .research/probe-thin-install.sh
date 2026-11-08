#!/bin/bash
# Diagnose why the SWA_THIN hook's receipt never appears on the engine's request path.
# Run on the head node. Prints the install path and the mark state inside the container.
set -u
C=dsv41-head

echo "=== where sitecustomize lives ==="
docker exec "$C" python3 -c 'import sitecustomize; print(sitecustomize.__file__)'

echo "=== the install branch for unified_radix_cache ==="
docker exec "$C" grep -n "unified_radix_cache" -A 5 /opt/dsv41/adapter/sitecustomize.py

echo "=== finder registration ==="
docker exec "$C" grep -n "EngramFinder()" -B 3 -A 2 /opt/dsv41/adapter/sitecustomize.py

echo "=== does the server's boot log show a thin install from the scheduler? ==="
docker logs "$C" 2>&1 | grep -E "SWA_THIN" | head -5

echo "=== receipt lines total ==="
docker logs "$C" 2>&1 | grep -c "SWA_THIN receipt"

echo "=== thin module: does install() return True here? ==="
docker exec "$C" python3 -c '
import sys; sys.path.insert(0, "/opt/dsv41")
import sitecustomize
import sglang.srt.mem_cache.unified_radix_cache as urc
import swa_thin
print("marked after import:", getattr(urc.UnifiedRadixCache.cache_unfinished_req, "_dsv41_swa_thin", None))
print("STATS:", swa_thin.STATS.active, swa_thin.STATS.installed, swa_thin.STATS.prompt_end)
'

echo "=== is sitecustomize even active in the server process? PYTHONPATH ==="
docker exec "$C" printenv PYTHONPATH
docker exec "$C" printenv PYTHONNOUSERSITE
