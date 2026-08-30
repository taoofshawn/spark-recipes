#!/usr/bin/env bash
# GB10 UMA page-cache guard (vendored from tonyd2wild/GLM-5.3-Flash-NVFP4-2x-DGX-Spark,
# cache_flusher.sh). Run this on BOTH hosts during model load: heavy buffered
# I/O fills the page cache and the NVRM allocator (which only counts MemFree,
# not MemAvailable) refuses the KV slab. Flushes whenever Cached > 40 GiB.
# Runs for 25 minutes max.
#
# Usage:  sudo ./tools/cache_flusher.sh &
end=$((SECONDS+1500))
while [ $SECONDS -lt $end ]; do
  c=$(awk '/^Cached:/{print int($2/1048576)}' /proc/meminfo)
  if [ "${c:-0}" -gt 40 ]; then
    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  fi
  sleep 5
done
