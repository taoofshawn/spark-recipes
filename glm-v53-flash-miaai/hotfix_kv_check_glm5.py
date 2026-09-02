#!/usr/bin/env python3
"""Make the GHCR image's GLM-5.3 KV accounting self-consistent.

Two baked components of ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3
(digest 9bb1557a...) disagree, and the disagreement surfaces as a false
positive KV-capacity failure at max_model_len 1000000:

1. The DFLASH2-DRAFTER-GROUP group builder (_get_kv_cache_groups_glm5_next)
   emits the drafter's 5 SWA layers as PADDED slot-share
   (block_size=64, page_size_padded=mla_page) — its own comment:
   "Manager 64 matches the SWA kernel, so padding the page to mla_page is a
   safe strided view".
2. _glm5_next_tensor_layout() REJECTS any drafter spec with
   page_size_padded is not None ("NEVER page_size_padded") — a stale
   precondition from the pre-padded era (upstream boot-8 OOB note).

With the layout detector returning None, every glm5-aware consumer
(_pool_bytes_per_block, _max_memory_usage_bytes_from_groups,
get_kv_cache_config_from_groups) falls back to generic paths, and the
all-uniform "DeepseekV4" branch of _max_memory_usage_bytes_from_groups
charges the drafter group's padded page for every block of a full 1M request:

    ceil(1_000_000 / 64) * 2_351_104 B = 34.15 GiB  (vs 16.16 GiB available)

-> "ValueError: To serve at least one request with the model's max seq len
(1000000), (34.15 GiB KV cache is needed, ...)" — while the actual allocation
only needs the honest slot-share cost. Upstream MiaAI-Lab does not hit this
because their locally built image keeps builder and detector in sync; the
GHCR image lags it (their issue #97, "Local rebuild gets overwritten by
GHCR").

Fixes, both in vllm/v1/core/kv_cache_utils.py:

  A) _glm5_next_tensor_layout: accept drafter specs whose page_size_padded
     equals mla_page (the builder's padded slot-share geometry); still reject
     any other padding.
  B) _max_memory_usage_bytes_from_groups: exclude glm5 layouts from the
     all-uniform "DeepseekV4" branch so slot-share layouts reach the
     glm5-aware branch directly below, matching the allocator's
     _pool_bytes_per_block accounting.

Fail-closed: preflights all anchors before writing, atomic replace,
idempotent via marker.
"""
import os
import sys
import tempfile
from pathlib import Path

TARGET = Path(
    os.environ.get(
        "GLM53_KV_CHECK_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py",
    )
)
MARK = "[glm53-miaai-kvcheck]"

GUARD = "def _max_memory_usage_bytes_from_groups("

# Fix B: exclude glm5 layouts from the all-uniform branch
ANCHOR_B = """    elif all(
        isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        for group in kv_cache_groups
    ):
        # Special case (only DeepseekV4 for now): all groups are
        # UniformTypeKVCacheSpecs."""

PATCHED_B = """    elif all(
        isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        for group in kv_cache_groups
    ) and _glm5_next_tensor_layout(kv_cache_groups) is None:  # [glm53-miaai-kvcheck]
        # Special case (only DeepseekV4 for now): all groups are
        # UniformTypeKVCacheSpecs."""

# Fix A: accept the builder's padded slot-share drafter geometry
ANCHOR_A = """        if any(s.page_size_padded is not None for s in draft_inner.values()):
            return None"""

PATCHED_A = """        if any(  # [glm53-miaai-kvcheck]
            s.page_size_padded is not None and s.page_size_padded != mla_page
            for s in draft_inner.values()
        ):
            return None"""


def main() -> int:
    if not TARGET.is_file():
        print(f"[kvcheck-hotfix] FAIL {TARGET} missing", flush=True)
        return 1
    src = TARGET.read_text()
    if MARK in src:
        print("[kvcheck-hotfix] already patched", flush=True)
        return 0
    if GUARD not in src:
        print("[kvcheck-hotfix] FAIL guard function not found", flush=True)
        return 1
    for name, anchor in (("A", ANCHOR_A), ("B", ANCHOR_B)):
        if src.count(anchor) != 1:
            print(
                f"[kvcheck-hotfix] FAIL anchor {name} count "
                f"{src.count(anchor)} (expected 1) — refusing to write",
                flush=True,
            )
            return 1
    patched = src.replace(ANCHOR_A, PATCHED_A).replace(ANCHOR_B, PATCHED_B)
    if patched.count(MARK) != 2:
        print("[kvcheck-hotfix] FAIL replacement produced no marks", flush=True)
        return 1
    compile(patched, str(TARGET), "exec")
    fd, tmp = tempfile.mkstemp(dir=str(TARGET.parent), suffix=".py")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(patched)
        os.replace(tmp, TARGET)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    print(f"[kvcheck-hotfix] patched {TARGET} (fixes A+B)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
