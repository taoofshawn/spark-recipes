#!/usr/bin/env python3
"""Route GLM-5-Next slot-share layouts to the glm5-aware KV memory check.

GHCR image ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3
(digest 9bb1557a...) carries the DFLASH2-DRAFTER-GROUP slot-share patch
(padded slot-share block=64, page_size_padded=mla_page), but its
vllm/v1/core/kv_cache_utils.py::_max_memory_usage_bytes_from_groups checks the
all-uniform "DeepseekV4" branch BEFORE the _glm5_next_tensor_layout branch.
GLM-5.3-Flash's 7 groups (MLA + indexer tail + 4 mamba + DFlash2 drafter) are
all UniformTypeKVCacheSpecs after grouping, so the DeepseekV4 branch captures
them and charges the drafter group's padded page (2351104 B) for every block
of a full 1M request:

    ceil(1_000_000 / 64) * 2_351_104 B = 34.15 GiB  (vs 16.16 GiB available)

That is a false positive: the drafter layers CO-OWN the MLA tensors via
slot-share, adding no per-block bytes (_pool_bytes_per_block already excludes
them, which is why allocation succeeds once the check is bypassed). Upstream
MiaAI-Lab does not hit this because their locally built image's accounting
agrees with the overlay; the GHCR image lags it (their issue #97, "Local
rebuild gets overwritten by GHCR").

Fix: add `and _glm5_next_tensor_layout(kv_cache_groups) is None` to the
all-uniform branch condition so slot-share layouts fall through to the
glm5-aware branch directly below it, which charges each block id the honest
per-block byte sum. Fail-closed: preflights both anchors before writing,
atomic replace, idempotent via marker.
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

ANCHOR = """    elif all(
        isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        for group in kv_cache_groups
    ):
        # Special case (only DeepseekV4 for now): all groups are
        # UniformTypeKVCacheSpecs."""

PATCHED = """    elif all(
        isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        for group in kv_cache_groups
    ) and _glm5_next_tensor_layout(kv_cache_groups) is None:  # [glm53-miaai-kvcheck]
        # Special case (only DeepseekV4 for now): all groups are
        # UniformTypeKVCacheSpecs."""

GUARD = "def _max_memory_usage_bytes_from_groups("


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
    if src.count(ANCHOR) != 1:
        print(
            f"[kvcheck-hotfix] FAIL anchor count "
            f"{src.count(ANCHOR)} (expected 1) — refusing to write",
            flush=True,
        )
        return 1
    patched = src.replace(ANCHOR, PATCHED)
    if patched == src:
        print("[kvcheck-hotfix] FAIL no-op replace", flush=True)
        return 1
    # compile check before touching the file
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
    print(f"[kvcheck-hotfix] patched {TARGET}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
