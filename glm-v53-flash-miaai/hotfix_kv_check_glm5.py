#!/usr/bin/env python3
"""Make the GHCR image's GLM-5.3 KV accounting self-consistent.

The GHCR image ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3 (digest
9bb1557a...) carries the DFLASH2-DRAFTER-GROUP slot-share patch, but its
KV-capacity accounting is out of sync with it: at max_model_len 1000000 the
boot dies with

    ValueError: To serve at least one request with the model's max seq len
    (1000000), (34.15 GiB KV cache is needed, ...)  [16.16 GiB available]

while the actual slot-share allocation only needs the honest cost. 34.15 GiB
= ceil(1M/64) * 2351104 — the drafter group's PADDED page charged per block.
Upstream MiaAI-Lab does not hit this because their locally built image keeps
builder, detector and accounting in sync; the GHCR image lags it (their
issue #97, "Local rebuild gets overwritten by GHCR").

Fixes applied to vllm/v1/core/kv_cache_utils.py (all idempotent, fail-closed,
atomic):

  A) _glm5_next_tensor_layout: accept drafter specs whose page_size_padded
     equals mla_page (the builder's padded slot-share geometry); still reject
     any other padding.
  B) _max_memory_usage_bytes_from_groups: prepend an early glm5 branch that
     computes needed memory the way the allocator does — one block id costs
     len(mla)*mla_page + len(idx)*idx_page (+ standalone drafter pages), and
     one max_model_len request occupies sum over groups of
     ceil(group_max_bytes / group_page) block ids (same cdiv definition as
     get_max_concurrency_for_kv_cache_config). The all-uniform "DeepseekV4"
     branch is additionally guarded so it can never capture a glm5 layout.
  D) diagnostic: the detector is renamed *_orig and wrapped; on a None
     verdict the wrapper dumps the group structure so a failing boot names
     the rejecting condition.
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

# Fix A: accept the builder's padded slot-share drafter geometry
ANCHOR_A = """        if any(s.page_size_padded is not None for s in draft_inner.values()):
            return None"""

PATCHED_A = """        if any(  # [glm53-miaai-kvcheck]
            s.page_size_padded is not None and s.page_size_padded != mla_page
            for s in draft_inner.values()
        ):
            return None"""

# Fix B: early glm5-aware accounting in _max_memory_usage_bytes_from_groups
ANCHOR_B = """    elif all(
        isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        for group in kv_cache_groups
    ):
        # Special case (only DeepseekV4 for now): all groups are
        # UniformTypeKVCacheSpecs."""

PATCHED_B = """    if (glm5n_early := _glm5_next_tensor_layout(kv_cache_groups)) is not None:
        # [glm53-miaai-kvcheck] Honest slot-share accounting, mirroring the
        # allocator (get_kv_cache_config_from_groups / _pool_bytes_per_block):
        # one block id costs len(mla)*mla_page + len(idx)*idx_page plus any
        # standalone drafter pages; one max_model_len request occupies
        # sum over groups of ceil(group_max_bytes / group_page) block ids
        # (same cdiv definition as get_max_concurrency_for_kv_cache_config).
        (
            attn_group,
            mamba_groups,
            mla_names,
            idx_names,
            mla_page,
            idx_page,
            tail_names,
            _tail_page,
            draft_group,
        ) = glm5n_early
        per_block = len(mla_names) * mla_page + len(idx_names) * idx_page
        draft_names: list[str] = []
        draft_page = 0
        draft_shared = False
        if draft_group is not None:
            draft_names = list(draft_group.layer_names)
            draft_page = next(
                iter(
                    cast(
                        UniformTypeKVCacheSpecs, draft_group.kv_cache_spec
                    ).kv_cache_specs.values()
                )
            ).page_size_bytes
            draft_shared = draft_page == mla_page
        if draft_names and not draft_shared:
            per_block += len(draft_names) * draft_page
        blocks_per_request = 0
        for group in [attn_group, *mamba_groups]:
            _gs = group.kv_cache_spec
            _mb = _gs.max_memory_usage_bytes(vllm_config)
            _pg = _gs.page_size_bytes
            blocks_per_request += (_mb + _pg - 1) // _pg
        if draft_group is not None:
            _ds = draft_group.kv_cache_spec
            _mb = _ds.max_memory_usage_bytes(vllm_config)
            _pg = _ds.page_size_bytes
            blocks_per_request += (_mb + _pg - 1) // _pg
        for group in kv_cache_groups:
            _inner = (
                cast(UniformTypeKVCacheSpecs, group.kv_cache_spec).kv_cache_specs
                if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
                else None
            )
            if not _inner or not all(
                isinstance(s, KpoolTailSpec) for s in _inner.values()
            ):
                continue
            _mb = group.kv_cache_spec.max_memory_usage_bytes(vllm_config)
            _pg = group.kv_cache_spec.page_size_bytes
            blocks_per_request += (_mb + _pg - 1) // _pg
        print(
            f"[kvcheck] early glm5 accounting: "
            f"blocks/req={blocks_per_request} per_block={per_block} "
            f"needed={blocks_per_request * per_block / 2**30:.2f} GiB",
            flush=True,
        )
        return blocks_per_request * per_block
    elif all(
        isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        for group in kv_cache_groups
    ) and _glm5_next_tensor_layout(kv_cache_groups) is None:  # [glm53-miaai-kvcheck]
        # Special case (only DeepseekV4 for now): all groups are
        # UniformTypeKVCacheSpecs."""

# Fix D: rename the detector and add a diagnostic wrapper
ANCHOR_D = "def _glm5_next_tensor_layout(\n"

WRAPPER_D = '''

def _glm5_next_tensor_layout(kv_cache_groups):  # [glm53-miaai-kvcheck]
    _r = _glm5_next_tensor_layout_orig(kv_cache_groups)
    if _r is None:
        _desc = []
        for _g in kv_cache_groups:
            _s = _g.kv_cache_spec
            _inner = getattr(_s, "kv_cache_specs", None)
            _types = (
                sorted({type(x).__name__ for x in _inner.values()})
                if _inner
                else [type(_s).__name__]
            )
            _desc.append(
                f"{type(_s).__name__}{_types}"
                f"block={getattr(_s, 'block_size', None)}"
                f"page={getattr(_s, 'page_size_bytes', None)}"
                f"padded={getattr(_s, 'page_size_padded', None)}"
            )
        print("[kvcheck] detector -> None; groups:", " | ".join(_desc),
              flush=True)
    else:
        print("[kvcheck] detector -> layout ok", flush=True)
    return _r
'''


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
    for name, anchor in (("A", ANCHOR_A), ("B", ANCHOR_B), ("D", ANCHOR_D)):
        if src.count(anchor) != 1:
            print(
                f"[kvcheck-hotfix] FAIL anchor {name} count "
                f"{src.count(anchor)} (expected 1) — refusing to write",
                flush=True,
            )
            return 1
    patched = src
    patched = patched.replace(ANCHOR_D, "def _glm5_next_tensor_layout_orig(\n", 1)
    patched = patched.replace(ANCHOR_A, PATCHED_A)
    patched = patched.replace(ANCHOR_B, PATCHED_B)
    patched = patched + WRAPPER_D
    if patched.count(MARK) != 4:
        print(f"[kvcheck-hotfix] FAIL marks={patched.count(MARK)} != 4",
              flush=True)
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
    print(f"[kvcheck-hotfix] patched {TARGET} (fixes A+B+D)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
