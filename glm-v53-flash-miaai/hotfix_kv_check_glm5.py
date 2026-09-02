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
        def _cdiv(group):
            # Window-bounded demand: UniformTypeKVCacheSpecs loses the inner
            # SlidingWindowSpec window (its max_memory_usage_pages scales with
            # max_model_len), so re-derive pages from the inner specs' window.
            # A windowed group only ever holds ceil(window / block) live
            # blocks; unwindowed groups keep the max_model_len scaling.
            _s = group.kv_cache_spec
            _mb = _s.max_memory_usage_bytes(vllm_config)
            _pg = _s.page_size_bytes
            _blk = getattr(_s, "block_size", None)
            _c = (_mb + _pg - 1) // _pg
            _inner = getattr(_s, "kv_cache_specs", None)
            if _inner and _blk:
                _wins = [
                    getattr(x, "sliding_window", None) for x in _inner.values()
                ]
                _wins = [w for w in _wins if w]
                _maxlen = vllm_config.model_config.max_model_len
                if _wins:
                    _w = min(_wins)
                    if _w < _maxlen:
                        _c = min(_c, (_w + _blk - 1) // _blk)
            return _c, _blk, _pg, type(_s).__name__

        blocks_per_request = 0
        _parts = []
        for group in kv_cache_groups:
            _inner = (
                cast(UniformTypeKVCacheSpecs, group.kv_cache_spec).kv_cache_specs
                if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
                else None
            )
            _is_tail = bool(_inner) and all(
                isinstance(s, KpoolTailSpec) for s in _inner.values()
            )
            if group is not attn_group and not _is_tail and group is not draft_group:
                # mamba groups: length-independent SSM state, slot-shared
                # with the MLA block ids (allocator shared_by) — excluded.
                _parts.append(f"mamba(~{_cdiv(group)[0]} excl)")
                continue
            _c, _blk, _pg, _t = _cdiv(group)
            blocks_per_request += _c
            _parts.append(f"{_t}(block={_blk},page={_pg})={_c}")
        print(
            f"[kvcheck] early glm5 accounting: "
            f"blocks/req={blocks_per_request} per_block={per_block} "
            f"needed={blocks_per_request * per_block / 2**30:.2f} GiB "
            f"| mla_names={len(mla_names)} mla_page={mla_page} "
            f"idx_names={len(idx_names)} idx_page={idx_page} | "
            + "; ".join(_parts),
            flush=True,
        )
        return blocks_per_request * per_block
    elif all(
        isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        for group in kv_cache_groups
    ) and _glm5_next_tensor_layout(kv_cache_groups) is None:  # [glm53-miaai-kvcheck]
        # Special case (only DeepseekV4 for now): all groups are
        # UniformTypeKVCacheSpecs."""


# Fix E: rescale the draft group like the CURRENT overlay's builder.
# The baked builder's else branch is an older revision ("STANDALONE",
# specs kept at block 16): with the drafter's window (16384) that makes
# a 1M request need 1025 draft blocks -> 1306 total > pool => the
# scheduler refuses ALL large requests (waiting{capacity} forever,
# GPU idle). The current overlay's padded slot-share (block 64,
# page_size_padded=mla_page) bounds draft demand to ceil(16384/64)=256.
ANCHOR_E = """        else:
        # STANDALONE: the drafter's geometry cannot exactly fill the MLA
        # page; keep its spec as-is and give its layers compact tensors
        # of their own (emitted in get_kv_cache_config_from_groups and
        # charged in the per-block cost).
        new_draft_specs = dict(draft_specs)"""

PATCHED_E = """        else:
        # [glm53-miaai-drafterfix] PADDED SLOT-SHARE (current overlay):
        # the drafter's bytes/token cannot exact-fill the MLA page.
        # Manager 64 matches the SWA kernel, so padding the page to
        # mla_page is a safe strided view (the boot-8 OOB was kernel 64
        # inside a 2304-token manager). Layer i co-owns MLA tensor i.
        # Without the rescale the standalone drafter keeps block 16 and
        # its window-bounded demand (1025 blocks at window 16384)
        # makes a 1M request unschedulable on a ~621-block pool.
        compact_block = 64
        logger.info(
            "DFlash2 drafter KV: padded slot-share block=%d "
            "mla_page=%d (was block=%s); exact-fit page mismatch "
            "draft_bytes/token=%d",
            compact_block,
            mla_page,
            any_draft.block_size,
            draft_bytes_per_token,
        )
        new_draft_specs = {
            name: replace(
                s,
                block_size=compact_block,
                page_size_padded=mla_page,
            )
            for name, s in draft_specs.items()
        }"""

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
    for name, anchor in (("A", ANCHOR_A), ("B", ANCHOR_B), ("E", ANCHOR_E),
                         ("D", ANCHOR_D)):
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
    patched = patched.replace(ANCHOR_E, PATCHED_E)
    patched = patched + WRAPPER_D
    if patched.count(MARK) != 5:
        print(f"[kvcheck-hotfix] FAIL marks={patched.count(MARK)} != 5",
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
    print(f"[kvcheck-hotfix] patched {TARGET} (fixes A+B+E+D)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
