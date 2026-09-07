#!/usr/bin/env python3
"""Make the GHCR image's GLM-5.3 KV accounting self-consistent.

The GHCR image ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3 (digest
9bb1557a...) carries partially-out-of-sync revisions of the DFLASH2
DRAFTER-GROUP patch set. Three consequences, all hit on 2026-09-01/02:

1. (boot check) `_max_memory_usage_bytes_from_groups` has no glm5 branch and
   the layout detector rejects the builder's padded drafter geometry ->
   `ValueError: ... 34.15 GiB KV cache is needed ... 16.16 GiB available`
   at max_model_len 1000000.
2. (runtime) the baked builder's else branch is an older "STANDALONE"
   revision: the draft group keeps block_size=16 with the drafter's
   16384-token window -> 1025 draft blocks per request -> a full 1M request
   needs ~1306 block ids vs a ~621-block pool -> the scheduler refuses every
   large request (num_requests_waiting{reason="capacity"} stuck forever,
   GPU idle, small requests OK at ~18 tok/s).
3. (cosmetic) two defunct helper processes on each node.

Upstream MiaAI-Lab does not hit this because their validated boots run
LOCALLY BUILT images where builder, detector and accounting are in sync;
the GHCR image lags (their issue #97, "Local rebuild gets overwritten by
GHCR"). With their own start.sh + this same image you would hit the same
failures; it is not a compose-conversion artifact.

Fixes applied to vllm/v1/core/kv_cache_utils.py (idempotent via marker,
fail-closed: every anchor preflighted, atomic replace, compile check):

  A) _glm5_next_tensor_layout: accept drafter specs whose
     page_size_padded == mla_page (the current overlay's padded slot-share
     geometry); still reject any other padding.
  B) _max_memory_usage_bytes_from_groups: prepend an early glm5 branch that
     computes needed memory allocator-consistently: per_block =
     len(mla)*mla_page + len(idx)*idx_page (+ standalone draft page),
     blocks/req = window-bounded cdiv over MLA + draft + tail groups,
     mamba groups EXCLUDED (length-independent SSM state, slot-shared with
     MLA block ids). Guard the all-uniform "DeepseekV4" branch against glm5
     layouts.
  E) _get_kv_cache_groups_glm5_next: replace the stale STANDALONE else
     branch with the current overlay's PADDED SLOT-SHARE rescale
     (block_size=64, page_size_padded=mla_page), bounding draft demand to
     ceil(window/64) blocks.
  D) diagnostic: detector renamed *_orig and wrapped; on a None verdict the
     wrapper dumps the group structure.

Verification (2026-09-02): blocks/req 409 -> expected ~537 with fix E
(280 attn + 256 draft-window + 1 tail); pool ~621 blocks; engine healthy,
serving glm-5.3-flash at 1M context with vision.
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

# ---- Fix A: accept the builder's padded slot-share drafter geometry ----
ANCHOR_A = """        if any(s.page_size_padded is not None for s in draft_inner.values()):
            return None"""

PATCHED_A = """        if any(  # [glm53-miaai-kvcheck]
            s.page_size_padded is not None and s.page_size_padded != mla_page
            for s in draft_inner.values()
        ):
            return None"""

# ---- Fix B: early glm5-aware accounting in the memory check ----
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
        # standalone draft pages; one max_model_len request occupies
        # sum over groups of ceil(group_max_bytes / group_page) block ids
        # (same cdiv definition as get_max_concurrency_for_kv_cache_config),
        # with SWA groups re-bounded by their inner sliding window (the
        # UniformType wrapper drops it) and mamba groups excluded
        # (length-independent, slot-shared with the MLA block ids).
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
                _parts.append(f"mamba(~{_cdiv(group)[0]} excl)")
                continue
            _c, _blk, _pg, _t = _cdiv(group)
            blocks_per_request += _c
            _parts.append(f"{_t}(block={_blk},page={_pg})={_c}")
        print(
            f"[kvcheck] early glm5 accounting: "
            f"blocks/req={blocks_per_request} per_block={per_block} "
            f"needed={blocks_per_request * per_block / 2**30:.2f} GiB "
            f"| " + "; ".join(_parts),
            flush=True,
        )
        return blocks_per_request * per_block
    elif all(
        isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        for group in kv_cache_groups
    ) and _glm5_next_tensor_layout(kv_cache_groups) is None:  # [glm53-miaai-kvcheck]
        # Special case (only DeepseekV4 for now): all groups are
        # UniformTypeKVCacheSpecs."""

# ---- Fix E: rescale the draft group like the CURRENT overlay's builder ----
ANCHOR_E = """        else:
            # STANDALONE: the drafter's geometry cannot exactly fill the MLA
            # page; keep its spec as-is and give its layers compact tensors
            # of their own (emitted in get_kv_cache_config_from_groups and
            # charged in the per-block cost).
            new_draft_specs = dict(draft_specs)"""

PATCHED_E = """        else:
            # [glm53-miaai-kvcheck] PADDED SLOT-SHARE (current overlay):
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

# ---- Fix D: rename the detector and add a diagnostic wrapper ----
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
