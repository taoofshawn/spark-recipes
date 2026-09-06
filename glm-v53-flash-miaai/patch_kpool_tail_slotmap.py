#!/usr/bin/env python3
"""Fix the K-pool tail out-of-bounds slot mapping on hybrid models (GLM-5.3).

Root cause, measured on the DGX Spark with a probe at the tail metadata
builder: ``common_attn_metadata.positions`` is None on every call, so the
one-block correction that already exists in ``compute_kpool_tail_slot_mapping``
(indexer.py) is SKIPPED and the tail group falls through to the generic paged
mapping. That mapping indexes a one-entry block-table row by
``pos // block_size``, produces garbage block ids, and the two kpool kernels
(``_kpool_tail_seed_kernel`` / ``_kpool_decode_update_batched_kernel``) write
through them — intermittent OOB writes that can corrupt a neighbouring layer's
sparse-attention index, decided by pool geometry, not by whether a run
completes.

Why positions are None: GLM-5.3 is a hybrid (KDA + sparse MLA) model, so its
attention metadata is built by ``vllm/v1/worker/gpu/model_states/
mamba_hybrid.py``. That path calls ``build_attn_metadata(...)`` WITHOUT
``positions=`` (parameter defaults to None), while the plain transformer path
in ``model_states/default.py`` passes ``positions=input_batch.positions``.

Second half: with positions present, ``compute_kpool_tail_slot_mapping`` runs
every step and returned ``slot_mapping.clone()``, a fresh allocation. CUDA
graph capture records that transient address; on replay the tail kernels read
a buffer that has since been freed or reused (the illegal memory access seen
with graphs on and not under --enforce-eager). The caller's slot_mapping is
the tail group's own persistent buffer, so writing in place is the correct
semantics as well as the safe one.

Source of the fix: vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe
``scripts/patch_kpool_tail_positions.py`` (docs/KPOOL_TAIL_BUG.md), vendored
for this recipe on 2026-09-04. Validated in that repo: 48 overruns -> 0,
19,575 decode-path tail updates with 0 OOB, 57,551-update soak with 0 OOB.
This patches ``mamba_hybrid.py`` and ``indexer.py`` IN PLACE at boot (the
GHCR :exl3 image's dist-packages; no rebuild). Fail-closed: exits non-zero if
an anchor is missing or not unique — the compose patch loop aborts the boot.

Applies to vllm/v1/worker/gpu/model_states/mamba_hybrid.py and
vllm/v1/attention/backends/mla/indexer.py, neither of which the rest of this
recipe's patch stack touches.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ANCHOR = """            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            model_specific_attn_metadata=mamba_attn_metadata,
            for_cudagraph_capture=for_capture,
            rswa_prefix_lens=input_batch.prompt_lens,
        )
"""

PATCHED = """            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            # Hybrid models never passed positions here, unlike default.py.
            # The K-pool tail builder needs them: without positions it skips
            # compute_kpool_tail_slot_mapping and uses the generic paged
            # mapping against a one-entry block-table row, which writes the
            # tail cache out of bounds. See docs/KPOOL_TAIL_BUG.md.
            positions=input_batch.positions,
            model_specific_attn_metadata=mamba_attn_metadata,
            for_cudagraph_capture=for_capture,
            rswa_prefix_lens=input_batch.prompt_lens,
        )
"""

REL = "vllm/v1/worker/gpu/model_states/mamba_hybrid.py"

# Second edit. Once positions are present, compute_kpool_tail_slot_mapping runs
# every step and returned a fresh clone. CUDA graph capture records that
# transient address; replay then reads a buffer that has since been freed or
# reused, which is the illegal memory access seen with graphs on and not under
# --enforce-eager. The caller's slot_mapping is the tail group's own persistent
# buffer, so writing in place is the correct semantics as well as the safe one.
REL2 = "vllm/v1/attention/backends/mla/indexer.py"
ANCHOR2 = """    out = slot_mapping.clone()
    if num_actual_tokens == 0:
        return out
"""
PATCHED2 = """    # In place: slot_mapping is the tail group's persistent buffer. A fresh
    # clone here is captured by CUDA graphs at a transient address and read
    # back stale on replay (illegal memory access). See docs/KPOOL_TAIL_BUG.md.
    out = slot_mapping
    if num_actual_tokens == 0:
        return out
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source",
        default="/usr/local/lib/python3.12/dist-packages",
        help="vLLM source root (default: the :exl3 image's dist-packages)",
    )
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()
    path = Path(args.source) / REL
    text = path.read_text(encoding="utf-8")

    path2 = Path(args.source) / REL2
    text2 = path2.read_text(encoding="utf-8")

    if args.revert:
        if PATCHED in text:
            path.write_text(text.replace(PATCHED, ANCHOR, 1), encoding="utf-8")
            print(f"reverted: {path}")
        if PATCHED2 in text2:
            path2.write_text(text2.replace(PATCHED2, ANCHOR2, 1), encoding="utf-8")
            print(f"reverted: {path2}")
        return

    if "Hybrid models never passed positions here" in text:
        print(f"already patched: {path}")
    else:
        if text.count(ANCHOR) != 1:
            raise SystemExit(
                f"{path}: expected exactly one anchor, found {text.count(ANCHOR)}. "
                "The hybrid model-state call has changed; re-derive the patch."
            )
        path.write_text(text.replace(ANCHOR, PATCHED, 1), encoding="utf-8")
        print(f"patched: {path}")
        print("mamba_hybrid.py now passes positions=input_batch.positions")

    if "In place: slot_mapping is the tail group" in text2:
        print(f"already patched: {path2}")
        return
    if text2.count(ANCHOR2) != 1:
        raise SystemExit(f"{path2}: expected exactly one anchor, found {text2.count(ANCHOR2)}")
    path2.write_text(text2.replace(ANCHOR2, PATCHED2, 1), encoding="utf-8")
    print(f"patched: {path2}")
    print("compute_kpool_tail_slot_mapping now writes the persistent buffer in place")


if __name__ == "__main__":
    main()
