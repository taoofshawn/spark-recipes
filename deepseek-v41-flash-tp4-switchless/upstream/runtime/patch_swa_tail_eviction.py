#!/usr/bin/env python3
"""Deprioritize evicting a branch's trailing sliding window in SWARadixCache.

Site backport of sgl-project/sglang#35515 ("[mem_cache] SWA radix cache: evict
branch tail sliding window last", @QAQEthan, +56/-7 on the same file), adapted to
this tree's allocator API (`free_swa_segment` instead of the PR-base `free_swa`).

Problem (measured on this lane, 2026-10-25/26): `evict()` walks the SWA LRU with
no notion of branch ends, so under prefill churn it tombstones a node inside the
last `sliding_window_size` (128 here) tokens of a parked branch. Because
`_match_prefix_helper` only extends a match past a tombstone when
`sliding_window_size` consecutive non-tombstone tokens follow it, one recycled
tail node makes the ENTIRE branch unmatchable — and step 4 of `evict()`
(`_iteratively_delete_tombstone_leaf`) then deletes the tombstone chain,
freeing the branch's full-attention KV with it. Agent conversations idle in a
tool call re-prefill their whole 130K-755K-token history on the next turn.

Fix: the SWA eviction loop skips any candidate whose nearest branch end (a leaf,
or the node before a tombstone) is less than one page-aligned window away, and
retries those only once no other candidate is left. Deprioritization, not
reservation: the eviction target is still met when the pool is genuinely
exhausted, `swa_evictable_size_` accounting and the scheduler contract are
unchanged, so there is no new OOM/retract risk. Cost of the protected set is
#branch_ends x sliding_window_size (128 tokens per branch here).

Idempotent, per anchor. Exits non-zero if an anchor no longer matches, so a
base-image change cannot silently drop the fix. Optionally takes the target path
as argv[1].
"""
import pathlib
import sys

DEFAULT_TARGET = (
    "/sgl-workspace/sglang/python/sglang/srt/mem_cache/swa_radix_cache.py"
)
TAG = "# [swa-radix-tail-evict]"

# ------------------------------------------------------------------ anchor 1
# The SWA eviction loop header inside `evict`. Body below `if len(x.children) > 0:`
# is untouched (this tree frees via free_swa_segment, the PR base used free_swa).
OLD1 = """        if swa_num_evicted < swa_num_tokens:
            # get the least recently used node that is not locked, doesn't have to be a leaf
            x = self.swa_lru_list.get_lru_no_lock()

            # evict lru leaf nodes until swa_num_tokens is reached
            while swa_num_evicted < swa_num_tokens and (self.swa_lru_list.in_list(x)):
                assert not x.swa_tombstone, f"duplicate swa tombstone node, {x.id=}"
                assert x != self.root_node, f"root node is not evictable, {x.id=}"
                assert x.swa_lock_ref == 0, f"node is in use by swa kv indices, {x.id=}"

                if len(x.children) > 0:
"""
NEW1 = """        if swa_num_evicted < swa_num_tokens:
            # Evict the last sliding window of a branch only as a last resort:
            # tombstoning it makes the whole branch unmatchable, because
            # `_match_prefix_helper` needs `sliding_window_size` consecutive
            # non-tombstone tokens after a tombstone, while evicting a node
            # closer to the root only costs that prefix. This is a preference,
            # not a reservation, so the eviction target is still met.
            skip_branch_tail = True  # [swa-radix-tail-evict]
            # get the least recently used node that is not locked, doesn't have to be a leaf
            x = self.swa_lru_list.get_lru_no_lock()

            # evict lru leaf nodes until swa_num_tokens is reached
            while swa_num_evicted < swa_num_tokens:
                if not self.swa_lru_list.in_list(x):
                    if not skip_branch_tail:
                        break
                    # Every non-tail candidate is gone, retry the branch tails.
                    skip_branch_tail = False
                    x = self.swa_lru_list.get_lru_no_lock()
                    continue

                assert not x.swa_tombstone, f"duplicate swa tombstone node, {x.id=}"
                assert x != self.root_node, f"root node is not evictable, {x.id=}"
                assert x.swa_lock_ref == 0, f"node is in use by swa kv indices, {x.id=}"

                if skip_branch_tail and self._is_swa_branch_tail(x):
                    x = self.swa_lru_list.get_prev_no_lock(x)
                    continue

                if len(x.children) > 0:
"""

# ------------------------------------------------------------------ anchor 2
# `_maybe_split_leaf_for_swa_lock`: factor the tail-size formula into
# `_swa_tail_size()` so the tail check and the lock split agree on page alignment.
OLD2 = """        # Smallest page-aligned size that still covers the sliding window.
        tail_size = (
            (self.sliding_window_size + self.page_size - 1)
            // self.page_size
            * self.page_size
        )
        if len(leaf.value) <= tail_size:
"""
NEW2 = """        # Smallest page-aligned size that still covers the sliding window.
        tail_size = self._swa_tail_size()  # [swa-radix-tail-evict]
        if len(leaf.value) <= tail_size:
"""

# ------------------------------------------------------------------ anchor 3
# New helpers, inserted directly before `_split_node` (same class, same indent).
OLD3 = """    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int) -> TreeNode:
"""
NEW3 = """    def _swa_tail_size(self) -> int:  # [swa-radix-tail-evict]
        \"\"\"Smallest page-aligned size that still covers the sliding window.\"\"\"
        return (
            (self.sliding_window_size + self.page_size - 1)
            // self.page_size
            * self.page_size
        )

    def _is_swa_branch_tail(self, node: TreeNode) -> bool:
        \"\"\"Whether ``node`` holds part of the last window of some branch.

        A branch end is a leaf or the node right before a tombstone: both are
        stop points ``_match_prefix_helper`` can return. If the nearest branch
        end below ``node`` is less than one (page-aligned) window away, freeing
        ``node`` breaks the only reusable window of that branch. The walk is
        pruned by the window size, so it visits at most the tail of the branch
        rather than the whole subtree.

        Site backport of sgl-project/sglang#35515.
        \"\"\"
        cap = self._swa_tail_size()
        best = cap
        stack = [(node, 0)]
        while stack:
            cur, distance = stack.pop()
            if distance >= best:
                continue
            if len(cur.children) == 0:
                best = distance
                continue
            for child in cur.children.values():
                if child.swa_tombstone:
                    best = min(best, distance)
                else:
                    stack.append((child, distance + len(child.value)))
        return best < cap

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int) -> TreeNode:
"""


def main() -> int:
    target = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TARGET)
    text = target.read_text(encoding="utf-8")

    if TAG in text:
        print("patch_swa_tail_eviction: already applied, nothing to do")
        return 0

    for i, (old, new) in enumerate(((OLD1, NEW1), (OLD2, NEW2), (OLD3, NEW3)), 1):
        if old not in text:
            print(f"patch_swa_tail_eviction: anchor {i} NOT FOUND in {target} — "
                  "the branch tree drifted; refusing to patch blindly", file=sys.stderr)
            return 1
        if text.count(old) != 1:
            print(f"patch_swa_tail_eviction: anchor {i} is not unique in {target}",
                  file=sys.stderr)
            return 1
        text = text.replace(old, new)

    try:
        compile(text, str(target), "exec")
    except SyntaxError as exc:
        print(f"patch_swa_tail_eviction: patched file does not compile: {exc}",
              file=sys.stderr)
        return 1

    target.write_text(text, encoding="utf-8")
    print("patch_swa_tail_eviction: applied 3/3 anchors "
          "(evict tail deprioritization, _swa_tail_size, _is_swa_branch_tail)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
