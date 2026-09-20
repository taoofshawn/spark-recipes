#!/usr/bin/env python3
"""Fail-closed installer for the glm-v53-flash-intel-w4a16-v029 patch stack.

Applies the bundled unified diffs to the vLLM tree of the digest-pinned
official base image (vllm/vllm-openai:v0.29.0-aarch64), gating every step on
exact before and after SHA-256. All outputs are derived and validated in memory
(hashes + syntax) BEFORE the first write; installation stages temp siblings and
atomically replaces each target. Any mismatch aborts without writing. Pure
Python: no `patch` or `git` binary is required inside the image.

Patch stack (order matters; see docs/PATCH-REBASE-NOTES.md for provenance):
  0001  PR #53388 backport  — disable_eagle_block_drop (mtp3 lane)
  0003  scheduler LCM/mamba block align (PMU128; one hunk)
  0011  PR #53969 backport — SM120 sparse-MLA NoPE zero-pad + effective topk
  0012  SM121 PDL gate       — is_arch_support_pdl major in (9,10)

Before/after hashes were captured empirically by applying the stack in a
scratch container of the exact pinned base image (2026-09-20); the image tree
is byte-identical to the clean v0.29.0 tag tree.

Usage:
  apply_runtime_patches.py [--root DIR]              apply all steps
  apply_runtime_patches.py --root DIR --verify-only  assert final state only

ROOT is the directory containing `vllm/` (inside the image:
/usr/local/lib/python3.12/dist-packages).
"""

import argparse
import ast
import hashlib
import os
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PATCH_DIR = SCRIPT_DIR / "patches"
DEFAULT_ROOT = "/usr/local/lib/python3.12/dist-packages"

BASE_IMAGE = "vllm/vllm-openai@sha256:18372a7224938643461b846fb64c5c9d3d6e9727e82caf2dc3043e620c9d4d7a"

# Ordered patch steps: patch file -> {repo-relative path: (before, after) SHA-256}.
PATCH_STEPS = [
    ("0001-vllm-53388-native-mtp-block-drop.patch", {
        "vllm/config/speculative.py": (
            "cf8fcd342b6e863bac18415705a0ec2864c12b9def8e23ee84ea3ab4ede19847",
            "4e3d0a9b93f54fc0f25897e8c7bba5412e3851dd819950964c201ed6376e08b7"),
        "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py": (
            "303be42819e8308ff3e3e1b660748d6ba45eac1b2be9c77eb4269a13c0f6ea6e",
            "e03fcf29da23d0d33c741c6234ceeaf882dd36108aeb27c494124fd00d9d89af"),
        "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py": (
            "0a66105e9cb9f537981247aac1e59164dba4410d03a93cf28eae3347f7abb35d",
            "cd43cf5f0c22e4e5616baf9c27a785bac16559d806f21f9e1b7bec056cb97a2d"),
        "vllm/v1/core/kv_cache_utils.py": (
            "5294bef00c2f127f672a37eda9acbe440418f01622ef49b9be57761c54ebd7f6",
            "f587863bc066cc1e115e0da41b11ebb0e226f806f82625284465e8bc545bcc1e"),
        "vllm/v1/core/sched/scheduler.py": (
            "abca7134821e2fb5cc8572df5c4a0b570ecf702646254327bc786fc452074983",
            "797a9d759fa6ef6068e2c795cdd891cc9233bc0cd64ba5788ab7d872e10d78e9"),
        "vllm/v1/core/single_type_kv_cache_manager.py": (
            "3038a12b187453a5e1a3b2fd4f5cb4f5be08b23108f8352263032d9179472d55",
            "14f5e759c6e2b5002d468485e1bc07439f64ccf49efef0468e53fd8c7c0a8d79"),
        "vllm/v1/simple_kv_offload/manager.py": (
            "e7c04346160dc4cdd885877b6229e9fee37addd7ae4dfbeed7416e594987ba73",
            "6877ac8f6fc0e9240ec28b40ab27a30a7cf4c632e1fa90f714fddcd8767f7bec"),
    }),
    ("0003-vllm-scheduler-lcm-mamba-block-align.patch", {
        "vllm/v1/core/sched/scheduler.py": (
            "797a9d759fa6ef6068e2c795cdd891cc9233bc0cd64ba5788ab7d872e10d78e9",
            "a14442fbed395d3137fad438c0d7b0d303f25435a990fd4eb2914d6d7048e2b5"),
    }),
    ("0011-sm121-sm120-nope-topk.patch", {
        "vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py": (
            "f206c30210acd7424f6cb356ad14c6d21a9cfb9216058d424da67177beb8f690",
            "3f42aabc0e4ea8742b4adebf008c9b72b0cc4e3dd7925cf07b8fd709a87f591c"),
        "vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py": (
            "a0023f72125cb0d5599b5bf940c86be1f0c9985bd62b0919f243a8fda76f4449",
            "cd1c9fb4675a2a1824d502852a0f3ff46b73f828a474d0feada7b76558b8863b"),
    }),
    ("0012-sm121-pdl-gate.patch", {
        "vllm/platforms/cuda.py": (
            "b40cf00705013017b0c5f32cd66f69156599932db8126eb9f980ebc4ae20cf01",
            "3d0914a534a7e037d9da78f4d460825117b8029e99a5f837fe42241f01e9bd31"),
    }),
]

# No complete-file overlays in this stack: the legacy kpool indexer overlay is
# obsolete against v0.29.0 (the file no longer exists; PR #53906 integrated the
# GLM indexer into the unified sparse_attn_indexer.py — see
# docs/PATCH-REBASE-NOTES.md).
OVERLAY = {}

HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")


def die(msg: str) -> "None":
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return fh.read()


def parse_unified_diff(text: str) -> dict:
    """Parse a unified diff into {path: [(old_lines, new_lines), ...]}.

    Strict: multi-file diffs supported, every hunk must be complete and
    well-formed or the installer aborts.
    """
    files: dict = {}
    state = {"path": None, "hunks": None, "old": None, "new": None,
             "rem_old": 0, "rem_new": 0, "in_hunk": False}

    def flush_file() -> None:
        if state["path"] is None:
            return
        if state["in_hunk"]:
            die(f"{state['path']}: truncated hunk in diff")
        if not state["hunks"]:
            die(f"{state['path']}: no hunks in diff section")
        files[state["path"]] = state["hunks"]

    def flush_hunk() -> None:
        if state["rem_old"] != 0 or state["rem_new"] != 0:
            die(f"{state['path']}: hunk ended early "
                f"(old missing {state['rem_old']}, new missing {state['rem_new']})")
        state["hunks"].append((state["old"], state["new"]))
        state["in_hunk"] = False
        state["old"] = state["new"] = None

    for line in text.split("\n"):
        if state["in_hunk"]:
            if line.startswith("\\"):
                die("\\ No newline at end of file is not supported by this installer")
            if line.startswith("+"):
                state["new"].append(line[1:])
                state["rem_new"] -= 1
            elif line.startswith("-"):
                state["old"].append(line[1:])
                state["rem_old"] -= 1
            elif line.startswith(" "):
                state["old"].append(line[1:])
                state["new"].append(line[1:])
                state["rem_old"] -= 1
                state["rem_new"] -= 1
            elif line == "" and (state["rem_old"] > 0 or state["rem_new"] > 0):
                state["old"].append("")
                state["new"].append("")
                state["rem_old"] -= 1
                state["rem_new"] -= 1
            else:
                flush_hunk()
            if state["in_hunk"] and state["rem_old"] == 0 and state["rem_new"] == 0:
                flush_hunk()
            continue
        if line.startswith("@@") and HUNK_HEADER.match(line):
            match = re.match(r"^@@ -\d+(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            state["rem_old"] = int(match.group(1)) if match.group(1) else 1
            state["rem_new"] = int(match.group(3)) if match.group(3) else 1
            state["old"], state["new"] = [], []
            state["in_hunk"] = True
        elif line.startswith("--- a/"):
            flush_file()
            state["path"] = line[6:]
            state["hunks"] = []
        elif line.startswith("+++ b/"):
            if state["path"] is None or line[6:] != state["path"]:
                die(f"unsupported diff header pair for {state['path']!r} / {line[6:]!r}")
        elif line.startswith("diff --git") or line.startswith("index ") or line == "":
            continue
        else:
            die(f"unexpected line outside hunk: {line!r}")
    if state["in_hunk"]:
        flush_hunk()
    flush_file()
    if not files:
        die("no usable hunks found in patch")
    return files


def apply_hunks(source: str, hunks: list, rel: str) -> str:
    lines = source.split("\n")
    for old_block, new_block in hunks:
        span = len(old_block)
        hits = [i for i in range(len(lines) - span + 1)
                if lines[i:i + span] == old_block]
        if len(hits) != 1:
            die(f"{rel}: hunk old-block match count {len(hits)} (need exactly 1), "
                f"first line {old_block[0]!r}")
        lines[hits[0]:hits[0] + span] = new_block
    return "\n".join(lines)


def plan_all(root: Path) -> list:
    """Preflight: derive every output in memory and validate before/after
    SHA-256 plus syntax, chaining in-memory state across steps (the scheduler
    is patched twice). Performs no writes."""
    state: dict = {}        # rel -> current in-memory bytes
    first_before: dict = {}
    last_after: dict = {}
    for patch_name, targets in PATCH_STEPS:
        parsed = parse_unified_diff(read_text(PATCH_DIR / patch_name))
        if set(parsed) != set(targets):
            die(f"{patch_name}: patch touches {sorted(parsed)}, expected {sorted(targets)}")
        for rel, (before, after) in targets.items():
            if rel not in state:
                target = root / rel
                if not target.is_file():
                    die(f"{rel}: missing from tree")
                state[rel] = target.read_bytes()
                first_before[rel] = before
            if hashlib.sha256(state[rel]).hexdigest() != before:
                die(f"{rel}: current SHA {hashlib.sha256(state[rel]).hexdigest()} "
                    f"!= expected before {before}")
            patched = apply_hunks(state[rel].decode("utf-8"), parsed[rel], rel).encode("utf-8")
            if hashlib.sha256(patched).hexdigest() != after:
                die(f"{rel}: computed post-patch SHA != expected after {after}")
            if rel.endswith(".py"):
                try:
                    ast.parse(patched.decode("utf-8"))
                except SyntaxError as exc:
                    die(f"{rel}: patched content fails syntax check: {exc}")
            state[rel] = patched
            last_after[rel] = after
    for rel, (before, after, asset) in OVERLAY.items():
        if rel not in state:
            target = root / rel
            if not target.is_file():
                die(f"{rel}: missing from tree")
            state[rel] = target.read_bytes()
            first_before[rel] = before
        if hashlib.sha256(state[rel]).hexdigest() != before:
            die(f"{rel}: current SHA {hashlib.sha256(state[rel]).hexdigest()} "
                f"!= expected before {before}")
        data = (PATCH_DIR / asset).read_bytes()
        if hashlib.sha256(data).hexdigest() != after:
            die(f"bundled asset {asset}: SHA mismatch against expected after {after}")
        try:
            ast.parse(data.decode("utf-8"))
        except SyntaxError as exc:
            die(f"bundled asset {asset}: syntax check failed: {exc}")
        state[rel] = data
        last_after[rel] = after
    return [(root / rel, state[rel], rel, first_before[rel], last_after[rel])
            for rel in sorted(last_after)]


def install(plans: list) -> None:
    staged = []
    try:
        for target, data, rel, before, after in plans:
            tmp = target.parent / f".{target.name}.tmp-{os.getpid()}"
            with open(tmp, "wb") as fh:
                fh.write(data)
            staged.append(tmp)
            os.replace(tmp, target)
            print(f"  {rel}: {before[:12]}.. -> {after[:12]}..")
    finally:
        for tmp in staged:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass


def verify(root: Path) -> None:
    expected = {}
    for _, targets in PATCH_STEPS:
        for rel, (_, after) in targets.items():
            expected[rel] = after
    for rel, (_, after, _) in OVERLAY.items():
        expected[rel] = after
    bad = []
    for rel in sorted(expected):
        target = root / rel
        if not target.is_file():
            bad.append(rel)
            print(f"  MISSING {rel}")
            continue
        got = sha256_file(target)
        print(f"  {'OK' if got == expected[rel] else 'MISMATCH'} {rel} {got}")
        if got != expected[rel]:
            bad.append(rel)
        if rel.endswith(".py"):
            try:
                ast.parse(read_text(target))
            except SyntaxError as exc:
                die(f"{rel}: syntax check failed: {exc}")
    if bad:
        die(f"{len(bad)} file(s) not at final hashes: {sorted(bad)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help="directory containing vllm/ (default: %(default)s)")
    ap.add_argument("--verify-only", action="store_true",
                    help="do not patch; assert every target is at its final hash")
    args = ap.parse_args()
    root = Path(args.root)
    if not (root / "vllm").is_dir():
        die(f"{root}: no vllm/ tree (base image {BASE_IMAGE} expected)")
    print(f"patch installer root={root} base={BASE_IMAGE}")
    if args.verify_only:
        verify(root)
        print("verify-only OK")
        return
    plans = plan_all(root)
    print(f"preflight OK: {len(plans)} file writes derived and validated in memory")
    install(plans)
    verify(root)
    print("all patches applied and final hashes verified")


if __name__ == "__main__":
    main()
