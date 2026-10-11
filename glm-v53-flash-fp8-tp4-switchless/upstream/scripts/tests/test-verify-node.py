#!/usr/bin/env python3
"""Offline checks of the node verifier's selected operator-payload hashes."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
source = (REPO / "scripts/verify-node.sh").read_text(encoding="utf-8")
match = re.search(r"^check_selected_payload\(\) \{.*?^\}", source, re.M | re.S)
assert match, "selected-payload verifier missing"
function = match.group(0)


def check(path: str, expected: str) -> str:
    result = subprocess.run(
        ["bash", "-c", function + '\ncheck_selected_payload "$1" "$2"',
         "verify-selected-payload", path, expected],
        capture_output=True, text=True, check=True,
    )
    assert not result.stderr, result.stderr
    return result.stdout


with tempfile.TemporaryDirectory(prefix="tp4-verify-payload.") as temp:
    root = Path(temp)
    current = root / "spark_context_cache_connector.py"
    previous = root / "spark_context_cache_connector-20260918.py"
    encoder = root / "spark_context_cache_hybrid.py"
    current.write_text("current connector\n", encoding="utf-8")
    previous.write_text("historical connector\n", encoding="utf-8")
    encoder.write_text("current encoder\n", encoding="utf-8")
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()

    assert check(str(current), sha(current)) == "ok"
    assert check(str(previous), sha(previous)) == "ok"
    assert check(str(current), sha(previous)) == "mismatch"
    assert check(str(encoder), sha(encoder)) == "ok"
    assert check(str(root / "missing.py"), sha(current)) == "missing"
    assert check(str(current), "") == "invalid-pin"
    assert check(str(current), "not-a-sha256") == "invalid-pin"
    assert check("", "") == "disabled"

    # Exercise the launcher's supported home-relative forms without changing HOME.
    relative = os.path.relpath(current, Path.home())
    assert check("$HOME/" + relative, sha(current)) == "ok"
    assert check("~/" + relative, sha(current)) == "ok"
    original_sha = sha(current)
    current.write_text("changed after deployment\n", encoding="utf-8")
    assert check(str(current), original_sha) == "mismatch"

assert "P_SPARKCACHE_CONNECTOR_SHA256='${SPARKCACHE_CONNECTOR_SHA256:-}'" in source

# Dual-lib SHA selection: the expected NCCL sha must follow NCCL_DIR's basename
# (nccl-patched -> the 2-PF line, nccl-patched-4pf -> the 4-PF line), and an
# unlisted staging dir must abort the run rather than pick any line.
match = re.search(r"^NCCL_SHA_SRC=.*?^GID_CHECK_SHA=\$\(.*?\)\n", source, re.M | re.S)
assert match, "NCCL sha selection block missing from verify-node.sh"
block = match.group(0)

with tempfile.TemporaryDirectory(prefix="tp4-verify-nccl-pin.") as temp:
    root = Path(temp)
    sums = root / "scripts/node/nccl/SHA256SUMS"
    sums.parent.mkdir(parents=True)
    sums.write_text(
        "afe5f48626284eae89988516e450c7f20cc303904ba4b7083b88aa3ca1e9b85f  nccl-patched/libnccl.so.2\n"
        "8de68def30bcde20fabded468bd60f0df82ea12d3ead02d5120c7feda7a41a87  nccl-patched-4pf/libnccl.so.2\n",
        encoding="utf-8",
    )
    two_pf = root / "nccl-patched"
    four_pf = root / "nccl-patched-4pf"
    two_pf.mkdir()
    four_pf.mkdir()
    (two_pf / "libnccl.so.2").write_text("two-pf lib\n", encoding="utf-8")
    (four_pf / "libnccl.so.2").write_text("four-pf lib\n", encoding="utf-8")
    # The block calls die() (from lib/common.sh) and shasums the GID checker; both are
    # outside the block. A stub die lets expect_sha's abort path return non-zero instead
    # of failing on the missing helper, and the checker file satisfies the shasum probe.
    preamble = 'die() { echo "$1" >&2; exit 1; }\n'
    gid_check = root / "scripts/nccl_gid_check.py"
    gid_check.write_text("# test stub\n", encoding="utf-8")
    def expect_sha(nccl_dir: str) -> tuple[int, str, str]:
        result = subprocess.run(
            ["bash", "-c", preamble + block + '\nprintf "%s" "$NCCL_SHA"'],
            capture_output=True, text=True,
            env={**os.environ, "REPO": str(root), "NCCL_DIR": nccl_dir},
        )
        return result.returncode, result.stdout, result.stderr

    rc, out, err = expect_sha(str(two_pf))
    assert rc == 0 and not err, err
    assert out == "afe5f48626284eae89988516e450c7f20cc303904ba4b7083b88aa3ca1e9b85f"
    rc, out, err = expect_sha(str(four_pf))
    assert rc == 0 and not err, err
    assert out == "8de68def30bcde20fabded468bd60f0df82ea12d3ead02d5120c7feda7a41a87"
    rc, out, err = expect_sha(str(root / "other"))
    assert rc != 0, "unlisted staging dir must abort"
    assert "SHA256SUMS" in err
assert "P_SPARKCACHE_ENCODER_SHA256='${SPARKCACHE_ENCODER_SHA256:-}'" in source
print("test-verify-node: PASS")
