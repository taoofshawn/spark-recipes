"""Cross the thin adapter's RECEIPT_EVERY=1024 threshold to prove the hook fires.

The adapter logs on the 1st hooked insert and then every 1024th. One receipt at boot
therefore does not mean "never fires" -- it means "fired once and has been counting
silently since". Send enough cold chunked prompts to cross 1024 inserts and read the
next receipt line.
"""
import json
import re
import subprocess
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"


def logs():
    out = subprocess.run(
        ["docker", "logs", "dsv41-head"], capture_output=True, text=True, errors="replace"
    )
    return (out.stdout or "") + (out.stderr or "")


def receipts():
    return re.findall(r"SWA_THIN receipt[^\n]*", logs())


def send(nonce, tokens=20000):
    # Unique prefix per call so each is a cold miss; long enough to chunk.
    filler = "The quick brown fox jumps over the lazy dog. " * (tokens // 12)
    body = json.dumps(
        {
            "model": "deepseek-v41-flash",
            "messages": [
                {"role": "user", "content": f"[{nonce}] " + filler + " Reply with exactly: OK"}
            ],
            "max_tokens": 8,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return r.status


budget = int(sys.argv[1]) if len(sys.argv) > 1 else 40
start = receipts()
print(f"receipts before: {len(start)}")
if start:
    print(f"  last: {start[-1].strip()}")

t0 = time.time()
for i in range(budget):
    try:
        send(f"cold-{int(time.time())}-{i}")
    except Exception as exc:  # noqa: BLE001
        print(f"  request {i} failed: {exc!r}")
        break
    if (i + 1) % 10 == 0:
        print(f"  sent {i + 1} ({time.time() - t0:.0f}s)")

time.sleep(3)
now = receipts()
print(f"receipts after: {len(now)}  (+{len(now) - len(start)})")
for line in now[-4:]:
    print("  ", line.strip())
