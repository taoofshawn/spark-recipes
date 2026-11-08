"""Drive a decode stream FIRST, then a long prefill, and read the yield receipts.

arm_after only arms when the plan's batch is the chunked request's EXTEND batch AND
plan.running_batch still has decode rows. Starting the long prefill first (the earlier
attempt) leaves no decode rows beside it once its first chunk lands. So this starts a
long-lived streaming decode, then launches a cold multi-chunk prefill beside it.
"""
import json
import re
import subprocess
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"


def logs():
    out = subprocess.run(
        ["docker", "logs", "dsv41-head"], capture_output=True, text=True, errors="replace"
    )
    return (out.stdout or "") + (out.stderr or "")


def yield_lines():
    return [ln for ln in logs().splitlines() if "decode_yield" in ln.lower() or "DYIELD" in ln]


def long_prefill(nonce, tokens=300000):
    filler = "The quick brown fox jumps over the lazy dog. " * (tokens // 12)
    body = json.dumps(
        {
            "model": "deepseek-v41-flash",
            "messages": [{"role": "user", "content": f"[{nonce}] " + filler + " Reply with exactly: OK"}],
            "max_tokens": 4,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        urllib.request.urlopen(req, timeout=1800).read()
    except Exception as exc:  # noqa: BLE001
        print(f"  long prefill failed: {exc!r}")
    return time.time() - t0


def long_decode(nonce, seconds):
    """Stream tokens for up to `seconds`, recording per-token timestamps."""
    body = json.dumps(
        {
            "model": "deepseek-v41-flash",
            "messages": [
                {
                    "role": "user",
                    "content": f"[{nonce}] Write a long detailed paragraph about the history of computing.",
                }
            ],
            "max_tokens": 2000,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    stamps = []
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                d = ((obj.get("choices") or [{}])[0].get("delta") or {})
                if d.get("content") or d.get("reasoning_content"):
                    stamps.append(time.time() - t0)
                if time.time() - t0 > seconds:
                    break
    except Exception as exc:  # noqa: BLE001
        print(f"  long decode failed: {exc!r}")
    return stamps


before = yield_lines()
print(f"yield lines before: {len(before)}")

nonce = f"{int(time.time() * 1000)}"

# 1. start a long streaming decode so decode rows exist
stamps_holder = {}
dt = threading.Thread(target=lambda: stamps_holder.setdefault("s", long_decode(f"{nonce}-D", 40)), daemon=True)
dt.start()
time.sleep(2.0)  # let decode get going and produce rows

# 2. launch a cold multi-chunk prefill beside it
t0 = time.time()
long_wall = long_prefill(f"{nonce}-L")
dt.join(timeout=60)

stamps = stamps_holder.get("s") or []
print(f"long prefill wall: {long_wall:.1f}s; decode tokens seen: {len(stamps)}")
if len(stamps) > 5:
    # gaps between consecutive tokens during the prefill window
    gaps = [round(b - a, 3) for a, b in zip(stamps, stamps[1:])]
    big = [g for g in gaps if g > 1.0]
    print(f"  max gap {max(gaps):.2f}s, gaps >1s: {len(big)}")
    print(f"  first 10 stamps: {[round(s,2) for s in stamps[:10]]}")

time.sleep(2)
after = yield_lines()
new = [ln for ln in after if ln not in before]
print(f"new yield lines: {len(new)}")
for ln in new[-5:]:
    print("  ", ln.strip()[:220])
