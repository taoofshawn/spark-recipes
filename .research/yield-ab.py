"""Quantify DSV41_DECODE_YIELD: inter-token gaps in a decode stream beside a long prefill.

Starts a long streaming decode, waits for it to reach steady state, then launches a cold
multi-chunk prefill beside it. Reports the decode stream's inter-token gap distribution
during the prefill window. Run once per boot configuration (yield on / yield off) and
compare the medians and the count of long stalls.
"""
import json
import statistics
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
LONG_TOKENS = 300000
DECODE_SECONDS = 45


def long_prefill(nonce):
    filler = "The quick brown fox jumps over the lazy dog. " * (LONG_TOKENS // 12)
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
    body = json.dumps(
        {
            "model": "deepseek-v41-flash",
            "messages": [
                {
                    "role": "user",
                    "content": f"[{nonce}] Write a long detailed paragraph about the history of computing.",
                }
            ],
            "max_tokens": 3000,
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


nonce = f"{int(time.time() * 1000)}"
holder = {}

dt = threading.Thread(target=lambda: holder.setdefault("s", long_decode(f"{nonce}-D", DECODE_SECONDS)), daemon=True)
dt.start()
time.sleep(2.0)

# launch the long prefill while decode is streaming
pt = threading.Thread(target=lambda: holder.setdefault("p", long_prefill(f"{nonce}-L")), daemon=True)
pt.start()

dt.join(timeout=DECODE_SECONDS + 60)
pt.join(timeout=1800)

stamps = holder.get("s") or []
prefill_wall = holder.get("p", -1)

if len(stamps) < 5:
    print(f"too few decode tokens ({len(stamps)}) to measure")
    raise SystemExit(1)

gaps = [b - a for a, b in zip(stamps, stamps[1:])]
gaps.sort()
n = len(gaps)
p50 = statistics.median(gaps)
p90 = gaps[int(n * 0.9)] if n > 10 else gaps[-1]
p99 = gaps[int(n * 0.99)] if n > 100 else gaps[-1]
stalls = len([g for g in gaps if g > 1.0])
stall_time = sum(g for g in gaps if g > 1.0)

print(f"nonce={nonce}")
print(f"decode tokens: {len(stamps)} over {stamps[-1]:.1f}s")
print(f"prefill wall: {prefill_wall:.1f}s")
print(f"gap p50={p50:.3f}s p90={p90:.3f}s p99={p99:.3f}s max={gaps[-1]:.2f}s")
print(f"stalls >1s: {stalls}  total stall time: {stall_time:.1f}s")
print(f"steady rate during prefill: {len(stamps) / stamps[-1]:.2f} tok/s")
