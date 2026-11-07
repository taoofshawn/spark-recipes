"""Measure DSV41_DECODE_YIELD: a short stream's rate while a COLD long prefill runs.

Upstream's receipt: agents decoding beside a long prefill ran at 0.080 -> 0.393 of their
normal rate. The yield only engages "after a non-final chunk beside running streams", so
the long request must be genuinely cold and multi-chunk -- a warm cached document produces
too few chunks to exercise it. Each run uses a unique nonce so the long prompt is cold.
"""
import json
import sys
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
LONG_TOKENS = 300000


def long_prefill(nonce):
    filler = "The quick brown fox jumps over the lazy dog. " * (LONG_TOKENS // 12)
    body = json.dumps(
        {
            "model": "deepseek-v41-flash",
            "messages": [
                {
                    "role": "user",
                    "content": f"[{nonce}] " + filler + " Reply with exactly: OK",
                }
            ],
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


def short_stream(nonce, max_tokens=120):
    body = json.dumps(
        {
            "model": "deepseek-v41-flash",
            "messages": [
                {"role": "user", "content": f"[{nonce}] Count from one to twenty in words."}
            ],
            "max_tokens": max_tokens,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    n = 0
    with urllib.request.urlopen(req, timeout=1200) as resp:
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
            choices = obj.get("choices") or [{}]
            delta = choices[0].get("delta") or {}
            text = delta.get("content") or delta.get("reasoning_content") or ""
            if text:
                if ttft is None:
                    ttft = time.time() - t0
                n += 1
    wall = time.time() - t0
    return {"ttft": ttft, "tokens": n, "wall": wall, "rate": (n / wall if wall else 0.0)}


def show(label, r):
    ttft = f"{r['ttft']:.2f}s" if r["ttft"] is not None else "n/a"
    print(f"{label:8} ttft={ttft:>7} tokens={r['tokens']:>3} wall={r['wall']:>6.2f}s rate={r['rate']:.2f} tok/s")


nonce = f"{int(time.time() * 1000)}"
results = {}

# idle baseline first
results["idle"] = short_stream(f"{nonce}-idle")
show("IDLE", results["idle"])

# now a short stream beside a COLD multi-chunk prefill
holder = {}


def run_long():
    holder["elapsed"] = long_prefill(f"{nonce}-long")


t = threading.Thread(target=run_long, daemon=True)
t.start()
time.sleep(1.5)  # let the long prefill enter its chunk loop
results["beside"] = short_stream(f"{nonce}-beside")
show("BESIDE", results["beside"])
t.join(timeout=1800)
print(f"  long prefill wall: {holder.get('elapsed', -1):.1f}s")

idle_rate = results["idle"]["rate"]
beside_rate = results["beside"]["rate"]
ratio = round(beside_rate / idle_rate, 3) if idle_rate else 0
print(f"RESULT beside/idle = {ratio} (upstream receipt: 0.393 at DECODE_YIELD=33)")
