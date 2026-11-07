"""Measure the SWA_THIN adapter's effect with a guaranteed-cold document each run.

Earlier runs reused a document that was already cached, so "cold" timings were warm
hits. Each run now embeds a unique nonce so A is a genuine cold prefill. Reports the
hit rate and the revisit latency so thin-on and thin-off are comparable.
"""
import json
import os
import statistics
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
DOC_TOKENS = 96000
REVISITS = 4


def call(messages, max_tokens=4, timeout=900):
    body = json.dumps(
        {"model": "deepseek-v41-flash", "messages": messages, "max_tokens": max_tokens, "stream": False}
    ).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read())
    elapsed = time.time() - t0
    usage = payload.get("usage", {}) or {}
    details = usage.get("prompt_tokens_details") or {}
    return {
        "elapsed": round(elapsed, 2),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens"),
    }


def hit_rate(r):
    p = r["prompt_tokens"] or 0
    c = r["cached_tokens"] or 0
    return round(100.0 * c / p, 1) if p else 0.0


# Unique nonce so this document has never been seen by the engine.
nonce = f"{int(time.time() * 1000)}-{os.getpid()}"
filler = "The quick brown fox jumps over the lazy dog. " * (DOC_TOKENS // 12)
doc = [{"role": "user", "content": f"Document {nonce}.\n" + filler}]
thin = os.environ.get("DSV41_SWA_THIN")
print(f"SWA_THIN={thin} nonce={nonce}")

a = call(doc)
print(f"A  cold     cached={a['cached_tokens']}/{a['prompt_tokens']} ({hit_rate(a)}%) {a['elapsed']}s")

revisits = []
for i in range(REVISITS):
    r = call(doc)
    revisits.append(r)
    print(f"B{i} revisit cached={r['cached_tokens']}/{r['prompt_tokens']} ({hit_rate(r)}%) {r['elapsed']}s")

print("   churning 3 x ~40K unrelated sessions")
for i in range(3):
    churn = [{"role": "user", "content": f"[churn {nonce} {i}] " + "Unrelated text about weather. " * 3500}]
    c = call(churn)
    print(f"   churn{i}  cached={c['cached_tokens']}/{c['prompt_tokens']} ({hit_rate(c)}%)")

c = call(doc)
print(f"C  churned  cached={c['cached_tokens']}/{c['prompt_tokens']} ({hit_rate(c)}%) {c['elapsed']}s")

med = statistics.median([r["elapsed"] for r in revisits])
print(
    "RESULT thin=%s cold=%ss revisit_median=%ss churned=%ss B_hit=%s%% C_hit=%s%%"
    % (thin, a["elapsed"], med, c["elapsed"], hit_rate(revisits[0]), hit_rate(c))
)
