"""Prove when the SWA_THIN hook fires: cold multi-chunk prefill vs warm one-pass.

Run inside the engine container. Sends two requests with the same long body:
  1. cold   -- nothing cached, extend_len > chunked_prefill_size => chunked steps
  2. warm   -- same body again, most tokens cached => one batch, finishes immediately
Prints the hook counters and the matching "Prefill batch" lines so the call path is
attributable rather than inferred.
"""
import json
import sys
import time
import urllib.request

sys.path.insert(0, "/opt/dsv41")

import sitecustomize  # noqa: F401  (registers the import hook from the container env)
import sglang.srt.mem_cache.unified_radix_cache as urc  # noqa: F401  (triggers the install)

import swa_thin

s = swa_thin.STATS
URL = "http://127.0.0.1:8000/v1/chat/completions"


def snap():
    return dict(
        active=s.active,
        calls=s.calls,
        prompt_calls=s.prompt_calls,
        demoted=s.demoted,
        errors=s.errors,
    )


def send(body, timeout=600):
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    started = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
        return resp.status, time.time() - started, payload.get("usage", {})


# ~24K tokens: well past chunked_prefill_size so a cold prefill must split into chunks.
filler = "The quick brown fox jumps over the lazy dog. " * 2400
body = {
    "model": "deepseek-v41-flash",
    "messages": [{"role": "user", "content": filler + " Reply with exactly: OK"}],
    "max_tokens": 8,
    "stream": False,
}

print("=== cold (expect chunked prefill) ===")
before = snap()
status, elapsed, usage = send(body)
after = snap()
print(f"status={status} elapsed={elapsed:.1f}s prompt_tokens={usage.get('prompt_tokens')}")
print("  counters before:", json.dumps(before))
print("  counters after: ", json.dumps(after))
print("  moved:", json.dumps({k: after[k] - before[k] for k in ("calls", "prompt_calls", "demoted")}))

print("=== warm (same body, expect one-pass) ===")
before = snap()
status, elapsed, usage = send(body)
after = snap()
print(f"status={status} elapsed={elapsed:.1f}s prompt_tokens={usage.get('prompt_tokens')}")
cached = usage.get("prompt_tokens_details") or {}
print(f"  cached_tokens={cached.get('cached_tokens')}")
print("  moved:", json.dumps({k: after[k] - before[k] for k in ("calls", "prompt_calls", "demoted")}))
