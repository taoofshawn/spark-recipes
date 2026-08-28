#!/usr/bin/env python3
"""6-way concurrent tool-carrying load test for the GLM-5.3-Flash recipe.

Modeled on the Qwen recipe's load_test_qwen.py (which originated in
tonyd2wild's Qwen fleet repo). Single-request smokes cannot catch the
day-0 failure classes on this stack (silent FP4 MoE corruption loops,
thinking+tools issues, NaN logits) — those only surface under concurrent,
tool-carrying sessions. Run after every config change and check for a
PASS verdict.

Usage:
    python3 tools/load_test_glm.py [base_url]

Exit code 0 = PASS, 1 = FAIL.
"""

import concurrent.futures
import json
import sys
import time
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:4000/v1/chat/completions"

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file from disk", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "glob", "description": "Find files matching a pattern", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "run_command", "description": "Run a shell command", "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}},
]

PROMPTS = [
    "Summarize the pros and cons of tensor parallelism vs pipeline parallelism in about 300 words. Do not call tools for this.",
    "Explain how speculative decoding with a built-in MTP layer works, about 300 words. No tools needed.",
    "Use the glob tool to find all python files under /src, then explain what you would do next.",
    "Write a 250-word explanation of NVFP4 quantization tradeoffs. No tools needed.",
    "Use the read_file tool to read /etc/hostname, then describe what you'd check next on a GPU server.",
    "Describe the InfiniBand vs RoCE tradeoffs for a 2-node tensor-parallel deployment in 300 words. No tools.",
]


def one(i):
    body = {
        "model": "glm-5.3-flash",
        "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
        "tools": TOOLS,
        "max_tokens": 400,
        "temperature": 0.7,
    }
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    dur = time.perf_counter() - start
    msg = data["choices"][0]["message"]
    content = (msg.get("content") or "") + ""
    # A repeated single-token loop is the signature of the day-0 corruption
    # classes (FP4 MoE garbage, token-0 loops). Flag any degenerate repetition.
    degenerate = False
    for tok in ("!!!!", "locklock", "[[[[[", "%%%%%"):
        if tok in content:
            degenerate = True
    usage = data.get("usage", {})
    return {
        "req": i,
        "secs": round(dur, 1),
        "finish": data["choices"][0].get("finish_reason"),
        "tool_call": bool(msg.get("tool_calls")),
        "completion_tokens": usage.get("completion_tokens"),
        "degenerate": degenerate,
        "content_head": content[:80].replace("\n", " "),
    }


results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
    futs = [ex.submit(one, i) for i in range(6)]
    for f in concurrent.futures.as_completed(futs):
        try:
            r = f.result()
        except Exception as e:
            r = {"error": str(e)}
        results.append(r)
        print(json.dumps(r), flush=True)

bad = [r for r in results if r.get("degenerate") or "error" in r]
print("VERDICT:", "FAIL - degenerate output or errors" if bad else "PASS - all 6 concurrent streams clean")
sys.exit(1 if bad else 0)
