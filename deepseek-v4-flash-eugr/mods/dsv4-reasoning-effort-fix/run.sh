#!/bin/bash
# DeepSeek V4 reasoning-effort fix for the vllm-node-b12x / SparkInfer container.
#
# Symptom: DeepSeek-V4-Flash-0731 ships a THREE-level reasoning_effort
# (low / high / max) with distinct prompt prefixes, but the fork's bundled
# vllm/tokenizers/deepseek_v4_encoding.py only implements a TWO-level scheme:
#   - "max" gets a single "Absolute maximum ..." prefix,
#   - "high" and "low" get NO prefix at all,
# and apply_chat_template in vllm/tokenizers/deepseek_v4.py routes "low" into the
# "high" bucket. Result: every reasoning level is scrambled/collapsed.
#
# Fix: align the encoding module with the official 0731 spec (3 levels with
# distinct prefixes) and correct the apply_chat_template routing.
set -euo pipefail

ENCODING=/usr/local/lib/python3.12/dist-packages/vllm/tokenizers/deepseek_v4_encoding.py
TOKENIZER=/usr/local/lib/python3.12/dist-packages/vllm/tokenizers/deepseek_v4.py

for f in "$ENCODING" "$TOKENIZER"; do
  [ -f "$f" ] || { echo "--- [dsv4-reffix] missing $f; ABORT."; exit 1; }
done

# ---------------------------------------------------------------------------
# Patch 1: deepseek_v4_encoding.py - replace the single REASONING_EFFORT_MAX
# constant with the official 3-level dict and use it in render_message.
# ---------------------------------------------------------------------------
python3 - "$ENCODING" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()

marker = "dsv4_reffix_three_level_reasoning"
if marker in text:
    print("--- [dsv4-reffix] deepseek_v4_encoding.py already patched.")
    raise SystemExit(0)

def fail(message: str) -> None:
    print(f"--- [dsv4-reffix] {message}")
    raise SystemExit(1)

# 1a) Swap the single REASONING_EFFORT_MAX constant for the 3-level dict.
old_const_start = "REASONING_EFFORT_MAX = ("
old_const_end = ")\n"
if old_const_start not in text:
    fail("REASONING_EFFORT_MAX anchor not found; layout changed.")
head, tail = text.split(old_const_start, 1)
constant_block, tail = tail.split(old_const_end, 1)
new_const = (
    "REASONING_EFFORT_PROMPTS = {\n"
    '    "low": "",\n'
    '    "high": (\n'
    '        "Reasoning Effort: Absolute maximum with no shortcuts permitted.\\n"\n'
    '        "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\\n"\n'
    '        "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely nothing is left unchecked.\\n\\n"\n'
    '    ),\n'
    '    "max": (\n'
    '        "Reasoning Effort: Beyond maximum \\u2014 exhaustive, relentless, and uncompromising.\\n"\n'
    '        "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\\n"\n'
    '        "Do not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\\n\\n"\n'
    '    ),\n'
    "}\n"
)
text = head + new_const + tail

# 1b) Fix the assertion + prefix logic in render_message. Mirror the official
# encoder: default None -> "low" (matching the assert in the tokenizer), then
# validate against the 3-level dict.
old_assert = "    assert reasoning_effort in ['max', None, 'high'], f\"Invalid reasoning effort: {reasoning_effort}\"\n"
if old_assert not in text:
    fail("reasoning_effort assertion anchor not found; layout changed.")
text = text.replace(
    old_assert,
    "    reasoning_effort = reasoning_effort or \"low\"\n"
    "    assert reasoning_effort in REASONING_EFFORT_PROMPTS, \\\n"
    "        f\"Invalid reasoning effort: {reasoning_effort}, expected one of {list(REASONING_EFFORT_PROMPTS)}\"\n",
    1,
)

old_if = "    if index == 0 and thinking_mode == \"thinking\" and reasoning_effort == 'max':\n        prompt += REASONING_EFFORT_MAX\n"
if old_if not in text:
    fail("reasoning_effort prefix anchor not found; layout changed.")
text = text.replace(
    old_if,
    "    if index == 0 and thinking_mode == \"thinking\":\n"
    "        prompt += REASONING_EFFORT_PROMPTS[reasoning_effort]\n",
    1,
)

path.write_text(text)
print("--- [dsv4-reffix] patched deepseek_v4_encoding.py")
PY

# ---------------------------------------------------------------------------
# Patch 2: deepseek_v4.py - fix apply_chat_template routing so "low" is not
# swallowed into the "high" bucket and each level passes through verbatim.
# ---------------------------------------------------------------------------
python3 - "$TOKENIZER" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()

marker = "dsv4_reffix_routing"
if marker in text:
    print("--- [dsv4-reffix] deepseek_v4.py already patched.")
    raise SystemExit(0)

def fail(message: str) -> None:
    print(f"--- [dsv4-reffix] {message}")
    raise SystemExit(1)

old = (
    "            reasoning_effort = kwargs.get(\"reasoning_effort\")\n"
    "            if not isinstance(reasoning_effort, str):\n"
    "                reasoning_effort = None\n"
    "            elif reasoning_effort == \"none\":\n"
    "                thinking_mode = \"chat\"\n"
    "                reasoning_effort = None\n"
    "            elif reasoning_effort in (\"max\", \"xhigh\"):\n"
    "                reasoning_effort = \"max\"\n"
    "            else:\n"
    "                reasoning_effort = \"high\"\n"
)
if old not in text:
    fail("apply_chat_template reasoning_effort block not found; layout changed.")
new = (
    "            reasoning_effort = kwargs.get(\"reasoning_effort\")\n"
    "            if not isinstance(reasoning_effort, str):\n"
    "                reasoning_effort = None\n"
    "            elif reasoning_effort == \"none\":\n"
    "                thinking_mode = \"chat\"\n"
    "                reasoning_effort = None\n"
    "            elif reasoning_effort == \"max\":\n"
    "                reasoning_effort = \"max\"\n"
    "            elif reasoning_effort == \"xhigh\":\n"
    "                # Backward compat: xhigh is not a V4 level; fold into max.\n"
    "                reasoning_effort = \"max\"\n"
    "            else:\n"
    "                # low and high pass through verbatim.\n"
    "                reasoning_effort = reasoning_effort\n"
)
text = text.replace(old, new, 1)
path.write_text(text)
print("--- [dsv4-reffix] patched deepseek_v4.py")
PY

echo "=== OK"
