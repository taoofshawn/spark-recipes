"""Check whether DSV41_DECODE_YIELD is installed on the live Scheduler class.

Run inside the engine container. Prints the mark on get_next_batch_to_run and the
adapter's parsed share, so "armed" in the boot log can be checked against the class.
"""
import sys

sys.path.insert(0, "/opt/dsv41")
import sitecustomize  # noqa: F401

import sglang.srt.managers.scheduler as S
import decode_yield as D

fn = S.Scheduler.get_next_batch_to_run
print("marked get_next_batch_to_run:", getattr(fn, "_dsv41_decode_yield", None))
print("wrapped:", hasattr(fn, "__wrapped__"))
print("defer marked:", getattr(S.Scheduler._should_defer_prefill, "_dsv41_decode_yield", None))
print("STATS installed:", D.STATS.installed, "pct:", D.STATS.pct)
print("ENV DSV41_DECODE_YIELD:", __import__("os").environ.get("DSV41_DECODE_YIELD"))
