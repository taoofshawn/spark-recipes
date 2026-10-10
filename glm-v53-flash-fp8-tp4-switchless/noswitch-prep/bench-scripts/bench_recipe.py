#!/usr/bin/env python3
"""bench_recipe.py — before/after regression benchmark for spark-recipes recipes.

Run on the HEAD node against the serving engine (default :8000):

  python3 bench_recipe.py bench --label before --model glm-5.3-flash \
      --container glm53-intel-w4a16 --out benchmarks/20260919-before
  python3 bench_recipe.py compare benchmarks/20260919-before benchmarks/20260919-after

Design invariants (each one was paid for in a real session — do not "fix"):
- Streams with `stream_options.include_usage` and counts tokens from the FINAL
  usage chunk. Streamed deltas under-report tok/s up to ~4x (steps/s, not
  tok/s); some GLM-intel builds also omit `usage` on non-stream responses.
  Final usage + wall clock is the valid pair either way.
- Prompts are baked byte-identical constants (no timestamps, no salts):
  before/after boots must see identical bytes or prefix-cache state (PMU128)
  differs and prefill numbers lie.
- temp 0 for variance control. At temp 1.0, c1 decode is bimodal spec-decode
  acceptance luck (same config sampled 17-32 tok/s) — c1 cells are
  informational; the cN aggregate cell is the regression discriminator.
- Rates always use actual `usage.completion_tokens` (EOS truncation makes
  max_tokens an upper bound, not a count).
- The warm-up stage is mandatory: fresh boots are ~30% slower until several
  hundred-token generations pass; short calls do not clear it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

# UTF-8 out regardless of the host console (Δ/≈/– appear in reports).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# --------------------------------------------------------------------------
# Fixed prompts — DO NOT change between a before run and an after run.
# --------------------------------------------------------------------------
_PARA = (
    "The shared memory bus of the machine carried register traffic between the "
    "arithmetic unit and the backing store, and each microcycle moved sixteen "
    "words while the drum rotated into position for the next track fetch. "
    "Engineers logged the error rates by hand and plotted them against ambient "
    "temperature, because the drift was thermal before it was logical."
)
_CODE = (
    "def checksum(rows):\n"
    "    total = 0\n"
    "    for row in rows:\n"
    "        total ^= sum(row)\n"
    "    return total\n"
    "\n"
    "def load(path):\n"
    "    with open(path) as fh:\n"
    "        return [line.split() for line in fh]\n"
    "\n"
)


def _repeat(unit: str, approx_tokens: int, chars_per_token: float = 4.0) -> str:
    """Repeat `unit` until it is roughly `approx_tokens` tokens long.

    The estimate only shapes the prompt; the engine's own usage.prompt_tokens
    is what gets recorded, so the approximation never enters the numbers.
    """
    n = max(1, int(approx_tokens * chars_per_token / len(unit)))
    return (unit * n).strip()


PROMPTS = {
    "short_prose": _repeat(_PARA, 1000),
    "short_code": _repeat(_CODE, 1000),
    "medium": _repeat(_PARA, 16000),
    "long": _repeat(_PARA, 64000),
}

WARMUP = [
    (_repeat(_PARA, 800), 640),
    ("Implement a binary search tree in Python with insert, search, delete and "
     "in-order traversal. Code only.", 640),
    (_repeat(_PARA, 400), 384),
]

# Metrics counters (Prometheus text format; values may repeat per engine rank).
_DRAFT_RE = re.compile(
    r"^vllm:spec_decode_num_draft_tokens_total(?:\{[^}]*\})?\s+([0-9.eE+]+)\s*$", re.M)
_ACCEPTED_RE = re.compile(
    r"^vllm:spec_decode_num_accepted_tokens_total(?:\{[^}]*\})?\s+([0-9.eE+]+)\s*$", re.M)

PMU_UNIT = 128  # --prefix-match-unit 128 (PMU128 profiles)


# --------------------------------------------------------------------------
# HTTP plumbing
# --------------------------------------------------------------------------
def _post_stream(url: str, model: str, prompt: str, max_tokens: int,
                 temperature: float, timeout: float) -> dict:
    """One streaming completion; returns token counts + wall + TTFT."""
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if ttft is None and (obj.get("choices") or obj.get("usage")):
                ttft = time.perf_counter() - t0
            if obj.get("usage"):
                usage = obj["usage"]
    wall = time.perf_counter() - t0
    u = usage or {}
    details = u.get("prompt_tokens_details") or {}
    return {
        "completion_tokens": u.get("completion_tokens"),
        "prompt_tokens": u.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens"),
        "wall": round(wall, 3),
        "ttft": round(ttft, 4) if ttft is not None else None,
    }


def _get_json(url: str, path: str, timeout: float = 30) -> dict:
    with urllib.request.urlopen(url.rstrip("/") + path, timeout=timeout) as r:
        return json.load(r)


def _get_metrics(url: str, timeout: float = 30) -> dict:
    with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=timeout) as r:
        text = r.read().decode("utf-8", "replace")
    draft = sum(float(m) for m in _DRAFT_RE.findall(text))
    accepted = sum(float(m) for m in _ACCEPTED_RE.findall(text))
    return {
        "draft_tokens_total": draft if _DRAFT_RE.search(text) else None,
        "accepted_tokens_total": accepted if _ACCEPTED_RE.search(text) else None,
    }


def _mem_available_gib() -> float | None:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return round(float(line.split()[1]) / (1024 * 1024), 2)
    except OSError:
        pass
    return None


def _docker_log_markers(container: str) -> dict:
    """Boot-marker extraction from the head container's log (best effort).

    Tries the last 3000 lines first; on a long-running boot the boot-time
    markers (KV pool line, spec config) scroll out of the tail, so fall back
    to the FULL log for those lines only. The traceback count stays
    tail-scoped: the full log may hold tracebacks from an earlier failed
    boot of the same container.
    """
    try:
        out = _docker_log_text(container, tail=3000)
    except Exception as exc:  # container name wrong / no docker — record, don't die
        return {"error": f"docker logs failed: {exc}"}

    def _find(text: str, *needles: str) -> str | None:
        for line in text.splitlines():
            if any(n in line for n in needles):
                return line.strip()
        return None

    kv = _find(out, "GPU KV cache size")
    spec = _find(out, "speculative_config", "Resolved architecture")
    if kv is None or spec is None:
        try:
            full = _docker_log_text(container)
            if kv is None:
                kv = _find(full, "GPU KV cache size")
            if spec is None:
                spec = _find(full, "speculative_config", "Resolved architecture")
        except Exception:
            pass  # keep tail-scoped results
    return {
        "kv_cache_line": kv,
        "spec_line": spec,
        "tracebacks": out.count("Traceback (most recent call last)"),
    }


def _argv_has(container: str, needle: str) -> bool:
    """True if `needle` is in the engine's live argv (config-actually-took check)."""
    try:
        out = subprocess.run(
            ["docker", "exec", container, "sh", "-c",
             'tr "\\0" "\\n" < /proc/1/cmdline'],
            capture_output=True, text=True, timeout=60,
        ).stdout
        return needle in out
    except Exception:
        return False


def _docker_log_text(container: str, tail: int | None = None) -> str:
    cmd = ["docker", "logs"]
    if tail is not None:
        cmd += ["--tail", str(tail)]
    cmd += [container]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return r.stderr + r.stdout


def _container_age_minutes(container: str) -> float | None:
    """Boot age from docker inspect — the boot-state-matching check.

    Fresh boot (<60 min) vs long-warm boot changes decode numbers by more
    than the regression threshold; before/after sides must match.
    """
    try:
        out = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.StartedAt}}", container],
            capture_output=True, text=True, timeout=60,
        ).stdout.strip()
        started = datetime.fromisoformat(out.replace("Z", "+00:00"))
        return round((datetime.now(timezone.utc) - started).total_seconds() / 60, 1)
    except Exception:
        return None


# --------------------------------------------------------------------------
# Bench
# --------------------------------------------------------------------------
def _cell_list(tier: int, rounds: int, max_conc: int) -> list:
    cells = [
        ("c1_prose_short", 1, "short_prose", 512, rounds),
        ("c1_code_short", 1, "short_code", 512, rounds),
        ("c1_prose_medium", 1, "medium", 256, max(2, rounds // 2)),
        ("c1_prose_long", 1, "long", 256, 2),
        (f"c{max_conc}_prose_short", max_conc, "short_prose", 384, rounds),
    ]
    if tier >= 2:
        cells.append((f"c{max_conc}_code_short", max_conc, "short_code", 384, rounds))
    cells.append(("pmu_replay_long", 1, "long", 8, 1))
    return cells


def _run_cell(name, conc, prompt_key, max_tokens, rounds, args, results_path):
    prompt = PROMPTS[prompt_key]
    rounds_out = []
    for r in range(rounds):
        def one(i):
            return _post_stream(args.url, args.model, prompt, max_tokens,
                                args.temperature, args.timeout)
        t0 = time.perf_counter()
        if conc == 1:
            res = [one(0)]
        else:
            with ThreadPoolExecutor(max_workers=conc) as ex:
                res = list(ex.map(one, range(conc)))
        wall_slowest = max(x["wall"] for x in res)
        total_toks = sum(x["completion_tokens"] or 0 for x in res)
        agg = round(total_toks / wall_slowest, 2) if wall_slowest > 0 else None
        rec = {
            "type": "cell", "cell": name, "concurrency": conc, "round": r,
            "aggregate_tok_s": agg, "wall_slowest_s": round(wall_slowest, 3),
            "streams": res,
        }
        rounds_out.append(rec)
        _record(rec, results_path)
        short = ", ".join(
            f"{x['completion_tokens']}tok/{x['wall']}s" for x in res)
        print(f"  [{name}] round {r}: agg={agg} tok/s  ({short})")
    rates = [r["aggregate_tok_s"] for r in rounds_out if r["aggregate_tok_s"]]
    return {
        "cell": name, "concurrency": conc, "prompt_key": prompt_key,
        "max_tokens": max_tokens, "rounds": rounds,
        "median": statistics.median(rates) if rates else None,
        "min": min(rates) if rates else None,
        "max": max(rates) if rates else None,
        "values": rates,
    }


def _record(obj: dict, path: str) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps(obj) + "\n")


def cmd_bench(args) -> int:
    out = args.out or datetime.now(timezone.utc).strftime("benchmarks/%Y%m%d-%H%M-") + args.label
    os.makedirs(out, exist_ok=True)
    # Node working trees must stay pristine — results belong OUTSIDE the repo
    # checkout (e.g. ~/benchmarks/ on the head).
    try:
        _t = subprocess.run(["git", "-C", out, "rev-parse", "--show-toplevel"],
                            capture_output=True, text=True, timeout=30)
        if _t.returncode == 0:
            print(f"WARN: --out is inside git repo {_t.stdout.strip()} — results will "
                  f"sit in the checkout; prefer ~/benchmarks/ to keep node trees pristine",
                  file=sys.stderr)
    except Exception:
        pass
    results_path = os.path.join(out, "results.jsonl")
    if os.path.exists(results_path):
        print(f"FATAL: {results_path} already exists — pick a new --out", file=sys.stderr)
        return 2

    meta = {
        "type": "meta", "label": args.label, "tier": args.tier,
        "model": args.model, "url": args.url, "container": args.container,
        "lane": args.lane, "rounds": args.rounds, "max_conc": args.max_conc,
        "temperature": args.temperature,
        "boot_age_minutes": _container_age_minutes(args.container) if args.container else None,
        "host": socket.gethostname(),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        meta["git_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=30).stdout.strip() or None
    except Exception:
        meta["git_commit"] = None
    _record(meta, results_path)
    print(f"[bench] label={args.label} tier={args.tier} out={out}")

    # ---- health gate ----
    try:
        with urllib.request.urlopen(args.url.rstrip("/") + "/health", timeout=30) as r:
            if r.status != 200:
                raise SystemExit(f"FATAL: /health returned {r.status}")
    except Exception as exc:
        print(f"FATAL: engine not healthy at {args.url}: {exc}", file=sys.stderr)
        return 2
    print("[bench] /health 200")

    # ---- tier-0 markers ----
    markers = {}
    if args.container:
        markers = _docker_log_markers(args.container)
        markers["type"] = "marker"
        markers["argv_has_async"] = _argv_has(args.container, "--async-scheduling")
        markers["argv_has_pmu"] = _argv_has(args.container, "--prefix-match-unit")
        _record(markers, results_path)
        print(f"[bench] kv: {markers.get('kv_cache_line')}")
        print(f"[bench] spec: {markers.get('spec_line')}")
        if markers.get("tracebacks"):
            print(f"FATAL: {markers['tracebacks']} tracebacks in boot log — fix before benching", file=sys.stderr)
            return 2
        if markers.get("kv_cache_line") is None:
            print("WARN: no 'GPU KV cache size' line found in tail OR full log", file=sys.stderr)

    # ---- warm-up (mandatory) ----
    print("[bench] warm-up (3 generations, temp 0)...")
    for prompt, mt in WARMUP:
        _post_stream(args.url, args.model, prompt, mt, args.temperature, args.timeout)

    # ---- acceptance counters: before ----
    m_before = _get_metrics(args.url)

    summaries = []
    for cell in _cell_list(args.tier, args.rounds, args.max_conc):
        summaries.append(_run_cell(*cell, args, results_path))
    # meminfo AFTER the long cells (KV-pin risk check)
    mem = _mem_available_gib()
    _record({"type": "meminfo", "mem_available_gib": mem}, results_path)
    print(f"[bench] MemAvailable: {mem} GiB")

    # ---- acceptance counters: after ----
    m_after = _get_metrics(args.url)
    acceptance = None
    dd = None
    if m_before["draft_tokens_total"] and m_after["draft_tokens_total"]:
        dd = m_after["draft_tokens_total"] - m_before["draft_tokens_total"]
        da = m_after["accepted_tokens_total"] - m_before["accepted_tokens_total"]
        if dd and dd > 0:
            acceptance = round(da / dd, 4)
    rec = {"type": "metrics", "acceptance": acceptance, "draft_delta": dd}
    _record(rec, results_path)
    print(f"[bench] spec-decode acceptance (delta): {acceptance}")

    # ---- summary.md ----
    lines = [f"# bench {args.label} — {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
             f"- tier {args.tier}, rounds {args.rounds}, max_conc {args.max_conc}, temp {args.temperature}",
             f"- model {args.model} @ {args.url} (container: {args.container or 'n/a'})",
             f"- KV: {markers.get('kv_cache_line')}" if args.container else "",
             f"- acceptance: {acceptance}", f"- MemAvailable: {mem} GiB", "",
             "| cell | conc | median tok/s | min–max | rounds |",
             "|---|---|---|---|---|"]
    for s in summaries:
        lines.append(f"| {s['cell']} | {s['concurrency']} | {s['median']} | "
                     f"{s['min']}–{s['max']} | {s['rounds']} |")
    with open(os.path.join(out, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # ---- PMU replay verdict ----
    replay_cached = None
    with open(results_path) as fh:
        for line in fh:
            obj = json.loads(line)
            if obj.get("type") == "cell" and obj.get("cell") == "pmu_replay_long":
                for st in obj["streams"]:
                    if st.get("prompt_tokens"):
                        replay_cached = (st.get("cached_tokens"), st["prompt_tokens"])
    if replay_cached:
        cached, ptoks = replay_cached
        expected = (ptoks // PMU_UNIT) * PMU_UNIT
        ok = cached is not None and cached >= 0.9 * expected
        print(f"[bench] PMU replay: cached_tokens={cached} / expected≈{expected} "
              f"-> {'PASS' if ok else 'FAIL'}")
        _record({"type": "pmu_replay", "cached_tokens": cached,
                 "prompt_tokens": ptoks, "expected_floor": expected, "pass": ok},
                results_path)

    print(f"[bench] done -> {out}")
    return 0


# --------------------------------------------------------------------------
# Compare
# --------------------------------------------------------------------------
def _load(outdir: str) -> dict:
    cells, markers, metrics, pmu, mem, meta = {}, None, None, None, None, None
    with open(os.path.join(outdir, "results.jsonl")) as fh:
        for line in fh:
            obj = json.loads(line)
            if obj.get("type") == "cell":
                cells.setdefault(obj["cell"], []).append(
                    (obj.get("round", 0), obj["aggregate_tok_s"]))
            elif obj.get("type") == "marker":
                markers = obj
            elif obj.get("type") == "metrics":
                metrics = obj
            elif obj.get("type") == "pmu_replay":
                pmu = obj
            elif obj.get("type") == "meminfo":
                mem = obj
            elif obj.get("type") == "meta":
                meta = obj
    return {"cells": cells, "markers": markers, "metrics": metrics,
            "pmu": pmu, "mem": mem, "meta": meta}


def _stable_values(pairs: list) -> list:
    """Post-warm-up rounds only (round 0 excluded when later rounds exist).

    The warm-up stage clears the engine's cold path, but each cell's round 0
    still runs colder than rounds 1+ (JIT batch shapes, PMU population). A
    cold round 0 inside an otherwise-separated range once masked a real 12%
    c4 regression — never let round 0 into a verdict.
    """
    vals = [v for r, v in pairs if r >= 1]
    return vals if vals else [v for _, v in pairs]


def cmd_compare(before_dir: str, after_dir: str) -> int:
    b, a = _load(before_dir), _load(after_dir)
    names = [n for n in b["cells"] if n in a["cells"]]
    rows, regressions = [], []
    for name in names:
        bv, av = sorted(_stable_values(b["cells"][name])), sorted(_stable_values(a["cells"][name]))
        bmed = round(statistics.median(bv), 2)
        amed = round(statistics.median(av), 2)
        delta = round(amed - bmed, 2)
        pct = round(100 * delta / bmed, 1) if bmed else None
        if name.startswith("c1"):
            verdict = "INFO (c1 is acceptance-bimodal — do not judge)"
        elif re.match(r"^c\d+_", name):
            overlap = av[0] <= bmed <= av[-1] or bv[0] <= amed <= bv[-1]
            no_overlap = av[-1] < bv[0] or bv[-1] < av[0]
            if no_overlap and amed < 0.93 * bmed:
                verdict = "REGRESSION"
                regressions.append(name)
            elif no_overlap and amed > 1.07 * bmed:
                verdict = "IMPROVEMENT"
            else:
                verdict = "NOISE"
        else:
            verdict = "INFO"
        rows.append((name, bmed, bv, amed, av, delta, pct, verdict))

    print(f"### bench compare: {os.path.basename(before_dir)} -> {os.path.basename(after_dir)}\n")
    print("(verdicts use post-warm-up rounds; round 0 of each cell is excluded)\n")
    print("| cell | before median (min–max) | after median (min–max) | delta | verdict |")
    print("|---|---|---|---|---|")
    for name, bmed, bv, amed, av, delta, pct, verdict in rows:
        print(f"| {name} | {bmed} ({bv[0]}–{bv[-1]}) | {amed} ({av[0]}–{av[-1]}) "
              f"| {delta:+} ({pct}%) | {verdict} |")

    # cross-run checks
    if b["metrics"] and a["metrics"] and b["metrics"].get("acceptance") is not None \
            and a["metrics"].get("acceptance") is not None:
        d = a["metrics"]["acceptance"] - b["metrics"]["acceptance"]
        v = "REGRESSION?" if abs(d) > 0.05 else "NOISE"
        print(f"\n- acceptance: {b['metrics']['acceptance']} -> {a['metrics']['acceptance']} (Δ{d:+.3f}) {v}")
        if v != "NOISE":
            regressions.append("acceptance")
    if b["pmu"] and a["pmu"]:
        for side, p in (("before", b["pmu"]), ("after", a["pmu"])):
            print(f"- pmu_replay[{side}]: cached={p['cached_tokens']} expected≈{p['expected_floor']} "
                  f"-> {'PASS' if p['pass'] else 'FAIL'}")
        if not a["pmu"]["pass"]:
            regressions.append("pmu_replay")
    if b["markers"] and a["markers"]:
        bk, ak = b["markers"].get("kv_cache_line"), a["markers"].get("kv_cache_line")
        same = "same" if bk == ak else "DIFFERENT"
        print(f"- KV pool: {same}\n  - before: {bk}\n  - after:  {ak}")
        if bk != ak:
            print("  NOTE: a pool change is EXPECTED if the update changed the KV pin/GMU; "
                  "unexpected if the diff didn't touch KV knobs — judge against the diff.")
    if b["mem"] and a["mem"] and b["mem"]["mem_available_gib"] and a["mem"]["mem_available_gib"]:
        bm, am = b["mem"]["mem_available_gib"], a["mem"]["mem_available_gib"]
        v = "WATCH (host headroom shrank >2 GiB)" if am < bm - 2.0 else "ok"
        print(f"- MemAvailable: {bm} GiB -> {am} GiB ({v})")

    # boot-state matching: a fresh boot vs a long-warm boot shifts decode by
    # more than the regression threshold — the bench is then invalid, not the
    # update. (This session's 32h-warm 'before' vs fresh 'after' pair needed a
    # third control boot to separate warm-up from a real regression.)
    ba = (b["meta"] or {}).get("boot_age_minutes")
    aa = (a["meta"] or {}).get("boot_age_minutes")
    boot_mismatch = None
    if ba is not None and aa is not None:
        print(f"- boot age: before {ba} min, after {aa} min")
        if (ba < 60) != (aa < 60):
            boot_mismatch = f"before {ba} min vs after {aa} min"
            print("  WARNING: boot states UNMATCHED (one side fresh, one long-warm) — "
                  "verdicts unreliable; re-bench both sides on matched boot states")

    print("\nVERDICT: " + ("REGRESSION — do not merge; investigate: " + ", ".join(regressions)
                          if regressions else "no regression detected (medians within noise bands)"))
    if boot_mismatch:
        print("NOTE: boot-state mismatch reported — treat the verdicts above as unreliable.")
    return 1 if regressions else 0


# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("bench", help="run the benchmark against a serving engine")
    pb.add_argument("--label", required=True, help="side label, e.g. before|after")
    pb.add_argument("--model", required=True, help="served model name (e.g. glm-5.3-flash)")
    pb.add_argument("--url", default="http://127.0.0.1:8000")
    pb.add_argument("--container", default=None,
                    help="container name for boot-marker/argv checks (head node only)")
    pb.add_argument("--lane", default=None, help="lane label for the record (e.g. mtp3)")
    pb.add_argument("--tier", type=int, default=1, choices=(0, 1, 2))
    pb.add_argument("--rounds", type=int, default=3)
    pb.add_argument("--max-conc", type=int, default=4)
    pb.add_argument("--temperature", type=float, default=0.0)
    pb.add_argument("--timeout", type=float, default=1200.0)
    pb.add_argument("--out", default=None, help="output dir (default benchmarks/<ts>-<label>)")
    pb.set_defaults(fn=cmd_bench)

    pc = sub.add_parser("compare", help="compare two bench runs")
    pc.add_argument("before_dir")
    pc.add_argument("after_dir")
    pc.set_defaults(fn=lambda a: cmd_compare(a.before_dir, a.after_dir))

    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
