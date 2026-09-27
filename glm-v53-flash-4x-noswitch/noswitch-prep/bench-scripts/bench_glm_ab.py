#!/usr/bin/env python3
"""bench_glm_ab.py — one-knob A/B bench for glm-v53-flash-nvfp4-0rand.

Shape = the 2026-09-18 async-A/B lane (validated for this recipe; DFlash2
scales poorly at >=3 concurrency so c2 is capped at 2):
  warm-up  : 2 generations (~640 tok prose + short code) — never skipped
  c1       : 2 rounds, single stream, tg 1024
  c2       : 2 rounds, 2 parallel streams, tg 512 (aggregate = sum/max-wall)

Conventions: stream:false; rates from real usage.completion_tokens; rounds
ending early on EOS (completion_tokens < max_tokens) are recorded but EXCLUDED
from aggregates; spec-decode acceptance alpha from /metrics counter deltas
(snapshot taken after warm-up so drafter JIT is out); boot markers (KV pool,
spec config, boot time) from the head container log.
"""
import argparse, json, os, re, statistics, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

_SENTS = [
    "The compiler inlined the hot loop and unrolled it twice, but the profiler "
    "still blamed the branch predictor for the stall cycle in stage three.",
    "A synchronous rectifier on the secondary side cut conduction losses by "
    "eleven percent, though the gate-drive timing needed a 40 ns skew trim.",
    "She calibrated the interferometer at dawn, when the lab's vibration floor "
    "was lowest, and logged the fringe contrast for each of five alignments.",
    "The freight manifest listed turbine blades, a crate of optical flats, and "
    "two drums of insulating varnish bound for the assembly plant in Korea.",
    "When the river flooded in March, the gauge station upstream recorded a "
    "rise of 2.4 meters in under nine hours, the fastest since records began.",
    "His lecture on elliptic curves began with congruences of integer points, "
    "then moved to the group law, and finished with a sketch of the descent.",
    "The bakery's sourdough starter had been fed daily since 1987, surviving "
    "two relocations, a summer power cut, and one forgetful apprentice.",
    "Voting on the amendment closed at noon; the count favored the minority "
    "report by three votes, pending the proxy ballots that arrived later.",
    "Ice cores from the last glacial period show ash layers at 1,032 and "
    "1,047 meters, correlating with eruptions dated by argon isotopes.",
    "The compiler inlined the parser's dispatch table, and the interpreter "
    "benchmark improved by eighteen percent on the release build only.",
    "Coolant entered the exchanger at 12 degrees and left at 41, matching "
    "the model within measurement error once fouling was included.",
    "A green heron stalked the shallows for twenty minutes, then struck "
    "once, and swallowed the minnow head-first while still afloat.",
    "The manuscript's marginalia identify three scribes, of whom the second "
    "worked only on quires eight through eleven and favored a heavy hand.",
    "Delays in the tunnel boring were attributed to fault gouge, not to "
    "water inflow, contrary to what the earlier survey report claimed.",
    "The choir rehearsed the motet in sections, resolved the dotted "
    "rhythm at bar 47, and agreed on a tempo of quarter note equals 66.",
    "Radio telemetry from the tagged leopard showed a nightly range of "
    "eleven square kilometers, compressed during the dry season.",
    "The contract specified liquidated damages of one percent per week, "
    "capped at ten, with a cure period of thirty days after notice.",
    "Rain fell on the parade, so the flypast was cancelled and the "
    "marchers took a shortened route along the covered colonnade.",
    "In the greenhouse, basil planted in late February reached harvest "
    "height by mid-April, outpacing the direct-sown beds by two weeks.",
    "The negotiation hinged on warranty scope: the buyer wanted whole-life "
    "coverage, while the seller offered twelve months plus spares support.",
    "A fault in the timing chain tensioner produced a rattle at cold "
    "start, audible for roughly two seconds until oil pressure built.",
    "The museum's conservation lab x-rayed the panel painting and found "
    "an earlier composition beneath, rotated ninety degrees.",
    "Sales of electric cargo bikes doubled year over year, driven mostly "
    "by courier fleets replacing short-route vans in the city core.",
    "The tide table predicted a 4.1-meter low at 07:44, and the crew "
    "planned the sandbar crossing for the following morning's high.",
    "Hydrophones moored at 200 meters recorded beaked whale clicks in "
    "short bursts, each bout lasting under a minute, mostly after dusk.",
    "The audit found no discrepancies in petty cash but flagged two "
    "purchase orders signed by a delegate whose authority had lapsed.",
    "Bakers know that humidity changes the flour's effective hydration, "
    "so the recipe's water column carries a footnote for summer months.",
    "The bridge deck's expansion joints were replaced over four weekends, "
    "with single-lane closures and a signed detour through the valley.",
    "Observation of the variable star spanned three seasons, yielding a "
    "period of 41.7 days and an amplitude of 0.8 magnitudes in V band.",
    "The recipe calls for blooming the saffron in warm milk for ten "
    "minutes, then folding it through the rice in three additions.",
]

def _varied_prompt(approx_tokens, cpt=4.0):
    """Varied, non-repeating prose (~1K tokens); shuffled deterministically
    so no sentence repeats adjacently — avoids the degenerate echo bench
    (temp-0 repeated-paragraph prompts pin acceptance at ~1.0)."""
    import random
    rng = random.Random(42)
    out = []
    total = 0
    target_chars = approx_tokens * cpt
    while total < target_chars:
        batch = _SENTS[:]
        rng.shuffle(batch)
        out.extend(batch)
        total = sum(len(s) for s in out)
    return " ".join(out)

PROMPT = _varied_prompt(1000)          # ~1K-token varied prose, all boots
_DRAFT_RE = r"^vllm:spec_decode_num_draft_tokens_total(?:\{[^}]*\})?\s+([0-9.eE+]+)\s*$"
_ACCEPTED_RE = r"^vllm:spec_decode_num_accepted_tokens_total(?:\{[^}]*\})?\s+([0-9.eE+]+)\s*$"


def post(url, model, max_tokens, timeout):
    body = {"model": model, "prompt": PROMPT, "max_tokens": max_tokens,
            "temperature": 0.0, "stream": False}
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        obj = json.loads(r.read().decode("utf-8", "replace"))
    wall = time.perf_counter() - t0
    u = obj.get("usage") or {}
    ct = u.get("completion_tokens")
    if not ct:
        raise RuntimeError(f"no usage.completion_tokens in response: {list(obj)[:6]}")
    return {"completion_tokens": ct, "prompt_tokens": u.get("prompt_tokens"),
            "wall": round(wall, 3), "rate": round(ct / wall, 2)}


def metrics(url):
    with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=30) as r:
        t = r.read().decode("utf-8", "replace")
    d = sum(float(m) for m in re.findall(_DRAFT_RE, t, re.M))
    a = sum(float(m) for m in re.findall(_ACCEPTED_RE, t, re.M))
    return d, a


def container_markers(container):
    """KV pool line, spec config line, boot duration from the head log."""
    try:
        out = subprocess.run(["docker", "logs", container], capture_output=True,
                             text=True, timeout=300).stdout + subprocess.run(
            ["docker", "logs", container], capture_output=True, text=True,
            timeout=300).stderr
    except Exception as exc:
        return {"error": f"docker logs failed: {exc}"}
    def find(*needles):
        for line in out.splitlines():
            if all(n in line for n in needles):
                return line.strip()
        return None
    kv = find("GPU KV cache size")
    spec = find("speculative_config")
    ready = find("Application startup complete")
    started = ready_secs = None
    try:
        st = subprocess.run(["docker", "inspect", "-f", "{{.State.StartedAt}}", container],
                            capture_output=True, text=True, timeout=60).stdout.strip()
        started = datetime.fromisoformat(st.replace("Z", "+00:00"))
    except Exception:
        pass
    boot_secs = None
    if ready and started:
        m = re.match(r"^(\d{4}-\d{2}-\d{2}T[\d:.]+Z?)", ready)
        if m:
            ts = m.group(1)
            try:
                ts_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                boot_secs = round((ts_dt - started).total_seconds())
            except ValueError:
                pass
    return {"kv_line": kv, "spec_line": spec, "boot_seconds": boot_secs}


def record(obj, path):
    with open(path, "a") as fh:
        fh.write(json.dumps(obj) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--container", default=None)
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rp = os.path.join(args.out, "results.jsonl")
    if os.path.exists(rp):
        print(f"FATAL: {rp} exists — pick a new --out", file=sys.stderr)
        return 2

    def rec(o):
        record(o, rp)

    rec({"type": "meta", "label": args.label, "model": args.model, "url": args.url,
         "host": os.uname().nodename,
         "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})

    # health gate
    with urllib.request.urlopen(args.url.rstrip("/") + "/health", timeout=30) as r:
        if r.status != 200:
            print("FATAL: /health not 200", file=sys.stderr)
            return 2
    print("[bench] /health 200")

    # markers
    mk = container_markers(args.container) if args.container else {}
    mk["type"] = "marker"
    rec(mk)
    print(f"[bench] KV: {mk.get('kv_line')}")
    print(f"[bench] boot_seconds: {mk.get('boot_seconds')}")

    # warm-up (mandatory; also clears drafter JIT before alpha snapshot)
    print("[bench] warm-up (2 generations, temp 0)...")
    post(args.url, args.model, 640, args.timeout)
    post(args.url, args.model, 256, args.timeout)

    d0, a0 = metrics(args.url)

    # c1: 2 rounds single-stream tg1024
    c1 = []
    for rnd in range(2):
        x = post(args.url, args.model, 1024, args.timeout)
        eos_short = x["completion_tokens"] < 1024
        x.update({"type": "cell", "cell": "c1_tg1024", "round": rnd,
                  "eos_short": eos_short, "counted": not eos_short})
        rec(x)
        print(f"  [c1_tg1024] round {rnd}: {x['rate']} tok/s "
              f"({x['completion_tokens']} tok/{x['wall']}s{' EOS-short' if eos_short else ''})")
        if not eos_short:
            c1.append(x["rate"])

    # c2: 2 rounds of 2 parallel tg512
    c2 = []
    for rnd in range(2):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as ex:
            res = list(ex.map(lambda _: post(args.url, args.model, 512, args.timeout), range(2)))
        wall = max(x["wall"] for x in res)
        toks = sum(x["completion_tokens"] for x in res)
        agg = round(toks / wall, 2)
        eos_short = any(x["completion_tokens"] < 512 for x in res)
        rec({"type": "cell", "cell": "c2_2xtg512", "round": rnd, "aggregate_tok_s": agg,
             "wall_slowest_s": round(wall, 3), "streams": res,
             "eos_short": eos_short, "counted": not eos_short})
        parts = ", ".join("%dtok/%ss" % (x["completion_tokens"], x["wall"]) for x in res)
        print("  [c2_2xtg512] round %d: agg=%s tok/s (%s%s)" % (
            rnd, agg, parts, " EOS-short" if eos_short else ""))
        if not eos_short:
            c2.append(agg)

    d1, a1 = metrics(args.url)
    alpha = round((a1 - a0) / (d1 - d0), 4) if (d1 - d0) > 0 else None
    rec({"type": "metrics", "acceptance": alpha,
         "draft_delta": d1 - d0, "accepted_delta": a1 - a0})
    print(f"[bench] alpha (counter delta): {alpha}")

    summary = {
        "label": args.label,
        "kv_line": mk.get("kv_line"), "boot_seconds": mk.get("boot_seconds"),
        "c1_tok_s": c1, "c2_tok_s": c2,
        "c1_median": round(statistics.median(c1), 2) if c1 else None,
        "c2_median": round(statistics.median(c2), 2) if c2 else None,
        "alpha": alpha,
        "excluded_eos_short": True,
    }
    rec({"type": "summary", **summary})
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[bench] done -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
