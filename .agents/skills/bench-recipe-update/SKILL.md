---
name: bench-recipe-update
description: >-
  Use when a recipe config change on the 4x-switchless lanes needs a before/after
  benchmark before adoption — the change is benched as currently pinned versus the
  candidate config and must not regress. Triggers: "bench the recipe update",
  "before/after benchmark", "did the update regress anything", "A/B the configs",
  "run the regression bench", "verify the recipe change with numbers". Does NOT deploy
  or tear down lanes itself (bring-up-spark-recipe / LANE-SWITCHING.md), does NOT
  update recipe sources (update-recipe), and does NOT merge PRs.
---

# Bench a Recipe Update (before/after regression benchmark)

Produce a measured answer to one question: **did the candidate change make the lane
worse?** The bench runs the same fixed workload against the lane as currently pinned
and as pinned with the candidate change, and compares per-cell medians with
noise-aware verdicts.

## Scope boundary

This skill benchmarks only. Deploying/tearing down lanes between sides is
`bring-up-spark-recipe` + `LANE-SWITCHING.md`'s job; recipe changes are
`update-recipe`'s. Never merge/adopt as a result of a bench — posting the comparison
and the verdict is the stopping point.

## Core rules (non-negotiable)

1. **Rates come from real token counts** (`usage.completion_tokens`), never from
   streamed deltas (streamed deltas measure steps/s and under-report up to ~4×).
2. **Warm up before measuring.** A fresh boot is ~30% slower until a few hundred
   tokens of traffic pass; short calls do NOT clear it. The warm-up stage is
   mandatory, never skip it; never bench within ~30 min of a boot.
3. **Byte-identical prompts across sides.** Prompts are baked constants so both sides
   see exactly the same bytes — otherwise prefix-cache state differs and prefill
   numbers lie.
4. **Never judge on a single c1 sample.** c1 decode is acceptance luck and
   prompt-dependent (same engine, same day: prose 44 / code 77 / json 96 tok/s on the
   NVFP4 lane). c1 cells are informational; the cN aggregate cell is the discriminator.
5. **Verify the config actually took before benching** (container env / engine argv,
   KV pool line from the boot log) — a silently-inert env knob costs a whole boot.
6. **Match boot states across sides.** A long-warm boot and a fresh boot differ by
   more than the regression threshold. Prefer benching BOTH sides on fresh boots,
   each after the mandatory warm-up stage; if one side is a long-warm boot, re-bench
   the other on a matched fresh boot as a control, and record boot age.

## Tooling per lane

- **NVFP4 lane** (`glm-v53-flash-nvfp4-4x-noswitch/upstream/bench/`):
  `accept_probe` (c1 acceptance + ms/step + tok/s per prompt type), `conc_bench`
  (the concurrency ladder — the discriminator), `prefix_scan.py` (correctness gate —
  run it too; a bench on a corrupting boot is meaningless). Flags: `--base` WITHOUT
  `/v1`, `--model glm-5.3-flash`.
- **FP8 lane** (`bench_recipe.py`, in this skill's `scripts/`): deliver to the head's
  `$HOME` once per session (`scp .agents/skills/bench-recipe-update/scripts/bench_recipe.py
  spark-0f0b.shawndo.intra:~/`), run from the head with `--out` OUTSIDE the repo
  checkout (`~/benchmarks/`). **Known gotcha (T1, 2026-10-10): the boot-log gate
  counts the benign vLLM `usage_lib`/py-cpuinfo background-thread traceback that
  fires on every boot of this build and refuses to bench — a scratch copy whose gate
  whitelists ONLY that traceback (real tracebacks still block) is the sanctioned
  workaround; keep the whitelist minimal.**

## Procedure

1. **Scope the bench from the diff** — which knobs changed? Only one lane's knobs →
   bench that lane. Transport/NCCL changes → both lanes' ladders + the fabric
   preflight before each boot. KV-pin/memory changes → also record host
   `MemAvailable` and compare the KV pool line from the boot logs.
2. **BEFORE side:** the lane up with the current pinned config
   (`LANE-SWITCHING.md` flow; never edit files on the nodes — config travels through
   the repo + the derived-file mirror steps). Wait for READY + health 200 + correct
   model id + the expected `GPU KV cache size` line + zero real tracebacks. A Tier-0
   failure = STOP; a bench against a broken boot measures the break.
3. **AFTER side:** apply the candidate config through the repo flow, relaunch
   (all-ranks teardown, worker-first), verify the config took (rule 5), same boot
   state as the BEFORE side (rule 6), run the same commands.
4. **Compare + report:** per-cell medians, post-warm-up rounds only (exclude round 0
   — a cold round 0 once masked a real 12% regression as "noise"). Write a dated
   changelog block into the lane's `research.md`: what changed, before/after medians,
   verdict, gotchas hit. Post the comparison; do NOT merge or adopt.

## Interpretation cautions

- c1 is latency-bound on this ring; small c1 deltas are noise. The aggregate ladder
  cells (c4–c16) and TTFT percentiles carry the verdict.
- json/prefill-heavy cells are where transport changes show (the NVFP4 four-PF A/B:
  json +26%/+15% at c8/c16, c1 flat).
- No same-config historical baseline = directional evidence only (the FP8 4-PF bench
  caveat) — say so in the verdict instead of overclaiming.
