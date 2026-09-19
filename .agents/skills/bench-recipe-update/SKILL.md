---
name: bench-recipe-update
description: >-
  Use when a recipe update branch in spark-recipes needs a before/after regression
  benchmark before merge — the update is benched as pinned on `main` versus the
  update branch and must not regress. Triggers: "bench the recipe update",
  "before/after benchmark", "did the update regress anything", "A/B the branch
  against main", "run the regression bench", "verify the recipe change with
  numbers". Does NOT deploy or tear down the recipe itself (bring-up-spark-recipe
  does that), does NOT update recipe sources (update-recipe), and does NOT merge PRs.
---

# Bench a Recipe Update (before/after regression benchmark)

Produce a measured answer to one question: **did the update branch make the
recipe worse?** The bench runs the same fixed workload against the recipe as
pinned on `main` and as pinned on the update branch, and compares per-cell
medians with noise-aware verdicts.

## Scope boundary

This skill benchmarks only. Deploying/tearing down the recipe between sides is
the `bring-up-spark-recipe` skill's job; recipe changes are `update-recipe`'s.
Never merge the update PR as a result of a bench — posting the comparison and
the verdict is the stopping point.

## Core rules (non-negotiable)

1. **`stream:false`-equivalent counting: rates come from real token counts**
   (`usage.completion_tokens`), never from streamed deltas (streamed deltas
   measure steps/s and under-report up to ~4x). `bench_recipe.py` streams with
   `stream_options.include_usage` and reads the FINAL usage chunk — that is the
   valid pair (real count + wall clock) and also survives builds that omit
   `usage` on non-stream responses (GLM intel quirk).
2. **Warm up before measuring.** A fresh boot is ~30% slower until a few
   hundred tokens of traffic pass; short calls do NOT clear it. The script's
   warm-up stage is mandatory, never skip it.
3. **Byte-identical prompts across sides.** Prompts are baked constants in the
   script so `before` and `after` see exactly the same bytes — otherwise
   prefix-cache state (PMU128) differs and prefill numbers lie.
4. **Never judge on a single c1 sample.** c1 decode is bimodal acceptance luck
   (same config sampled 17–32 tok/s). c1 cells are informational; the cN
   aggregate cell is the discriminator.
5. **Verify the config actually took before benching** (engine argv via
   `/proc/1/cmdline`, KV pool line from the boot log) — the 2026-09-18 async
   A/B lost a whole boot to a silently-inert env knob.
6. **`drop_caches` ritual on BOTH nodes before every launch** (GB10 unified
   memory swap-wedges instead of OOMing); boots are then slow (cold shard
   reads) — that cost is accepted for reproducibility.
7. Worker (rank 1) starts FIRST, head ~30 s later; `docker compose down` on
   BOTH nodes between relaunches. (Full mechanics: `bring-up-spark-recipe`.)

## Procedure

### 0) Scope the bench from the diff

```bash
git diff main...<update-branch> -- <recipe>/ | stat  # which knobs changed?
```

- Only lane X's knobs changed → **Tier 1 on lane X, Tier 0 on the other lane.**
- KV-pin/GMU/max-seqs changes → also record host `MemAvailable` after the long
  cell (the script does this) and compare KV pool sizes from the boot logs.
- Both lanes touched, or a fragile knob (KV dtype, backend, block size) → run
  both lanes, consider Tier 2.

### 1) BEFORE side (main)

Ensure the recipe is up from `main` on both nodes (bring-up flow; do NOT edit
files on the nodes — land nothing, just checkout `main` + `git pull origin
main` there; branch protection means `main` is already what's deployed).
Then, from the head node:

```bash
cd ~/code/spark-recipes/<recipe>
python3 .agents/skills/bench-recipe-update/scripts/bench_recipe.py bench \
  --label before --model <served-name> --container <container-name> \
  --lane <lane-or-defaults> --out benchmarks/<YYYYMMDD>-before
```

Wait for `READY` + Tier-0 checks to pass (health 200, correct model id,
`GPU KV cache size` line matches the expected pool for this profile, zero
tracebacks). If a Tier-0 check fails, STOP — fix the deployment first; a bench
against a broken boot measures the break, not the update.

### 2) AFTER side (update branch)

Land the branch on the nodes via the normal git flow (PR merged OR user
explicitly says to use the branch), `docker compose down` both nodes,
`drop_caches` ritual, worker-first relaunch, then the same command with
`--label after --out benchmarks/<YYYYMMDD>-after`.

### 3) Compare + report

```bash
python3 .../bench_recipe.py compare benchmarks/<YYYYMMDD>-before benchmarks/<YYYYMMDD>-after
```

Read the verdicts with the interpretation table below. Write the results table
into the recipe's `research.md` as a dated changelog block (repo convention):
what changed, the before/after medians, the verdict, and any gotchas hit.
Post the comparison; do NOT merge.

## The cell matrix (Tier 1, ~15–20 min/boot)

| cell | shape | rounds | why |
|---|---|---|---|
| warmup | 3 × ~640-tok gens | — | clears JIT/cold path; never skipped |
| `c1_prose_short` | 1 stream, ~1K prompt, tg 512 | 3 | baseline decode; informational only |
| `c1_code_short` | 1 stream, ~1K code prompt, tg 512 | 3 | code drives spec-decode acceptance — the sensitive canary |
| `c1_prose_medium` | 1 stream, ~16K prompt, tg 256 | 2 | mid-depth prefill+decode blend |
| `c1_prose_long` | 1 stream, ~64K prompt, tg 256 | 2 | the production pain: big prefills / TTFT axis |
| `cN_prose_short` | N parallel, tg 384, aggregate = Σtokens/slowest-wall | 3 | **primary discriminator** (N = max concurrency, default 4) |
| `pmu_replay_long` | repeat the exact long prompt, tg 8 | 1 | prefix-cache regression check: `cached_tokens` ≈ floor(prompt/128)×128 |

Tier 0 = health + boot markers + argv check + warm-up + one medium PMU replay
+ acceptance counters (~5 min). Tier 2 = Tier 1 at 5 rounds, N=6, plus a
parallel code cell — only for fragile changes.

## Interpretation (baked into `compare`)

| cell class | regression if | notes |
|---|---|---|
| `cN_*` (aggregate, incl. long/medium depth cells) | after median < 0.93 × before median AND ranges don't overlap | observed same-config spread ≈ ±7 tok/s; overlap = noise. Rate-drop ⇔ wall-rise: a long-cell rate drop is a prefill/TTFT regression |
| `c1_*` | never auto-flagged | acceptance-bimodal; report only |
| `pmu_replay_long` | cached < 0.9 × floor(prompt_tokens/128)×128 | cache-mechanics regression |
| spec-decode acceptance | abs delta > 0.05 | from `/metrics` counters delta |
| KV pool (boot markers) | pool differs from expected/profile value | pin or GMU drift |
| `MemAvailable` (long cell) | host swaps or free < prior run's − 2 GiB | the KV-pin trap |

Verdict wording: `REGRESSION` / `IMPROVEMENT` / `NOISE` / `INFO`. A bench with
any `REGRESSION` means: do NOT merge; investigate before proceeding.

## Time budget

Scoped (one lane): 2 boots ≈ 60–90 min total. Both lanes: 4 boots ≈ 2–3 h.
Boots dominate (GLM intel ≈ 8–10 min cold with the ritual; DSv4 warm ≈ 6 min).

## Common mistakes

- Benchmarking right after boot without the warm-up stage.
- Judging a knob from c1 numbers (bimodal acceptance luck).
- Trusting streamed deltas for tok/s.
- Changing prompts between sides (invalidates the prefix-cache axis).
- Letting the model-name proxy target the wrong backend (`BACKEND_MODEL`) —
  bench the real served name on :8000.
- EOS truncation: rates must use actual `completion_tokens` (the script does);
  single short samples with early EOS are normal.
- One-off JIT lines mid-bench are benign; sustained JIT during serving is not.

## Supporting file

`scripts/bench_recipe.py` — self-contained (stdlib only), two subcommands:
`bench` (writes `meta.json`, `results.jsonl`, `summary.md` into `--out`) and
`compare` (joins two runs, prints a markdown table + verdicts). Runs on the
head node against `http://127.0.0.1:8000` by default.
