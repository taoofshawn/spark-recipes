# research.md — glm-v53-flash-fp8-tp4-switchless

Audit trail and update playbook for this recipe. This is the dated changelog /
research file for the 4-node switchless ring; `README.md` stays the running ops doc
only. If you are running the `update-recipe` skill against this recipe, **read §2
first** — this recipe deliberately violates one of that skill's defaults, and §2 is
the authority on how.

## 1. Provenance

| what | value |
|---|---|
| upstream source | `github.com/jnardiello/GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless` |
| vendored pin | commit `080fe09` (E29 recipe era) |
| vendored on | 2026-09-26 (moved from `C:\Users\sdrew\code\glm-4x-noswitch`, `.git` deleted — **detached** from origin, adopted not tracked) |
| committed into branch | 2026-09-26, commit `9cd37a0` (alongside the `noswitch-prep/` reorg) |
| layout | `upstream/` = live recipe checkout (tp4ctl/deploy/verify run from here) · `noswitch-prep/` = this site's config + prep/ops scripts + `build-record.md` (build history) · `README.md` = ops doc |

## 2. Local modifications inside `upstream/` — DO NOT OVERWRITE on update

**Exception to the repo's usual rule.** For every *other* recipe, `upstream/` is a
pristine pinned mirror and the recipe dir is the customization layer. **This recipe
is different**: its `upstream/` is a *detached live checkout* that carries five
deliberate site modifications. When re-syncing against a newer jnardiello commit
(`git diff 080fe09..<new>` or re-vendoring), these files must be **kept
ours** — merge upstream's structural changes if any, but never take the upstream
version wholesale:

### 2.1 `upstream/scripts/node/bootstrap/versions.env` — SITE RE-PIN

The block marked `# SITE RE-PIN (shawndo 4x DGX Spark cluster, 2026-09-26)` plus the
site values:

- `KERNEL=6.17.0-1032-nvidia` (upstream may carry a different kernel pin for their
  cluster — e.g. the 7.0.0-1019 line this site explicitly avoids, forum 383023)
- `KERNEL_PKGS` / `KERNEL_PKGS_EXTRA` at the `.32` build
- `DRIVER=580.173.02`
- `OS_RELEASE=""` (intentionally empty: ranks 0-1 are 24.04.4, ranks 2-3 24.04.5)

Upstream updates may add *new* keys — adopt those. Never adopt their `KERNEL`,
`KERNEL_PKGS*`, `DRIVER`, or non-empty `OS_RELEASE` values.

### 2.2 `upstream/scripts/node/nccl/SHA256SUMS` — adopted rebuild

Single line: `afe5f48626284eae89988516e450c7f20cc303904ba4b7083b88aa3ca1e9b85f  libnccl.so.2`
— our on-site rebuild (2.30.7-1, sm_121, 61,581,280 bytes / 165 symbols), adopted
after the full-stack window on 2026-09-26 (see `build-record.md` §7c). NCCL builds
are **not bit-reproducible**: upstream's SHA256SUMS is the hash of *their* build and
will not match ours. Do not overwrite it as part of a routine update. Only change it
by deliberately re-running the candidate NCCL flow (`scripts/node/nccl/README.md`)
on our cluster — never by copying upstream's hash.

### 2.3 `upstream/scripts/node/nccl/build.sh` — SITE FIX (PATCH_FILE re-assert)

After `tp4_load_env` (~line 60), lines 61–65 re-assert
`PATCH_FILE="$HERE/nccl-v2.30.7-1-spark-switchless.patch"` because `cluster.env`
defines `PATCH_FILE` to the *vLLM indexer* rollback-lane path (expanded on the
nodes) and clobbers the vendored NCCL patch path, making the build die with
`vendored patch missing: $HOME/patches/…`. If an upstream update rewrites
`build.sh`, **re-apply the re-assert** after `tp4_load_env` before running any
build. (Consider upstream having fixed this properly a candidate to adopt — verify
their fix handles the cluster.env clobber the same way.)

### 2.4 `upstream/CHANGELOG.md` — site section

The dated section `## 2026-09-26 — shawndo 4x DGX Spark site` at the bottom is ours.
On update: keep our site section(s), take upstream's new entries above it.

### 2.5 Site path relocation (2026-09-27) — mechanical, applies across `scripts/`

This site keeps no runtime directories in `$HOME`: the deployed runtime lives at
`$HOME/.local/tp4` (upstream default: `~/tp4`), the patched NCCL at
`$HOME/.local/lib/nccl-patched` (upstream: `~/nccl-patched`), the vLLM/JIT cache at
`$HOME/.cache/tp4-vllm-cache` (upstream: `~/vllm-cache`), and the model/drafter
weights live inside the default HF cache
(`~/.cache/huggingface/hub/models--zai-org--GLM-5.3-Flash/snapshots/690b705278a3…` /
`~/.cache/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4ea…`,
upstream: flat `~/glm53-*-…` dirs). These are literal rewrites of the `~/tp4`,
`$HOME/patches/`, `$HOME/glm53-flash-fp8-zai`, `$HOME/glm53-dflash2-draft`,
`$HOME/nccl-patched` and `$HOME/vllm-cache` strings across `upstream/scripts/**`,
`upstream/cluster.env(.example)` and `noswitch-prep/cluster.env` (48 files). On an
upstream update, any NEW occurrences of the upstream paths in adopted files must be
rewritten the same way; `grep -rnE '\$HOME/tp4|~/tp4|glm53-flash-fp8-zai'` must
return nothing. `tp4ctl` is on the node PATH via `~/.local/bin/tp4ctl → ~/.local/tp4/tp4ctl`.

**Addendum (2026-09-29):** the relocation sed rewrote the literal path strings but missed
`deploy.sh`'s *relative* remote-path construction — `FILES`/`REMOTE_DIRS` destinations
(`tp4/…`, `patches/…`) are prefixed with `$HOME` at push/probe/verify time, so after the
relocation `deploy.sh --check` reported every managed file MISSING (they live under
`~/.local/tp4`, not `~/tp4`) and a full push would have recreated a stray `~/tp4` tree the
launcher never reads. Fixed with a single `remote_rel()` mapping (`tp4/* → .local/tp4/*`,
`patches/* → .local/lib/patches/*`) applied at the ssh/scp/chmod/bash-n call sites; the
launcher preflight (`-v` source existence) and `deploy.sh` now agree again. `--check` is
green against the deployed tree except for intentionally changed files (cluster.env, the
new §2.6 override).

### 2.6 `upstream/scripts/node/overrides/vllm/parser/glm47_moe.py` — required/named tool-choice enforcement (2026-09-29)

Site mod 6-7 (mod 6 = the override file, mod 7 = its `EXTRA_DOCKER_ENV` mount +
`deploy.sh` `REMOTE_DIRS` entry `tp4/overrides/vllm/parser`). The file is an exact
copy of the image's `vllm/parser/glm47_moe.py` (`ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache`,
vLLM `0.11.2.dev279+eldritch.final.fcc6141`, pristine copy kept at the workstation's
`%TEMP%\glm47_moe.py.orig`) plus a SITE PATCH: `STRUCTURAL_TAG_MODEL = "glm_4_7"` and an
`adjust_request`/`_apply_structural_tag` mirror of `DelegatingParser._apply_structural_tag`.

Why: the image's `parser/parser_manager.py:142-143` returns the raw engine class when both
parser roles resolve to `Glm47MoeParser`, discarding the registered tool parser's
`structural_tag_model` — so `tool_choice="required"`/named never applies the xgrammar
`glm_4_7` structural tag and silently behaves as `auto` (tool-eval-bench TC-45, −2 pts).
The patch applies `TagsWithSeparatorFormat(tags=…, at_least_one=True)` for required/named
requests behind the `VLLM_ENFORCE_STRICT_TOOL_CALLING` gate (defaults `True` in this build);
plain `auto` with non-strict tools is a no-op (registry returns `None`). Do NOT overwrite
this file wholesale on an image/upstream update — re-diff against the new image's
`glm47_moe.py` and re-apply the patch. Rollback: drop the `-v` pair from `EXTRA_DOCKER_ENV`
in `upstream/cluster.env(.example)` and `noswitch-prep/cluster.env`.

Mounted as
`-v $HOME/.local/tp4/overrides/vllm/parser/glm47_moe.py:/usr/local/lib/python3.12/dist-packages/vllm/parser/glm47_moe.py:ro`
in `upstream/cluster.env.example` (comment block above line 342), `upstream/cluster.env`
(gitignored site file) and `noswitch-prep/cluster.env`. The launcher preflight aborts a rank
whose `-v` source is missing, so deploy.sh must push the file before any restart.

**Addendum (2026-09-29, post-restart):** the first rollout exposed a second layer of the
same bug. `StreamingParserEngine._process_lex_tokens` runs in a "strict" mode once the
stream carries token ids: any terminal matched purely by TEXT whose name is in
`_token_id_terminal_names` (= the config's `token_id_terminals` keys) is demoted to plain
content, because a text occurrence "didn't arrive as the special-token id". Under the
forced xgrammar structural tag the model emits the tool tags as ordinary tokens (BPE
pieces / added-token ids), NOT as the 4 mapped special ids — so `<tool_call>` text-matches were
discarded, the state machine never left CONTENT, the arg tags became invalid, and the raw
envelope leaked into `content` on ~half of `tool_choice:"required"` responses (both
streaming and non-streaming; `finish_reason:"tool_calls"` masked it because serving sets
that unconditionally for required+stop). Fix in the override's `glm47_moe_config`:
`token_id_terminals` now keeps ONLY the THINK entries — THINK ids must stay because
`is_reasoning_end`/`extract_content_ids` resolve them from the config; TOOL_START/TOOL_END
are dropped so text-matched tool tags go through the normal state machine (adjust_request
forces `skip_special_tokens=False`, so the literal text is always present). Validated
in-container: non-stream with the actual generated id stream (the failing shape), streaming
with real prompt ids, thinking-off envelope-only, auto regression, and reasoning-end id
detection all pass.

## 3. Files outside `upstream/` an update must not touch

All of `noswitch-prep/` is site-owned, not upstream-owned:

| file | note on update |
|---|---|
| `noswitch-prep/cluster.env` | this site's active config (mirror of `upstream/cluster.env`). If the update changes `upstream/cluster.env.example` (new keys, new recipe E-number), regenerate `upstream/cluster.env` with `noswitch-prep/scripts/make-cluster-env.sh` and re-apply the site diffs, then mirror into `noswitch-prep/cluster.env`. Watch for: new `*_BY_RANK` arrays, changed `EXTRA_DOCKER_ENV`, TP4_ENV/recipe bumps — an upstream recipe bump changes the served identity and needs a full deploy+gate window, not a file edit. |
| `noswitch-prep/versions.env` | re-pinned copy consumed by the site docs; keep consistent with §2.1 |
| `noswitch-prep/sircl/rank{0-3}.env` + `SHA256SUMS` | generated private site files (mirrors of upstream-ignored `sircl-site-files.sh` output); only regenerate via the upstream tooling, never hand-edit |
| `noswitch-prep/preflight-report.json` | point-in-time build evidence; never regenerate during an update |
| `noswitch-prep/scripts/*` (9 helpers) | site build/ops helpers, written for this site; independent of upstream |
| `noswitch-prep/build-record.md` | closed historical record of the 2026-09-26 build; do not rewrite — add update entries below instead |

## 4. Update procedure (the `update-recipe` loop applied to this recipe)

1. **Snapshot**: note the pin (`080fe09`), the 5 mods (§2), current branch
   (`glm-v53-flash-4x-noswitch` — stay on it; this recipe's work never merges to
   `main` without the user), and the served identity (E29, `check-f0.py` default).
2. **Sources**: jnardiello repo commits/PRs/issues; NVIDIA forum (categories 721/723)
   with `glm`/`5.3`/`tp4`/`switchless` terms + general sweep since the last review
   date (this file's changelog dates are the last-review markers).
3. **Diff**: `git ls-remote` the upstream repo; diff `080fe09..HEAD`. Classify each
   change: adoptable (new knobs, fixes), inapplicable (their cluster's site values),
   or conflicting with §2/§3 (keep ours) — §2.5 paths included: re-apply the path
   rewrite to any newly adopted file referencing the old locations.
4. **Adopt surgically**: apply upstream changes to `upstream/` only where they are
   genuinely upstream (code, docs, third_party, reference recipes). Re-verify the
   4 mods survived. Recipe-**site** changes belong in `noswitch-prep/` or this
   branch's docs, not in `upstream/` beyond the §2 exceptions.
5. **Update the pin**: record the new jnardiello commit in this file's changelog
   (§5) and in `upstream/VENDORED-AT.md`.
6. **Validate offline**: `./scripts/check.sh` from `upstream/` on the workstation
   (WSL; needs `pip install Jinja2==3.1.6`). It is an offline check — no GPU, no
   cluster probing.
7. **Commit + push the branch**. Stop there (per `update-recipe` scope): do NOT
   deploy, restart, or bring anything up — that is `bring-up-spark-recipe` /
   explicit ops work, and the running cluster is live.
8. **Cluster rollout, when authorized separately**: `git pull` on the workstation
   and spark-0f0b clones → re-deploy from `upstream/` via `deploy.sh` + full
   `tp4ctl restart` (never single-rank) → `/health` 200 → both functional gates
   within 2 min → `check-f0.py`. Anything beyond that (kernel, NCCL, image) follows
   its own documented flow — see `build-record.md` and upstream docs.

## 5. Changelog

- **2026-10-12 — FIX-4 record re-pin VERIFIED against the live lane; check-f0 still FAILs
  on two pre-existing validator-expectation classes (neither is lane drift) — owner
  decisions parked (handoff 06 §FIX-4 finisher).**
  - Record re-pin verified: all 9 `container_file_sha256` entries in
    `docs/operational-identities/2026-09-30-e36-lm-head.json` — including the three
    re-pinned sources (gpu_worker `ae2bdaf5…`, scheduler `c5d8b99f…`, model_runner
    `598f9fd7…`) — MATCH the live containers on all four ranks, and the three
    `runtime_sources` hashes match the repo working tree (0.85-KV recipe pin
    `68d18652…` == fresh sha256 of `upstream/cluster.env.example`). The stale-hash
    mismatch row FIX-4 targeted is GONE.
  - The full check (`check-f0.py` operational-identity mode, 2026-10-12 11:24 UTC, lane
    serving 13 h) still returns FAIL with 8 problems: per rank, "patched NCCL not
    loaded by GPU worker" + "runtime file …/vllm/v1/engine/core.py". Diagnosed
    read-only (report: `.research/scratch/f0-20261012/report-check-f0-fix4-finisher.json`):
    1. **NCCL row = validator path-shape artifact.** The GPU worker (pid 3227213,
       /proc/<pid>/maps) HAS the patched lib loaded — as the symlink's RESOLVED path
       `/opt/patched-nccl/libnccl.so.2.30.7` (`ls -la /opt/patched-nccl/`:
       `libnccl.so.2 -> libnccl.so.2.30.7`). The probe accepts only a maps line ending
       with the literal symlink path `/opt/patched-nccl/libnccl.so.2`, which the kernel
       never reports. The stock wheel NCCL also mapped is normal (patched preload takes
       precedence). Lane healthy; validator too literal.
    2. **core.py row = stale frozen expectation, NOT a stale lane.** Live core.py
       `78063640…` == deployed `~/.local/tp4/experiments/e03/end-drain/core.py` == repo
       working tree (changed 2026-10-11 by the GRAMMAR-DIAG/withdrawal commits
       d8a5b2b/b8e19d2/dc0036e/2fc9478 — the same event family that stale-dated the three
       FIX-4 sources). The check's frozen value `3a9a038f…` lives in the E31 baseline's
       `system.operational_identity.container_file_sha256` (pre-dates those commits) and
       is NOT overridable by the identity record (`core.py` is outside the script's
       `allowed_override_targets`; FIX-4 scope explicitly excluded baseline items).
  - Both fixes are owner decisions, deliberately NOT applied here: (a) a one-line
    check-f0.py NCCL probe relaxation (accept the resolved versioned path), (b) an E31
    baseline core.py re-pin which CASCADES — `expected_operational` verifies the
    baseline file hash against the record's `performance_baseline.sha256`, so the
    baseline edit forces a record re-pin of that hash too. Flagging both instead of
    re-staling the frozen receipt or editing a validator to pass.
  - Lane untouched throughout (read-only SSH + HTTP; `glm53_fp8_dflash_tp4` serving).

- **2026-10-12 — Proxy `client_body_buffer_size 1m` deployed (handoff 02, cosmetic
  follow-up to the 10-11 TTFT attribution): the per-request "client request body
  buffered to temporary file" warn is silenced for bodies <=1 MB (nginx default buffer
  is 16 KB, so every ~100K-token agent body spilled). Zero TTFT intent — the 10-11
  entry exonerated the spill (<10 ms worst case, 6.3 GB/s temp dir); this is hygiene
  only. Bodies >1 MB (up to `client_max_body_size 64m`) still spill — expected,
  harmless.**
  - Change: one directive + comment in `model-name-proxy/nginx.conf.template` server
    block (after `client_max_body_size 64m;`); nothing else touched. Landed via topic
    branch `fp8-proxy-body-buffer-20261011` (commit `a5390b7`), PR opened/merged by the
    owner. Lane untouched throughout — no `tp4ctl`, no lane container restarts; :8000
    never blipped.
  - Deploy: owner-agreed quiet moment; proxy container recreated from the head node
    (`docker compose --env-file .env up -d --build`), created 2026-10-12T00:34:34Z,
    status `Up (healthy)`. Container recreate cuts in-flight proxied :4000 requests —
    window was quiet, nothing dropped.
  - Verification: `:4000/health` → 200; `/v1/models` → `spark-llm`; rendered config
    inside the container shows `client_body_buffer_size 1m;` (nginx.conf line 49).
    Probe: one ~100K-token body (625 KB, <1 MB) through `:4000/v1/chat/completions` →
    HTTP 200, completed normally (53 s wall, thinking model, unremarkable); proxy logs
    before/after the probe: **0** "temporary file"/"buffered" warn lines — the warn
    that previously fired on every such body is gone.

- **2026-10-11 (late) — TTFT queue-wait term measured DIRECTLY (handoff 01): 6-concurrent × 3 rounds of ~91K-token (tokenizer) prose probes, owner-agreed window. Verdict: the admission layer never queued (cap 6 == burst 6) — the wait term lives INSIDE the engine scheduler, which first-schedules big prefills one-at-a-time at ~40 s/position; measured worst server TTFT 240 s ≈ k×40 + 43, i.e. the k×(15–25 s) arithmetic UNDER-predicts ~1.7–2×. Stand pat: documented cost, no fix.**
  - Method / constraints: HTTP request load ONLY (no GPU-allocating probe — the OOM
    watch item stood), direct `:8000` (proxy latency excluded by design), lane UP
    throughout, zero restarts / zero config changes / zero engine-knob tuning. Bodies
    ~466 KB each, UNIQUE per request (round/request-indexed rotation + nonce lines) so
    prefix caching could not collapse the prefill term; `max_tokens=256` + streaming,
    client-side t0/first-byte/e2e stamped per request. Boot 21:55 UTC, probe 23:44–
    23:57 UTC → warm (1 h 49 m). Raw artifacts: `.research/ttft-burst-20261011/`
    (`probe.py`, `probe-results.json`, `vllm-window.log` full head log 23:44–23:57,
    `join_burst.py`, `decomposition.json` — untracked scratch).
  - Measured per-request decomposition (server-side, same definitions as the 10-11
    serial baseline; all 18/18 requests paired to ids by last-exec ≈ client e2e,
    admitted_total contiguous per round — 189–194 / 196–201 / 202–207 — so the pairing
    is exact; 26 window admits total = 18 probe + 8 background agent turns BETWEEN
    rounds):
    | term | med | max | reading |
    |---|---|---|---|
    | arrival→admit | ~0 | ~0 | bounded admission NEVER queued — cap 6 == burst 6, `active: 6` reached every round, `rejected_full=0`, `cancelled=0` (negative ~1–2.5 s = workstation-vs-node clock skew) |
    | admit→first-exec | 111 s | 194 s | **the wait term**: engine first-schedules the 6 big prefills ONE AT A TIME, ~0/31/71/111/152/190 s stagger per round (~40 s/position) |
    | first-exec→decode-start | 44 s | 46 s | own prefill wall-clock: ~91 K tokens ≈ 11 s of compute (8192-token chunks, ~9 K tok/s instantaneous) spread over a constant ~43 s by decode interleaving + stalls |
    | **server TTFT (admit→decode-start)** | **154 s** | **240 s** | position-k pattern ≈ k×40 + 43 s, all three rounds reproduced it (r0/r1/r2 worst: 233/234/240 s) |
    | client e2e | 176 s | 244 s | ≈ server TTFT + ~20 s decode; client first-byte ≈ server decode-start within ~2 s (cross-validation) |
  - Engine-tick economics during the burst: `run=1–2, wait=3–5` with all 6 admitted —
    the scheduler carries one prefilling + one decoding request, not 6 running. Exactly
    6 pp-burst ticks per round (~90 K prompt tokens each at ~9 K tok/s instantaneous);
    aggregate prefill ≈ 545 K tokens / 240 s round ≈ **2.3 K tok/s** vs the 6.8 K tok/s
    single-stream serial baseline. Barrier-form ticks (`pp=0 tg=0`, ~10 s each):
    3/4/1 per round — present but secondary; the dominant cost is the one-at-a-time
    big-prefill admission, not barriers alone. KV peaked at 26.1% of the 1.19 M pool —
    never the constraint, as designed.
  - **Verdict vs the 10-11 arithmetic:** direction confirmed (queue/scheduling wait
    under burst is the whole tail; proxy untouched, direct-:8000 numbers match the
    through-proxy 09-28 client band), magnitude UNDER-predicted ~1.7–2×: k×(15–25 s)
    + own prefill predicted 90–140 s at k=5, measured 240 s. The per-position
    coefficient under 6-active burst is ~40 s (serialized prefill admission + decode
    interleaving + barrier ticks), not 15–25 s. The 09-28 80–160 s agent-tail band
    corresponds to k≈1–3 — confirmed; deeper burst positions extend to ~240 s.
  - **Recommendation: (a) STAND PAT — documented cost, no fix.** The tail is the price
    of 6-active × 262K-context on this latency-bound ring, the engine behaves exactly
    as its validated E35/E31 stack is designed to (one big prefill at a time, bounded
    admission), and no error/rejection/OOM occurred. Any mitigation (e.g. admitting >1
    concurrent big prefill) touches the lane-invariant scheduler stack, is unmeasured
    risk, and would need its own bench-recipe-update session with a before/after A/B —
    the measurable targets would be the ~40 s/position stagger and the 43 s prefill
    wall (vs 15 s serial). NO branch, NO plan commit for (b): nothing applied.
  - Lane health post-probe: `/health` 200, waiting=0, running=0, zero NCCL WARNs /
    transport anomalies / OOM entries in the window; background agent traffic continued
    throughout and after (admitted_total 180→212 across the window).

- **2026-10-11 — TTFT-stall attribution (handoff 04): the 09-28 80–160 s agent-tail is ENGINE-side — bounded-admission queueing under burst concurrency + ~100K-token chunked prefill + E31 barrier ticks. The proxy's "buffered to temporary file" is a same-second passthrough marker, exonerated with hard numbers. No fix prepared (proxy is not the cause).**
  - Scope note: the 09-28 window's node logs and client logs are gone (both lane
    containers recreated at today's 21:55/22:14 UTC boot; omp client logs start 10-09),
    so the historical tail requests cannot be reconstructed per-request. Attribution is
    by live measurement of the SAME traffic class (the 2026-10-11 session's own agent
    turns, ~100K-token bodies through the model-name proxy) plus bounding arithmetic —
    all passive SSH reads, zero restarts, lane untouched.
  - Method: joined three server-side per-request signals on the head —
    `TP4_ADMISSION_STATE` admit/acquire/release (arrival + slot), `GRAMMAR-DIAG-EXEC`
    per-step dispatch (first appearance of a chatcmpl id = first schedule; first
    `spec_lens>0` = decode start ≈ first token), and 10-s Engine ticks; proxy warn
    events paired 1:1 by order with admit events (counts matched exactly, 112=112).
    Scratch: `.research/ttft-20261011/` (parser + per-request CSV, untracked).
  - **Proxy EXONERATED (the handoff's leading hypothesis, killed):** 112/112 requests
    were buffered to a temp file — `client_body_buffer_size` is unset in the proxy
    config (16 KB default) so every agent body exceeds it — but the warn→admit gap is
    **0 s median AND 0 s max over all 112 requests**: vLLM ingests the spilled body the
    same second it is written. With `proxy_request_buffering off` the temp file is
    spillover under upstream backpressure, not a staging stall. Temp-dir I/O measured
    **6.3 GB/s** (container overlay, 3.7 T disk 37% used) → worst-case 64 MB body
    (`client_max_body_size`) ≈ 10 ms. A stage bounded at <1 s cannot produce 80–160 s
    TTFTs. The warn is a MARKER of large turn bodies, not a cause of delay — and it is
    a useful one: it fires 1:1 with big-body arrivals.
  - **Engine-side serial ceiling (this session was fully serial: max active=1, max
    queued=1, queue wait ≈ 0 s for everyone):** server TTFT (admit→decode-start) mean
    8.7 s / med 6.5 s / p90 20 s / **max 25 s**; 74/112 ≤10 s, 38 in 10–30 s, 0 >30 s.
    100% of TTFT is first-schedule→decode-start (prefill): ~100K-token turns (KV
    8–14% of the 1.19M-token pool) prefill at ~6.8K tok/s (8192-token chunks) ≈ 15 s raw.
  - **Barrier amplifier quantified:** 28/175 ticks (16%) with a request in-system were
    zero-progress (`pp=0 tg=0 run=1`), with 20 E31 KV-transfer `wait` events
    (`held_ranks=[0,1,2,3]`) — each adds a ~10 s tick to a turn's prefill window. The
    known self-recovering between-turn barrier, now priced at +10–20 s on affected turns.
  - **80–160 s band arithmetic (the 09-28 bursts):** queueing behind bounded admission
    is the only term with the magnitude. At 6 active (`MAX_NUM_SEQS=6` /
    `TP4_ADMISSION_MAX_ACTIVE=6`), a request arriving behind k big turns pays
    k × (15–25 s prefill+barrier) ≈ 45–150 s, + its own 15 s prefill → the 80–160 s
    band for k≈3–5. Cross-check: mean server TTFT 8.7 s on this serial session vs
    18.9 s client-side on 09-28 is exactly what added burst queue time predicts.
  - **Verdict:** 80–160 s tail = engine queueing under burst concurrency (bounded
    admission working as designed) + ~100K-token chunked prefill + E31 barrier ticks;
    proxy contributes <1 s. Not a fault — the price of 6-active × 262K-context on this
    latency-bound ring.
  - No fix prepared (handoff step 4 is conditional on the proxy being the cause — it is
    not). Optional cosmetic-only tuning for a future owner window (NOT applied, NOT
    pushed, zero expected TTFT effect): `client_body_buffer_size 1m;` in the proxy
    server block to keep typical agent bodies (≤1 MB) in memory and silence the warn
    spam. ~~Open item for a future agreed window: a 6-concurrent ~100K-probe burst would
    measure the queue-wait term directly; skipped per the passive-observation
    constraint.~~ **CLOSED 2026-10-11 (late): measured directly — see the burst-probe
    entry at the top of this changelog.**

- **2026-10-11 — SparkCache capacity gauge VERDICT: `healthy=no used=0.0/0.0GiB` was the gauge truthfully reporting "no capacity policy configured" on the pre-E31-MB boot — cache data path was fine throughout; gauge live and healthy since the 09-30 capacity rollout (watch item closed).**
  - Mechanism (source-verified in the running image, read-only): the worker emits
    `report["capacity"]` only when `self._capacity_policy.enabled`
    (`spark_context_cache_connector.py:6432` in the deployed ram-budget copy;
    `CapacityPolicy.enabled` = `max_bytes > 0 or ttl_seconds > 0`,
    `persistent_context_cache/cache_manifest.py:443`). The engine-side gauge
    (`SparkCacheStats.reduce` → `format_log_lines`) sums the `capacity` dicts across rank
    reports and only sets `sparkcache_used_gib/limit_gib/healthy` when at least one rank
    reported one — with no capacity dict those keys default to `0.0/0.0` and
    `healthy=no`, while the separate `publication` field still populates the
    publications/writes counters. So `healthy=no used=0.0/0.0GiB` alongside
    `publications 460.9 GiB failed=0B` is exactly the signature of "policy disabled,
    data path healthy" — the gauge reads a different source than the publisher by
    construction, not by breakage.
  - Dating: the 09-28-observed boot predated the E31-MB rollout —
    `SPARK_CONTEXT_CACHE_MAX_BYTES` first appeared in the site mirror at `b51e8b6`
    (2026-09-30 23:48 ET); before that the env defaulted to `"0"`
    (`spark_context_cache_config.py:608`), i.e. the unbounded store E31-MB was written
    to fix (1.7 TB/node at audit). The gauge had no data source by design.
  - Current state (verified live 2026-10-11, read-only SSH only; container created
    21:55Z): every emission `healthy=yes used=703.6/800.0GiB` (4 ranks × 200 GiB
    E31-MB cap), publications `count=212 payload=54.2GiB`, writes `failed=0B`, deployed
    connector sha256 `aa046965…` == pinned `SPARKCACHE_CONNECTOR_SHA256`, deployed
    `~/.local/tp4/sparkcache/kv-transfer-config.json` leaves the capacity keys unset so
    the env applies as documented, and External prefix cache hit rate 24.8% — SparkCache
    restores are actively being served.
  - Deferral correlation (the 09-28 open question): live burst capture shows
    `Waiting: 1 reqs, Deferred: 1 reqs` at run≤2 — the Deferred counter is the E31
    deferred-admission path (KV transfer in progress), by design and unrelated to the
    gauge; no fault indicated.
  - Verdict: cosmetic AND already resolved — not a gauge bug, not a cache fault. The
    unhealthy readings were the designed "unwired" state of a capacity policy that did
    not exist pre-E31-MB; the 09-30 refresh wired it. No fix, no config change, no
    restart. Watch item closed.

- **2026-10-11 — py-spy installed host-side on all 4 nodes; live stack capture of the containerized vLLM workers VERIFIED (runbook gap from 09-27/09-28 closed).**
  - **What was done:** py-spy 0.4.2 installed at `~/.local/bin/py-spy` on every node's HOST
    (aarch64). Install path differs by node: `uv tool install py-spy` on spark-0f0b and
    spark-6d14 (uv present); on spark-6d90 and spark-6d24 (no uv) the manylinux_2_17
    aarch64 wheel was downloaded and the binary extracted with the node's python3:
    `curl -sL -o /tmp/pyspy.whl https://github.com/benfred/py-spy/releases/download/v0.4.2/py_spy-0.4.2-py2.py3-none-manylinux_2_17_aarch64.manylinux2014_aarch64.whl`
    then `python3 -c "import zipfile; zipfile.ZipFile('/tmp/pyspy.whl').extract('py_spy-0.4.2.data/scripts/py-spy', '.')"`
    + `install -m 755 py_spy-0.4.2.data/scripts/py-spy ~/.local/bin/py-spy`. Plain
    `pip install --user` is BLOCKED by PEP 668 (externally-managed env). Host Python is
    3.12.3 everywhere — same major version as the container's `/usr/bin/python3` (3.12),
    which is why host-side capture resolves container frames correctly.
  - **Verified live (lane serving, `/health` 200, zero disruption):** host-side
    `sudo ~/.local/bin/py-spy dump --pid <pid>` successfully attached to and dumped ALL
    rank processes from the host — head: `VLLM::Worker_TP0` + `VLLM::EngineCore`;
    spark-6d14: `Worker_TP1`; spark-6d90: `Worker_TP2`; spark-6d24: `Worker_TP3`.
    `sudo -n` works (NOPASSWD). Proof stack (Worker_TP0 idle-busy state, 2026-10-11):
    MainThread in `sched_yield (vllm/distributed/utils.py:48)` ← `wait
    (shm_broadcast.py:204)` ← `acquire_read (shm_broadcast.py:794)` ← `dequeue
    (shm_broadcast.py:907)` ← `worker_busy_loop (multiproc_executor.py:1033)` ←
    `worker_main (multiproc_executor.py:935)` — with the expected side threads
    (WorkerAsyncOutputCopy, DeathPipeMonitor, tqdm_monitor, spark-cache-capacity). Worker
    TP1–3 idle shape: MainThread blocked in zmq `poll` ← `recv (shm_broadcast.py:931)` ←
    `dequeue (shm_broadcast.py:922)` ← `worker_busy_loop`.
  - **The procedure for next wedge (capture BEFORE any restart):**
    1. On each node, find the worker pids: `docker top glm53_fp8_dflash_tp4 -eo pid,args
       | grep 'VLLM::'` (pids are per-boot — never reuse old ones).
    2. On the head, dump EngineCore + Worker_TP0; on each worker node, dump that node's
       `VLLM::Worker_TPr`: `sudo ~/.local/bin/py-spy dump --pid <pid>` (run from the node
       the process lives on; py-spy is at `~/.local/bin/py-spy`, which is NOT on the
       default SSH PATH — use the full path).
    3. A wedged rank will NOT look like the idle shapes above — expect a stuck frame in
       `shm_broadcast` (`acquire_read`/`wait`) with no progress across repeated dumps
       seconds apart. Take 2–3 dumps a few seconds apart per rank and save all of them.
    4. Only then `tp4ctl restart`. SIGQUIT (to a worker pid) remains the last resort if
       py-spy somehow fails.
  - **Constraints found:** container processes are root-owned and `yama/ptrace_scope=1`,
    so non-sudo capture fails with `Permission Denied: Try running again with elevated
    permissions` — `sudo` is REQUIRED (no CAP_SYS_PTRACE image change needed for
    host-side capture). The container itself still has no py-spy and does not need one:
    host capture works across the container boundary. Do NOT exec into the container to
    run py-spy — it is not there. Non-disruptive: no signals sent, lane kept serving.

- **2026-10-11 — OOM crash of the running lane + restart (no lane/config change).**
  - **What happened:** while the prior agent session ran its E20/GRAMMAR diagnostic probe
    campaign (`/tmp/probe*.py` on the head), `probe45.py` deliberately near-exhausted VRAM
    ("OO post-pressure" test: `torch.empty(64Mi×…, device='cuda')` in a loop until the CUDA
    allocation fails, then a conditional-node graph capture). On GB10, GPU memory is unified
    host LPDDR; with the model's ~101.6 GiB CUDA footprint resident there was no headroom.
    Timeline (UTC): 19:29:55 + 19:30:07Z earlier probe-container iterations died; 19:34:21 →
    19:36:55Z rank-0 worker's 1 Hz `E20_MEMORY_PROBE` thread stalled; 19:36:27Z kernel
    `docker invoked oom-killer` `CONSTRAINT_NONE` (host-wide); 19:36:47–19:37:02Z NVRM
    `NV_ERR_NO_MEMORY` @ `_memdescAllocInternal`/`system_mem.c`; 19:36:55Z the OOM killer
    killed `vllm` (pid 2625577) among pipewire/dbus/wireplumber; 19:36:58Z
    `glm53_fp8_dflash_tp4` died `OOMKilled=true` exit 137. Host had `Memory=0` (no cgroup
    limit); GPU allocator itself was healthy to the end (`num_ooms=0`).
  - **Root cause:** deliberate VRAM exhaustion by the probe on a memory-unified device —
    self-inflicted by the probe campaign, NOT a lane/engine/fabric fault.
  - **Restart:** `verify-fabric-p2.sh` ALL CHECKS PASS → `tp4ctl down` (all 4 ranks, hung
    workers 1–3 cleared) → `tp4ctl up`. Clean boot: 62/62 shards loaded, zero NCCL WARN,
    `/health` 200 at ~20:05Z, `/v1/models` `glm-5.3-flash` `max_model_len` 262144.
  - **Gates:** corruption probe exact ("echo test 🚀 海函数 ✅", 0× U+FFFD); forced
    `tool_choice:"required"` returned a structured `get_weather{"city":"Tokyo"}` call
    (thinking-on + required path confirmed working post-fix). No bench run — cold-boot
    JIT warm-up rule (~30 min) applies before any performance numbers.
  - **Watch item:** the probe scripts are part of the live GRAMMAR-DIAG/E20 instrumentation
    still compiled into the deployed overrides — any future probe that allocates GPU memory
    must run against a torn-down lane or stay strictly within a pre-reserved ballast budget;
    a repeat of probe45-style near-exhaustion while serving WILL kill the lane the same way.

- **2026-10-11 — Dual-lib NCCL SHA manifest: check-f0.py / verify-node.sh gate repaired
  (VENDORED-AT mod 12; closes the T1 entry's "check-f0.py / verify-node.sh FAIL — explained,
  expected" follow-up).**
  - **Mechanism:** after the 4-PF adoption `NCCL_DIR` points at `~/.local/lib/nccl-patched-4pf`
    (sha `8de68def…`) while the single legacy line in `scripts/node/nccl/SHA256SUMS` pinned
    `afe5f486…` — so every healthy 4-PF boot failed exactly the NCCL rows: "patched NCCL sha
    FAIL 8de68def… expected afe5486…" ×4 (verify-node.sh:656) + check-f0.py's
    "patched NCCL not loaded by GPU worker" (its maps check compares the END of
    `/proc/<pid>/maps` lines against `VLLM_NCCL_SO_PATH=/opt/patched-nccl/libnccl.so.2`, but
    the kernel resolves the symlink and maps the REAL path `…/libnccl.so.2.30.7` — checked
    live on the running rank-0 worker, pid 342). The frozen e36 `baseline.json` needed NO
    change: it pins the container-side mount destination `/opt/patched-nccl/libnccl.so.2`,
    identical for both libs.
  - **Fix (dual-lib manifest, Option B):** SHA256SUMS now carries one
    `<sha>  <staging-dir>/libnccl.so.2` line per generation (`nccl-patched/` legacy 2-PF
    afe5f486… = rollback artifact, kept; `nccl-patched-4pf/` 8de68def… = production).
    `verify-node.sh` gains `expect_sha` (selects by `NCCL_DIR` basename; unlisted dir =
    hard abort, keeping the no-silent-fallback invariant); `install-nccl.sh` selects the
    same way; `build.sh` pins the `nccl-patched/` line (it builds the legacy lib only);
    `expected.env`/README updated (legacy shape annotated, 4-PF provenance recorded).
    TDD: `scripts/tests/test-verify-node.py` covers both selections + the unlisted abort
    (RED verified: block failed exit 127 before implementation).
  - **Rollout:** read-only verification scripts — no boot, no restart, lane stays serving.
    deploy.sh fans SHA256SUMS to the nodes via the verify-manifest flow.
  - **Watch item:** verify-node.sh's patched-NCCL-sha row detail prints the selection
    source; the acceptance README section documents the dual-line procedure for future
    candidate adoption.
  - **Pre-existing, OUT OF SCOPE here:** `check.sh` (and test-check-f0.py) fail with
    "operational identity runtime source hash mismatch" on the CLEAN tree too — the
    grammar-fix commits (08b048c/34aa09e, dd1a039) modified three e03 override sources
    (prefill-cache-trim/gpu_worker.py, prefill-step-cap/scheduler.py,
    e36-lm-head-w8a16/model_runner.py) after
    `docs/operational-identities/2026-09-30-e36-lm-head.json` froze their hashes
    (e.g. model_runner: pinned d8fbfa51…, actual 598f9fd7…). The deployed files MATCH the
    working tree (verified live on the head node: `~/.local/tp4/experiments/e03/…` hashes
    598f9fd7/ae2bdaf5/c5d8b99f = workstation) — the crash-fix rollout deployed correctly.
    But `check-f0.py` in operational-identity mode hashes the WORKING-TREE copies against
    the frozen record, so the live check currently reports exactly that one problem
    (verified live 2026-10-11; the e36 BASELINE mode's 24 problems are frozen historical
    identity drift — that baseline predates the later site additions, needs no action).
    Refreshing the identity record for the three files is an owner decision (separate
    item, do NOT auto-remeasure — same rule as the 2026-09-29 note).
- **2026-10-11 — Clean transport A/B closed as MOOT (owner decision: 4-PF is permanent).**
  The A/B's only remaining purpose after the engine-crash root cause landed (2026-10-10 entry)
  was attributing this lane's bench numbers to the transport change; with the owner committing
  to the 4-PF configuration permanently there is no config choice left for the A/B to inform,
  so the rollback-config boot + re-adoption boot will NOT be performed. The 2026-10-10 T1
  bench (`noswitch-prep/benchmarks-fp8-4pf/`: c1_prose_short 94.4 tok/s, TTFT 0.50 s,
  c4_prose_short 170.2, acceptance 0.981) is adopted as the lane's 4-PF baseline for all
  future tuning comparisons; the NVFP4 Phase-E profile (json TTFT 6.5 → 0.5 s) remains the
  directional cross-lane evidence. Residual kept: the legacy 2-PF rollback path
  (`~/.local/lib/nccl-patched`, afe5f486…; rollback documented at `NCCL_IB_HCA` in
  `noswitch-prep/cluster.env`) stays staged and documented but is UNTESTED on the E36 stack —
  a future rollback boot under failure pressure would execute it for the first time.
  4-PF-active verification on 2026-10-11 (repo mirror `upstream/cluster.env` + deployed
  `~/.local/tp4/cluster.env` on spark-0f0b + live container env of `glm53_fp8_dflash_tp4`,
  up ~7 h, /health 200): 4-PF HCA list, `NCCL_DIR=nccl-patched-4pf` (lib mounted
  `/opt/patched-nccl`, v2.30.7), MERGE_NICS=0, SWITCHLESS_RING_ONLY=1,
  EXTENDED_IPV4_GIDS=1, PRESERVE_PCI_DOMAIN=1, zero NCCL WARN in boot logs.

- **2026-10-10 — REQUIRED+THINKING-ON ENGINE CRASH ROOT-CAUSED AND FIXED (site mods 8,
  9b, 10; the engine-level finding from the T1 four-PF boot entry below is closed).**
  - **Root cause (instrumented reproduction, four diagnostic boots):** the E35 runner-k
    worker-side draft trim (`e35_trim` in the e36 model_runner, promoted with the E35/E36
    stack) rewrites the step's `scheduled_spec_decode_tokens` AFTER arrival when its
    host-confidence policy trims drafts (K_HI=7 → K_LO=3). The engine built that step's
    grammar bitmask for the UNTRIMMED plan (rows = 7 drafts + bonus = 8) while the
    worker's logit mapping covers only the trimmed layout (3 + bonus = 4) —
    `apply_grammar_bitmask assert num_masks == len(mapping)` (measured 8 vs 4),
    EngineDeadError. Evidence trail: `GRAMMAR-DIAG-*` logs (scheduler BUILD/RETURN, core
    DISPATCH, shm_broadcast PICKLE, worker RECV/EXEC/MR) — wire carried 7, worker exec
    applied e35_trim to 3. The trim only fires on single-request batches verifying exactly
    K_HI drafts (`e35_participating_request`), which is why the batch-1
    dynamic_sd_table's k values (7/3) matched the two sides exactly.
  - **Why thinking-on + required:** thinking-phase drafts are low-confidence → the E35
    policy trims → first grammar step of the request arms the mismatch (deterministic).
    Thinking-off + required: structural-tag tokens are high-confidence → no trim → no
    crash. auto+thinking: no grammar request → no bitmask → trim harmless. NVFP4 lane:
    no E35 runner-k → unaffected (its old-style mapping + stock stack verified by source
    diff).
  - **Fix (two layers, e36 model_runner):** (1) when the E35 trim fires on a grammar
    step, sample() re-slices the bitmask to the verified-prefix layout (first k draft rows
    + bonus row) so row counts align; (2) grammar steps (`has_structured_output_requests`)
    never trim at all — the bonus-row mask and FSM advancement are computed for the full
    plan, and a trim leaves the FSM too-advanced (sampled tokens rejected: "Failed to
    advance FSM ... Terminating request" — observed at ~2/6 probes before layer 2 was
    added). Layer 2 makes layer 1 defense-in-depth.
  - **Withdrawn experiment (mod 9):** an async_scheduler pairing override (defer
    structured-output sampling on every grammar step) was tried first and withdrawn —
    instrumentation proved RPC dispatch is in-order (sample(N) always reaches the worker
    before exec(N+1)) and the pairing race never existed; the override instead caused
    empty-deque pops on isolated deferred steps and None-exec failures under the async
    runner protocol.
  - **Found-and-fixed en route — cluster.env 4-PF append regression:** the gitignored
    mirror had lost `-e NCCL_IB_EXTENDED_IPV4_GIDS=1 -e NCCL_IB_PRESERVE_PCI_DOMAIN=1`
    (noswitch-prep still had them); the first diagnostic boot failed on the four-PF
    listener contract. Restored to match noswitch-prep + the AGENTS.md shared contract.
  - **Diagnostics kept (mod 8/10, env-gated by `VLLM_GRAMMAR_DIAG=1`):** scheduler row
    accounting (GRAMMAR-DIAG-SCHED), worker state-dumping RuntimeError replacing the bare
    assert (GRAMMAR-DIAG-WORKER, always on — strictly better failure mode), pre-shard row
    view (GRAMMAR-DIAG-MR), pickle-boundary + worker first-touch views (GRAMMAR-DIAG-
    PICKLE/WORKER-RECV), e03 core immediate-vs-deferred visibility.
  - **Validation gates (all on the fixed stack, health 200 throughout, zero 500s):**
    required+thinking-on ×10 → tool_calls 10/10; required+`enable_thinking:false` →
    tool_calls + clean arguments; auto+thinking → tool_calls; corruption probe (🚀 + 海洋)
    intact; chat gate PASS. The engine survived the full suite back-to-back.
  - **Upstream context:** the crash family is vLLM #50924 (guided decode × spec decode
    engine death); the worker-side mapping fix #52436 is already in this image, the
    scheduler-side row-padding fix #50952 never merged, and #49738/#52477 are still open
    — the site fix supersedes them for this stack.

- **2026-10-10 — Lane switch NVFP4 → FP8 over the directory rename: first post-rename
  boot, PASS.** The switch executed the LANE-SWITCHING.md rename gotcha order (tear
  down → pull → restore mirrors → preflight → up via the NEW paths):
  - NVFP4 `./start.sh stop` from the OLD path (containers preserved, exited 0 on all
    4 ranks; left stopped — not `docker rm`'d, so a future NVFP4 relaunch must remove
    them first per its preflight).
  - `git pull` on all 4 nodes → dirs renamed on disk as predicted; the gitignored
    mirrors (`upstream/cluster.env`, `upstream/scripts/node/etc/<node>/*.yaml`) stayed
    orphaned under `glm-v53-flash-4x-noswitch/` — confirmed, so the mirror restore was
    MANDATORY (not the runbook's conditional "only if a repo config change is pending"):
    `cp noswitch-prep/cluster.env upstream/cluster.env` on all 4, `render-netplan.sh
    --write && --check` on the head (4/4 MATCH, 4/4 identical into the new paths), then
    `scripts/deploy.sh` from the head (first attempt hit a 180 s SSH timeout mid-push —
    the script is slow-but-idempotent; second run completed on every node). Deployed
    `~/.local/tp4/cluster.env` verified: 4-PF HCA list + `GPU_MEM_UTIL=0.85`.
  - Preflight `ALL CHECKS PASS` (netplan live==generated, 16 addrs, 16-ping two-view
    mesh, both NCCL SHAs). Gotcha: non-interactive SSH PATH lacks `~/.local/bin` —
    invoke `~/.local/bin/tp4ctl` by full path.
  - **Boot PASS first attempt post-rename:** `/health` 200 at ~760 s (fresh overlay,
    cold weight load), `/v1/models` `glm-5.3-flash` max_model_len 262,144. T1-style
    four-PF proof: zero NCCL WARN lines, engine past `ncclCommInitRank`; live container
    env on head + spark-6d24 = 4-PF HCA, `MERGE_NICS=0`, `SWITCHLESS_RING_ONLY=1`,
    `EXTENDED_IPV4_GIDS=1`, `PRESERVE_PCI_DOMAIN=1`, four-PF lib mounted. Worker-first
    start order observed (r3→r2→r1→head). Note: the SIRCL GID preflight line lists only
    the two f-port devices — that is SIRCL's own 2-HCA view, not the NCCL transport.
  - **Gates PASS:** corruption probe 0 U+FFFD (🚀 + 海洋 intact, `enable_thinking:false`)
    and forced `tool_choice:"required"` + thinking-off → clean
    `get_weather({"city":"Tokyo"})`. No bench (30-min warm-up rule; owner may bench
    later per bench-recipe-update).
  - Watch item for the NEXT switch (FP8 → NVFP4): same orphaned-mirror treatment may be
    needed for NVFP4-side untracked files (e.g. its `.env`) left under the old
    `glm-v53-flash-nvfp4-4x-noswitch/` dir.

- **2026-10-10 — RECIPE RENAMED:** `glm-v53-flash-4x-noswitch` →
  `glm-v53-flash-fp8-tp4-switchless` (owner reorg; the NVFP4 sibling renamed to
  `glm-v53-flash-nvfp4-tp4-switchless` in the same commit). Docs updated to the new
  paths; historical entries/branch names below keep their period names. Node-side:
  apply the rename at a lane switch (containers bind-mount paths under the recipe dir
  — see LANE-SWITCHING.md gotchas).

- **2026-10-10 — T1 four-PF bring-up COMPLETE (the gated cross-lane boot): PASS after
  two fast, diagnosed failures + one engine-level finding.** All work on branch
  `glm-v53-flash-nvfp4-4x-noswitch` (per the cross-lane handoff; the FP8 node checkout
  picks this branch up at the lane switch).
  - **Failure #1 — launcher GID preflight (fixed: site fix #7, `7844258`).** First
    `tp4ctl up` with the coupled flip died before Docker on every rank:
    `NCCL HCA/GID selection invalid: roceP2p1s0f0/1: no usable automatic IPv4-mapped
    RoCE v2 fabric GID` (same for the P2 f1 device). Root cause:
    `scripts/nccl_gid_check.py` hard-coded the addressed-netdev allowlist to the FIRST
    TWO `FABRIC_IFACES` entries (upstream 2-PF contract) — the P2-view HCAs could never
    pass. Fix: allowlist = first `len(hcas)` entries (N=2 reproduces upstream
    byte-for-byte; N=4 admits the P2 views). Re-verified on spark-0f0b: 4-PF PASS
    (GID 3 on all four HCAs, each mapped to its own netdev/address), 2-PF control PASS.
    **Also found that attempt: the stale-mirror gotcha has a second layer — `tp4ctl`/
    the launcher read the DEPLOYED `~/.local/tp4/cluster.env`, not the repo checkout.
    After any repo config change run `scripts/deploy.sh` (from the HEAD node — the
    workstation run hits the Windows python3 stub) before any boot.**
  - **Failure #2 — listener contract at ncclCommInitRank (fixed: coupled-edit fix,
    `253ae8f`).** Second boot failed ~30 s after distributed init on rank 0:
    `NCCL WARN TP4 listener contract: wrong listener count` → worker
    `RuntimeError: NCCL error: invalid usage` → engine init failed, container exited 1.
    Root cause: the four-PF lib's listener publishes the legacy TWO-GID handle unless
    `NCCL_IB_EXTENDED_IPV4_GIDS=1` (the NVFP4 render always sets it; this launcher
    never did). Fix: added `-e NCCL_IB_EXTENDED_IPV4_GIDS=1` and
    `-e NCCL_IB_PRESERVE_PCI_DOMAIN=1` (the NVFP4-validated PCI-root-preserving
    profile) to the `EXTRA_DOCKER_ENV` 4-PF append in `cluster.env` (the rollback line
    stays "delete the whole append"). Docker logs captured BEFORE teardown per the
    README rule.
  - **PASS boot (third attempt):** container env verified live — 4-PF HCA,
    `MERGE_NICS=0`, `SWITCHLESS_RING_ONLY=1`, `EXTENDED_IPV4_GIDS=1`,
    `PRESERVE_PCI_DOMAIN=1`, `LD_PRELOAD` mounted from `~/.local/lib/nccl-patched-4pf`
    (8de68def…), zero NCCL WARN lines, engine past `ncclCommInitRank` (attempt #2's
    failure point), weight load 186 s, E36 ready, `/health` 200 (max_model_len 262,144).
    The lib's listener contract passing IS the four-PF admission proof ("wrong listener
    count" is its hard gate).
  - **Functional gates:** chat gate "Rome" PASS; tool gate `auto` →
    `get_weather({"city": "Milan"})` PASS; corruption probe 0 U+FFFD (emoji + 海洋
    intact); forced `tool_choice:"required"` with `enable_thinking:false` → clean
    `get_weather({"city": "Tokyo"})` PASS.
  - **ENGINE-LEVEL FINDING (not transport): `tool_choice:"required"` WITH thinking
    enabled kills the engine.** The first required probe (default thinking) died on the
    first scheduled spec-decode step: `structured_outputs.py apply_grammar_bitmask
    assert num_masks == len(mapping)` → EngineDeadError → /health 000. `tp4ctl restart`
    recovered; required+thinking-off passes (above); auto+thinking passes. The crash
    path is vLLM grammar-bitmask × spec-decode scheduling — zero NCCL involvement — and
    the TC-45 structural-tag override is active. Almost certainly pre-existing on the
    E35/E36 stack (the thinking-on required combination may not have been exercised
    since those promotions; the documented TC-45 gate uses thinking-off or auto), but
    NOT A/B-verified — flagged for the owner as a separate bug; a rollback-config boot
    could isolate it, though the mechanics make a transport cause implausible.
  - **check-f0.py / verify-node.sh FAIL — explained, expected:** both compare against
    the frozen E36 identity whose NCCL staging SHA is `afe5f486…`; with `NCCL_DIR` at
    the four-PF lib they fail exactly the NCCL rows ("patched NCCL sha FAIL 8de68def…
    expected afe5f486…" ×4 + "patched NCCL not loaded by GPU worker" preload-pattern
    detection). 152 PASS / 5 FAIL / 4 WARN; fabric-check 8/8 PASS. The identity delta
    is the intended change; refreshing the frozen baseline (or dual-lib-aware checks)
    is a follow-up for the owner.
  - **deploy-host.sh --check DRIFT ×8 (orthogonal):** `/etc/netplan/40-cx7.yaml`
    (comment-header-only diff — the live file carries the NVFP4 Phase-E header;
    `verify-fabric-p2.sh`'s comment-stripped compare PASSES; do NOT re-deploy netplan
    while serving) and `/etc/default/grub.d/zz-tp4-perf.cfg` (unrelated to this work,
    likely pre-existing).
  - **Bench (measurement caveat recorded): `bench_recipe.py`'s boot-log gate counts the
    benign vLLM `usage_lib`/py-cpuinfo background-thread traceback that fires on every
    boot of this build and refuses to bench. Ran a SCRATCH copy (node /tmp, not repo)
    whose gate whitelists only that traceback (real tracebacks still block). Results
    (`noswitch-prep/benchmarks-fp8-4pf/`, tier 1, temp 0): c1_prose_short 94.4 tok/s
    (TTFT 0.50 s), c1_code_short 90.3, c1_prose_medium 53.9, c1_prose_long 27.3,
    **c4_prose_short 170.2 tok/s** (per-stream TTFT 1.11–1.27 s), spec acceptance
    0.981, KV pool 1,194,033 tokens. `pmu_replay_long` FAILs by lane design (no
    `--prefix-match-unit`; `cached_tokens` unreported — pre-existing, not transport).
    **No same-config 2-PF baseline exists for the E36 era** (2026-09-23 glmAB numbers
    are a different bench script + E21 config); the clean transport A/B on this lane
    would need one rollback-config boot + one re-adoption boot — owner decision,
    recommended before treating the bench comparison as conclusive. The T1 hypothesis
    (prefill-phase TTFT improves) is supported directionally by the NVFP4 Phase-E
    profile (json TTFT 6.5 → 0.5 s) but NOT re-measured here against a same-config
    control. The client-side 18.9 s mean TTFT (2026-09-28 agent traffic) is a different
    measurement (100K-token turn bodies through the proxy), not comparable to bench cells.
- **2026-10-10 — Phase-E cross-lane adoption, prepared on the sibling branch
  (**status: VALIDATED — see the T1 entry above, three-boot sequence recorded there**).** Three coupled
  pieces, all landing in the same commit on `glm-v53-flash-nvfp4-4x-noswitch` (worktree
  stays on the per-lane branch rule: the FP8 node checkout picks this up at the next
  switch):
  1. **Netplan generator fixed at the root (closes the re-deploy hazard).**
     `render-netplan.sh` gained the opt-in `FABRIC_P2_VIEWS` flag (upstream VENDORED-AT
     mod #6; default 0 = upstream behavior byte-identical); `cluster.env` sets it to 1.
     The regenerated files now byte-match the live `/etc/netplan/40-cx7.yaml` on all four
     nodes (verified live via sudo diff, and against the NVFP4 lane's
     `site/netplan/<node>/` reference) — `render-netplan.sh --check` passes and the
     render/deploy workflow is safe again. BEFORE this fix, re-deploying host config
     would have stripped the P2 addresses and broken the NVFP4 lane (the warning note
     below stays as history; the hazard is now closed, not just warned about).
  2. **Cross-lane preflight: `noswitch-prep/scripts/verify-fabric-p2.sh`** (read-only,
     safe while serving): render --check gate, live-netplan-vs-generated per node, all
     fabric interfaces UP @ expected addresses, the 16-ping two-view mesh (P2 views
     verified carrying traffic live, 16/16), and both NCCL staging SHAs (`afe5f486…`
     legacy 2-PF, `8de68def…` four-PF). Run before any lane bring-up or switch.
  3. **The four-PF NCCL contract adopted for THIS lane** (the NVFP4 Phase-E measured
     profile: json aggregate +26%/+15% at c8/c16, prefill TTFT 6.5 s → 0.5 s; c1 flat).
     One coupled edit in `cluster.env` (rollback documented at `NCCL_IB_HCA`):
     `NCCL_IB_HCA` → all four PFs, `NCCL_DIR` → `~/.local/lib/nccl-patched-4pf`
     (8de68def…, built+validated on this fabric by the NVFP4 Phase E), and an
     `EXTRA_DOCKER_ENV` append of `NCCL_IB_MERGE_NICS=0` +
     `NCCL_SWITCHLESS_RING_ONLY=1` (the four-PF build hard-rejects this launcher's
     former merged-vdev default; the launcher's hardcoded `NCCL_SKIP_TREE_CONNECT=1`
     stays — same othexmr patch stack). Expected payoff on this lane's measured pain:
     prefill-phase collectives under concurrency (mean TTFT 18.9 s, inter-turn prefill
     barriers) — NOT single-stream decode. GATED: the next FP8 boot must run the full
     gate suite + a TTFT/throughput comparison vs this lane's current numbers, and the
     lane switch both ways must be exercised, BEFORE this branch merges.
- **2026-10-10 — cross-lane note (no action taken on this recipe): the host netplan now
  also carries the NVFP4 lane's Phase-E P2-view addresses.** The sibling recipe
  `glm-v53-flash-nvfp4-tp4-switchless` completed its four-PF NCCL upgrade (its research.md
  "PHASE E COMPLETE"): `/etc/netplan/40-cx7.yaml` on all 4 nodes gained one address per
  node per cable on the second PCIe views (`enP2p1s0f0np0`/`enP2p1s0f1np1`, last octet =
  f-port octet + 2). This lane is UNAFFECTED — the f-port addresses are unchanged and the
  2-PF subset it uses still works; the FP8 NCCL build (`afe5f486…` in
  `~/.local/lib/nccl-patched/`) stays in its own directory (the four-PF lib lives in
  `~/.local/lib/nccl-patched-4pf/` and would hard-reject this lane's merged-vdev
  config). **Do NOT re-run this recipe's netplan re-deploy** (`scripts/node/etc/` 2-address
  files): it strips the P2 addresses and breaks the NVFP4 lane. If a netplan regeneration
  is ever needed, additively re-add the P2 addresses per the NVFP4 recipe's
  `site/netplan/<node>/40-cx7.yaml`.
- **2026-10-09 — Engine-fatal grammar-bitmask assert on a structured-output request (one-off); full restart, identity re-verified. WATCH ITEM.**
  ~01:32 node time; the stack had been up ~8 days with ~1370 admissions and NO
  config change (same E29/E35/E36 lane since the 09-30 rollout). Rank-0
  `Worker_TP0` died in `vllm/v1/worker/gpu/structured_outputs.py:102`
  (`assert num_masks == len(mapping)`) during `sample_tokens` → EngineCore fatal
  → chat 500s → clean container exit 0; ranks 1–3 stayed up waiting. Scheduler
  dump: exactly one in-flight chat completion
  (`chatcmpl-a3954f9006b57a47-a2572b4f`) with `structured_output_requests=true`
  / `structured_output_tokens=true`, mid draft-verify step
  (`num_scheduled_tokens=8`, 7 draft tokens — the E28 solo k=7 schedule). The
  image's adaptive-verification grammar path (`_build_grammar_mapping`, keyed by
  (request, position) over `cu_num_logits` / `num_draft_tokens_per_req`)
  mismatched the bitmask row count at a step boundary. First occurrence ever on
  this cluster. Upstream vLLM carries a same-family bugfix (#44297, structured
  output + spec-decode bitmask/advance at the reasoning boundary) — a known bug
  class in the spec-decode + grammar path, not a fabric/config fault.
  **Recovery**: full `tp4ctl down`/`up`; `/health` 200, both functional gates
  (thinking-off chat → "Rome"; tool gate → `get_weather {"city": "Milan"}`),
  `check-f0.py` → `2026-09-30-e36-lm-head CHECK PASS`.
  Ops note from this cycle: interrupting the `tp4ctl up` health-wait loop skips
  its post-health step that stops `tp4-flusher` — the flusher kept running on
  all four ranks and `check-f0` flagged "flusher not proven inactive" ×4 until
  `tp4-flusher` was stopped manually on every node. Let `tp4ctl up` run to
  completion, or stop the flusher before running `check-f0.py`.
  **If it recurs**: capture the scheduler dump and whether the triggering
  request used `tool_choice:"required"`/named (the `VLLM_ENFORCE_STRICT_TOOL_CALLING`
  grammar, §2.6/TC-45), then report to jnardiello upstream / forum thread
  382459 with the image digest. The assert sits in the base
  `structured_outputs.py` path, so TP4_ENV rollback lanes alone may not clear
  it — treat as image/upstream bug to adopt a fix for.

- **2026-09-30 — Upstream refresh 080fe09 → ed365a6: the SparkCache disk-capacity fix lands (branch `glm-v53-flash-4x-noswitch-sparkcache-diskbound`).**
  Upstream moved 15 commits (Sep 27–30) with NO image bump (digest stays
  `sha256:0d40…`/IMAGE_ID `5e32aaa1bbe3…`) — everything is runtime overlay/config. The
  headline is exactly our #2 long-term fix: **E31-MB disk capacity** — the SparkCache chunk
  store now evicts LRU above 200 GiB (`SPARK_CONTEXT_CACHE_MAX_BYTES=214748364800`) down to a
  160 GiB watermark (`SPARK_CONTEXT_CACHE_LOW_WATERMARK_BYTES=171798691840`), with per-rank
  worker scans at startup. This replaces the unbounded growth that filled the nodes' disks
  (SparkCache chunk store was 1.7 TB/node at audit). Delivery chain: `cdf9b17` (E31 indexer
  tail-ring + memory-bounded recipe: 8 MiB transfers, 1 GiB transient budget/rank, 1 GiB
  admission floor, 14 GiB KV pool, allocator trim, 6,912-token step cap, bounded admission
  6 active/128 queued) → `ed365a6` (E31-MB disk capacity + E35 confidence-based verify length
  (3-or-7 drafts from DFlash2 confidence, `hybrid` policy, rank-0 broadcast, 0.5 s re-read)
  + E36 INT8 W8A16 Marlin `lm_head` group 128 (~155 MB/rank freed, perplexity +0.018%); E36
  is the new baseline; E32 discarded). Rollback ladder:
  `operational-20260930-e35.env` → `operational-20260930-e31-mb.env` →
  `operational-20260929-memory-bounded.env` → `operational-20260929-sparkcache-protected.env`
  (complete protected E31, 16 GiB pool) → `baseline-20260928-e31.env`. The new default uses a
  new `ram-budget` SparkCache connector via patch-05 substitution in `EXTRA_DOCKER_ENV`; the
  E22b store namespace is unchanged, so the existing on-disk cache stays valid. Site merge:
  all 58 site-modified files accounted for (11 changed upstream — three-way merged, 6 clean
  + `scripts/node/README.md` resolved by hand + 3 resolved by taking upstream structure with
  `scripts/node/README.md` resolved by hand + 3 resolved by taking upstream structure with
  site path rewrites; 47 restored from git untouched); site path rewrites re-applied to all
  newly adopted functional files (reference envs, test constants, resilience tooling
  `campaign.py`/`prepare_overlay.py`, fidelity fixtures); docs keep upstream's `~/tp4` prose
  per site convention. SITE MOD 7 (parser override) re-added to `cluster.env.example`'s
  `EXTRA_DOCKER_ENV` (end-of-value) and handled in tests by stripping the adjacent
  `(-v, glm47_moe.py)` pair from template-derived commands only — the upstream-pure
  reference/fixture envs (incl. `baseline-20260925-e29.env` / `baseline-20260928-e31.env` /
  `operational-20260929-sparkcache-protected.env`) stay byte-faithful to upstream, since the
  accepted-recipe ladder reconstructs them from deltas and a parser injection would misrepresent
  the measured recipes. Post-merge state: full `./scripts/check.sh` → `check: PASS` (all
  gates; fidelity fixtures `le36-m*.env` regenerated from `make_overlays.py`, which is
  canonical). `VENDORED-AT.md` pin updated.
  NOT done here (deliberate): site `cluster.env` / `noswitch-prep/cluster.env`
  regeneration and the actual rollout — needs an authorized window and, per owner
  constraint, the owner switches backend models first; nothing is committed or pushed yet.

- **2026-09-30 — Correction to the 09-29 rollout note + offline parity gate repaired (`glm4x-parity-test-fix`).**
  The 09-29 note's premise ("`check-f0.py` will show a mount/identity delta vs the frozen
  2026-09-25-e29 baseline after the TC-45 rollout … needing an owner decision on a new
  baseline") is wrong: `check-f0.py` is additive-tolerant — its command identity check only
  pins the files and env keys listed in `baseline.json`'s
  `operational_identity.container_file_sha256` (and their env keys), so the additive parser
  override mount does not fail it. No new baseline record is needed on that account; the
  frozen E29 record stays valid (live confirmations below). The real breakage the rollout
  caused was **offline**: `./scripts/check.sh` had been failing since the 09-27
  `~/tp4` → `~/.local/tp4` relocation (stale `~/tp4` literals in seven test files,
  uutils-`stat` ordering in `test-agent-preflight.sh`, mutable image tag in the F0 rollback
  overlay + stale F0 reference-manifest artifact pins, and `f0-reference.py`'s
  macOS-only `/private/tmp` scratch dirs) and the TC-45 addition then also invalidated the
  E29 command-parity constants (the end-drain scheduler mount replaces the E27c mount in
  place → the delta carries one `-v`, not two; the rollback envs rebuild
  `EXTRA_DOCKER_ENV` completely → their deltas include the parser mount pair). All fixed on
  branch `glm4x-parity-test-fix`; `./scripts/check.sh` → PASS end to end (worktree
  validated; `docs/benchmarks/` copied locally for the check, stays gitignored).
  **Live state confirmed read-only, no restart**: `check-f0.py` → 2026-09-25-e29 CHECK
  PASS (first two runs returned transient FAILs — most plausibly cold-SSH probe timeouts;
  reports are auto-wiped with WSL `/tmp` on VM teardown, then three consecutive runs were
  clean PASS with zero rank problems). Open item unchanged from 09-29: the default now
  differs from the *measured* E29 candidate command by the parser override mount — cosmetic
  for `check-f0.py`, but the owner may still want a recorded identity note at the next
  authorized window; do NOT auto-remeasure.

- **2026-09-29 — TC-45 fixed at recipe level: `tool_choice="required"`/named enforcement via glm_4_7 structural tag (new mod §2.6).**
  Diagnosis (proven from the image's own source, extracted read-only on spark-0f0b via
  `docker create`/`docker cp` from `glm53_fp8_dflash_tp4`): the image's parser manager
  (`vllm/parser/parser_manager.py:142-143`) returns the raw engine class
  `Glm47MoeParser` when both parser roles resolve to it, discarding the registered tool
  parser's `structural_tag_model = "glm_4_7"`; the engine path's `adjust_request`
  (`vllm/parser/engine/parser_engine.py:204-207`) only sets `skip_special_tokens=False`,
  so no structural tag is ever applied and `required` behaves as `auto` — the model
  answers in content. The grammar side was already ready: the image's xgrammar ships the
  builtin `glm_4_7` structural tag (`builtin_structural_tag.py:1574`, required branch =
  `TagsWithSeparatorFormat(tags=…, at_least_one=True)`). The env gate is NOT the problem:
  `VLLM_ENFORCE_STRICT_TOOL_CALLING` defaults `True` (`envs.py:250,1793`) and the container
  does not override it. The official 0.28.1 GLM-5.3-Flash base (intel recipe image) has
  this wiring fixed, which is why intel runs pass TC-45.
  Fix: new override `upstream/scripts/node/overrides/vllm/parser/glm47_moe.py` (exact image
  copy + site patch: `STRUCTURAL_TAG_MODEL`, `adjust_request` → `_apply_structural_tag`
  mirror of `DelegatingParser._apply_structural_tag`, `reasoning=False` matching the
  intel-proven behavior; `auto` w/o strict tools = no-op) + `-v` mount appended to
  `EXTRA_DOCKER_ENV` in `upstream/cluster.env.example`, `upstream/cluster.env` (gitignored)
  and `noswitch-prep/cluster.env` + `deploy.sh` `REMOTE_DIRS` += `tp4/overrides/vllm/parser`.
  Expected effect: tool-eval-bench `--hardmode` TC-45 −2 → pass, ~90 → ~92 (intel median ~93;
  TC-43/68/80 are model-level on both stacks, TC-85 a coin-flip — not targeted).
  Rollout: commit/push + `git pull` on the four node clones ONLY, then PAUSE until the
  user switches the agent backend model; after go-ahead: copy gitignored `cluster.env`
  into rank0's clone → `deploy.sh --check` → `deploy.sh` → full-cluster `tp4ctl restart`
  → gates (coherent-response + tool-call + NEW `tool_choice:"required"` curl must return
  `tool_calls`) within 2 min of `/health` 200. Cold boot ~13 min. `check-f0.py` will show a
  mount/identity delta vs the frozen 2026-09-25-e29 baseline — expected consequence
  needing an owner decision (new baseline record), not a bug; do NOT auto-remeasure.
  Later (owner-authorized): rerun tool-eval-bench `--hardmode --seed 42` matched sampling
  (t=0.2, top_p=0.95) ×3.

- **2026-09-28 — "Hung again" report investigated: NO second wedge; recurring short TTFT stalls quantified instead.**
  User reported the recipe hung again (~01:38 EDT / 05:38 UTC). The engine was NOT wedged:
  generation probe + both functional gates passed on the spot, `/metrics` showed 0 running /
  0 waiting, and the retained rank logs (covering since the 21:17 UTC restart — the recovery
  boot from the 2026-09-27 incident) contain **zero** `shm_broadcast` "broadcast block"
  messages, zero tracebacks / NCCL warns / aborts / errors on any of the 4 ranks. All 555+
  requests since boot completed (486 stop / 69 length / **0 abort / 0 error**). What the
  night actually shows (10-s engine stats timeline, 1137 ticks): agent-session bursts
  00:08–02:44 and 04:00–05:24 UTC with a RECURRING single-tick stall signature —
  `pp=0 tg=0 run=1` for one 10-s window between turns (KV-transfer/prefill barrier) — i.e.
  the mild, self-recovering form of the 09-27 wedge signature (frozen `run` + zero
  progress). Client cost since boot: mean TTFT 18.9 s (337/558 ≤10 s, 26 requests 80–160 s),
  mean e2e 43.9 s, ~11 requests >4 min, none > ~16 min — agent turns genuinely "feel hung"
  in the tail, but everything completed. `model-name-proxy` (OpenResty) logged "client
  request body buffered to temporary file" ~every 45–90 s during the last burst,
  correlating 1:1 with the agent's ~100K+-token turn bodies; agent traffic reaches vLLM via
  that proxy (vLLM sees 127.0.0.1). **Tail attributed 2026-10-11 (see the dated entry above):
  engine queueing under burst concurrency + ~100K-token chunked prefill + E31 barrier ticks —
  the proxy's temp-file buffering measured at 0 s warn→admit and ≤10 ms of disk I/O; exonerated.** Idle-period stats-tick gaps (2.4 h / 75 min) are the
  logger going quiet with zero requests — NOT stalls (**refines the 09-27 lesson: judge
  stats-line gaps only while requests are in-system**). Hardware/fabric exonerated: all 4
  nodes P0, zero throttle counters, worker logs clean. Boundary marker if a real wedge
  follows: last server-visible request completed 05:24:14 UTC (sparkcache failed=0B); first
  probe after the report = `chatcmpl-875d4284a282190c` at 05:36:56 UTC.
  **Runbook gap found:** py-spy is NOT installed on the nodes or in the container — the
  "capture worker stacks BEFORE restarting" step of the 09-27 recurrence plan currently has
  only the destructive SIGQUIT fallback. Watch item (RESOLVED 2026-10-11 — see the dated
  verdict entry above): the sparkcache capacity line read
  `healthy=no used=0.0/0.0GiB` on every emission on that boot (entries 5337→6361, publications
  460.9 GiB, failed=0B) because the capacity policy did not exist yet pre-E31-MB — the gauge
  truthfully reports the absent policy while the data path (publications) stays healthy.
  Live and healthy (`yes`, real usage) on every boot since the 09-30 rollout. The burst
  deferrals (`wait=1` at run≤2) were captured live 2026-10-11: `Deferred: 1 reqs` is the
  E31 deferred-admission path (KV transfer in progress), by design, unrelated to the gauge.

- **2026-09-27 — Migration completion: five bring-up failures after the path moves, all fixed; identity re-verified.**
  After the §2.5 relocation, `tp4ctl up` failed until five issues were resolved:
  1. **HF-cache snapshot symlinks dangle inside container mounts** — the launcher mounts the
     snapshot dir itself (`/model`, `/draft`), so the hub-style `../../blobs/…` symlinks
     resolved on the host but broke inside the container. Fix: converted every snapshot
     symlink to a hardlink of its blob (both `models--zai-org--GLM-5.3-Flash` and
     `models--incoai--GLM-5.3-Flash-DFlash2`, all 4 nodes). Serving requires hardlinked
     snapshots; plain `hf download` still verifies them fine.
  2. **Node clones lack `cluster.env`** (gitignored upstream — and absent from this repo's
     commit because of that ignore rule). `check-f0.py` sources it from the repo checkout.
     Copied the site `upstream/cluster.env` onto rank0's clone. **Update-playbook rule: after
     re-cloning or a fresh deploy target, re-copy the site cluster.env into
     `upstream/` and run `bash scripts/render-netplan.sh --write` (derived netplan/iptables
     envs are also gitignored) before `verify-node.sh`/`check-f0.py`.**
  3. **Exec bits lost**: the vendored scripts committed as 644; a node `git reset --hard`
     restored 644 and `check-f0.py` died on `PermissionError: verify-node.sh`. Fixed by
     tracking mode 755 (`git update-index --chmod=+x`, commit `c9ab389`).
  4. **verify-node patches check** still pointed at `$HOME/patches` — the §2.5 sed missed it
     because the literal is `"$HOME"/patches/*.py` (slash outside the quotes). Fixed in
     commit `7d664d8`. Sed sweep lesson: also match `"$HOME"/<dir>` quoting variants.
  5. **Revision-provenance markers were lost with the flat dirs** and are not part of the HF
     hub layout: `.glm53-fp8-synced` (model rev, written by fetch-fp8-weights.sh after
     manifest verification) and the drafter's
     `.cache/huggingface/download/config.json.metadata` (line 1 = commit). Restored both with
     their exact original contents on all 4 ranks after re-verification (72-file size check +
     2-shard sha256 per node; drafter proven independently via fresh `hf download --revision`
     on two ranks). `hf download` ignores the extra files.
  Final state: `verify-node.sh --quick` 153 PASS / 0 FAIL (4 pre-existing tailscale WARNs),
  `check-f0.py` → **2026-09-25-e29 CHECK PASS**, `tp4ctl health` smoke coherent, tool-call
  gate returns correct arguments, `model-name-proxy` healthy (transient unhealthy flag during
  the down window only). 6d14 now on branch `glm-v53-flash-4x-noswitch` (was stale `main`);
  6d90/6d24 cloned on the branch; all 4 node clones at `7d664d8`.

- **2026-09-27 — Home-directory cleanup: runtime relocated out of `$HOME` (mod §2.5).**
  All recipe-owned runtime assets moved out of the node home dirs: `~/tp4` →
  `~/.local/tp4` (deployed runtime; on PATH via `~/.local/bin/tp4ctl`), `~/nccl-patched`
  → `~/.local/lib/nccl-patched`, `~/vllm-cache` → `~/.cache/tp4-vllm-cache`, weights →
  `~/.cache/huggingface/hub/models--zai-org--GLM-5.3-Flash/snapshots/690b705278a3…`
  (converted in place from the flat `--local-dir` download into proper HF cache layout:
  blobs + `snapshots/<rev>` symlinks, `hf download` now only verifies) and the drafter
  repointed to the already-cached
  `models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4ea…` (identical pinned
  revision; flat draft copy deleted). 48 files in `upstream/` + the site mirror had
  their path literals rewritten; `./scripts/check.sh` PASS. E03 benchmark artifacts and
  the one-off bench scripts moved into `noswitch-prep/benchmarks/` / `bench-scripts/`;
  scratch logs, `~/sparkinit`, stale duplicates and the 6d14 NCCL build tree deleted.
  Paths deployed to the nodes and the repo updated together before bring-up.

- **2026-09-27 — First silent TP-step hang (root cause unknown), recovered by full restart.**
  ~13:45 node time: ~25 min after a normal request completed (SparkCache committed clean,
  `failed=0B`), the engine core wedged: `shm_broadcast.py:801` "No available shared memory
  broadcast block found in 60 seconds" repeating every minute with no recovery. `/health`
  stayed 200 the whole time (API process independent), so outwardly the endpoint looked
  alive while nothing generated — agents saw request timeouts with zero tokens. Evidence:
  no error on ANY rank (1–3 logs clean), no NCCL/fabric message, no OOM (E20 probe clean,
  `num_ooms:0`), `num_requests_running: 1` frozen — a stuck generation holding the engine
  step while everything behind it queued. This message is in the recipe's known/benign list
  only when transient; **non-transient (looping for minutes) = wedged engine**.
  Diagnosis key: the engine core is the *victim* (it cannot broadcast a step) — look at
  *worker* process state first; here all workers looked alive but none advanced.
  **Recovery**: `tp4ctl restart` (full-cluster procedure, per the single-rank invariant).
  Cold boot ~13 min (checkpoint load + flashinfer autotune + graph capture), `/health` 200
  at 775 s, smoke gate passed, `check-f0.py` → 2026-09-25-e29 CHECK PASS.
  **If it recurs**: capture worker stack state BEFORE restarting — py-spy is now installed
  host-side on all 4 nodes and verified against the live workers (see the 2026-10-11
  entry for the exact procedure: `docker top glm53_fp8_dflash_tp4 -eo pid,args | grep
  'VLLM::'` for pids, then `sudo ~/.local/bin/py-spy dump --pid <pid>` per rank, 2–3 dumps
  a few seconds apart); SIGQUIT remains the last-resort fallback only. Then
  instrument the E29 end-drain scheduler + async-scheduling path as prime suspects; also
  note the last completed SparkCache ticket/request-id from the logs (this time
  `chatcmpl-a78184e6c18fa28c`, ticket `e216673e`) as the boundary marker. First such wedge
  on this cluster; no pattern yet.

- **2026-09-26** — Initial vendoring of jnardiello `080fe09` into `upstream/`
  (detached checkout, committed at `9cd37a0`); created this research.md documenting
  the 4 local mods (§2), site-owned files (§3), and the update procedure (§4).
  Added `upstream/VENDORED-AT.md` recording the pin + mods.
