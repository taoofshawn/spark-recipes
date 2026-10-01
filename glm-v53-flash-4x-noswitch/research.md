# research.md — glm-v53-flash-4x-noswitch

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
  that proxy (vLLM sees 127.0.0.1). Idle-period stats-tick gaps (2.4 h / 75 min) are the
  logger going quiet with zero requests — NOT stalls (**refines the 09-27 lesson: judge
  stats-line gaps only while requests are in-system**). Hardware/fabric exonerated: all 4
  nodes P0, zero throttle counters, worker logs clean. Boundary marker if a real wedge
  follows: last server-visible request completed 05:24:14 UTC (sparkcache failed=0B); first
  probe after the report = `chatcmpl-875d4284a282190c` at 05:36:56 UTC.
  **Runbook gap found:** py-spy is NOT installed on the nodes or in the container — the
  "capture worker stacks BEFORE restarting" step of the 09-27 recurrence plan currently has
  only the destructive SIGQUIT fallback. Watch item: the sparkcache capacity line has read
  `healthy=no used=0.0/0.0GiB` on every emission since boot (entries 5337→6361, publications
  460.9 GiB, failed=0B) — cosmetic vs unwired gauge unknown; correlate with the request
  deferrals seen during bursts (`wait=1` at run≤2, below the MAX_NUM_SEQS=6 cap; reason
  breakdown not captured live — likely the 'deferred'/KV-transfer reason).

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
  **If it recurs**: capture worker stack state (py-spy or SIGQUIT) BEFORE restarting, and
  instrument the E29 end-drain scheduler + async-scheduling path as prime suspects; also
  note the last completed SparkCache ticket/request-id from the logs (this time
  `chatcmpl-a78184e6c18fa28c`, ticket `e216673e`) as the boundary marker. First such wedge
  on this cluster; no pattern yet.

- **2026-09-26** — Initial vendoring of jnardiello `080fe09` into `upstream/`
  (detached checkout, committed at `9cd37a0`); created this research.md documenting
  the 4 local mods (§2), site-owned files (§3), and the update procedure (§4).
  Added `upstream/VENDORED-AT.md` recording the pin + mods.
