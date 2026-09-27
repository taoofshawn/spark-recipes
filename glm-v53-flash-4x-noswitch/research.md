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
is different**: its `upstream/` is a *detached live checkout* that carries four
deliberate site modifications. When re-syncing against a newer jnardiello commit
(`git diff 080fe09..<new>` or re-vendoring), these four files must be **kept
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

1. **Snapshot**: note the pin (`080fe09`), the 4 mods (§2), current branch
   (`glm-v53-flash-4x-noswitch` — stay on it; this recipe's work never merges to
   `main` without the user), and the served identity (E29, `check-f0.py` default).
2. **Sources**: jnardiello repo commits/PRs/issues; NVIDIA forum (categories 721/723)
   with `glm`/`5.3`/`tp4`/`switchless` terms + general sweep since the last review
   date (this file's changelog dates are the last-review markers).
3. **Diff**: `git ls-remote` the upstream repo; diff `080fe09..HEAD`. Classify each
   change: adoptable (new knobs, fixes), inapplicable (their cluster's site values),
   or conflicting with §2/§3 (keep ours).
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
