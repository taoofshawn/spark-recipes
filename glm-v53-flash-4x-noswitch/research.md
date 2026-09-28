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
