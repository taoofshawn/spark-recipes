# VENDORED-AT — upstream `christopherowen/spark-ds41f`

- **upstream:** `github.com/christopherowen/spark-ds41f` (Apache-2.0)
- **pin:** tag **`spark-ds41f-1120`** (annotated, 2026-11-20) → commit
  `d3d40d941f2752fea95aa770eb10d41b5ad5aa1a`
- **vendored:** 2026-11-22, closure-only (1,426 files). `.git` stripped; this tree is a
  subdirectory of `spark-recipes`, so `deployment.repository` is *this* repo and
  `deployment.path` is `{home}/code/spark-recipes/deepseek-v41-flash-christopherowen/upstream`.
- **previous plan:** a fork (`taoofshawn/spark-ds41f`, branch `site-shawndo`) — superseded.
  See `../../.research/HANDOFF-ds41f-inrepo-20261122.md` §1 for the comparison.
- **superseded research:** `.research/spark-ds41f-adoption/spec-and-plan.md` (1107 pin).

## What is vendored, and what is not

| vendored | why |
|---|---|
| `spark_ds41f/` (the CLI), `bin/`, `scripts/`, `docker/`, `config/`, `releases/`, `requirements/`, `upstreams.lock.json`, `pyproject.toml`, `LICENSE`, root `*.md`, `.gitignore`, `.gitattributes`, `.dockerignore` | the tool, build inputs, the Dockerfile, the release lock |
| `patches/` + the five series closures under `experiments/` (**730 files**: 5 series + 725 patches) | `build prepare` applies them and refuses a missing patch |
| `manifests/sources/`, `manifests/suites/2026-11-20-spark-ds41f-1120-tp4/` | `build prepare --lock`, and the qualified-shape identity |
| `experiments/2026-09-29-determinism/` (319 files) | `sonar.py` hard-codes it for `sonar mixed`; keeps the `bench --suite` menu working |
| `docs/` (4.6 MB) | `README.md` here cites `setup.md`, `memory-profiles.md`, `switchless-topology.md` |
| `host/recovery/` (10 files) | `scripts/host-recovery` (setup stage 1) sources it |

**Not vendored:** ~30,800 files — bench receipts under `experiments/**/{runs,receipts,results}`,
`tests/`, `state/`, `site/`. Nothing in build, doctor, cluster or bench reads them.

## The four site patches (re-apply after every re-vendor)

Each one exists because this checkout is **not** the repository root, and upstream's tooling
assumed it was. Scripts: `.research/spark-ds41f-adoption/scratch/patches/`.

| # | file | change | what breaks without it |
|---|---|---|---|
| O1 | `config/nodes-ring4.json` (**new**, committed) | our ring map, 2-PF subset | the node map cannot be named `*.local.json` (secret scan fails it) and cannot be `config/nodes.json` (the 3-node default that `cluster sync` rewrites) |
| O2 | `spark_ds41f/config.py` | `git status --porcelain --untracked-files=all -- .` and `git ls-files -z -- .` | the repo-wide checks see every other recipe → "deployment requires a clean local repository" forever |
| O3 | `spark_ds41f/cluster.py` | `profile_committed()` resolves `config/*.json` under the checkout's own path (via `git rev-parse --show-toplevel`) | git resolves `REV:config/x.json` from the top level → always False → the first `cluster sync` aborts "the deployed commit does not carry it" |
| O4 | `config/site.json` (**new**, `git add -f`) | `{"shape": "tp4-ring-4k"}` | ignored by upstream's `.gitignore`; without it the site's shape does not reach the nodes |

`config/cluster.json` is also rewritten for this site (home, deployment, master, sockets, GID
pin dropped, `tp4-ring-4k` added, served name). It is a **site file**, not a patch: the
release lock lives in `releases/spark-ds41f-1120/cluster.json` and is untouched.

### Verify a re-vendor

```bash
# 1. the patch closure is byte-exact (no CR rewriting, no missing patch)
python3 - <<'PY'
import pathlib
root = pathlib.Path(".")
bad = [str(p) for p in root.rglob("*")
       if p.is_file() and b"\r" in p.read_bytes()
       and p.suffix in {".py",".sh",".json",".md",".env",".patch",".yaml",".yml",".txt"}]
print("CR-bearing text files:", len(bad)); bad and print("\n".join(bad))
PY

# 2. only the intended files differ from the pin
git -C <a checkout at d3d40d94> diff --stat d3d40d94 -- .     # expect: config/cluster.json,
                                                              # spark_ds41f/config.py, spark_ds41f/cluster.py
                                                              # + the two added config files

# 3. it still works
python3 bin/spark doctor                    # configuration OK
python3 bin/spark --cluster-config config/cluster.json cluster start     # the plan, no --apply

# 4. modes survive: the node-side tooling runs bin/spark and the executed scripts/* as paths
git ls-files -s -- bin scripts | grep -v '100644'
```

**Step 4 is not optional.** The first vendor pass committed the whole tree `100644`, and upstream pins
`bin/spark` plus every executed script (`memguard.sh`, `memguard-start.sh`, `host-recovery`,
`configure-swap`, `install-*`, `lab_idle_start.sh`) at `100755`. A mode-stripped checkout fails on the
nodes with `Permission denied` at `cluster start` — it did, on the first bring-up. Restored in
`4f9d77f`, `2edb188`, `2bb68d0`; blob SHAs match the pin, so only modes changed.

## Byte identity and line endings

The build consumes 730 files and **none of them contain CR bytes upstream**, so `git am`
receives exact bytes. This recipe ships `.gitattributes` with `* -text`: measured on a
workstation with `core.autocrlf=input`, a CRLF blob hashes as `422c2b7a…` through the index
without the attribute and `c30dea8a…` (raw) with it. The cost of being wrong is a patch that
silently no longer applies, so the attribute stays.

**When re-vendoring, extract to a Linux filesystem.** Applying the site scripts on Windows
with a default text-mode writer is what produced CRLF in `cluster.json` and `cluster.env` on
the first attempt; the appliers now force `newline="\n"`, but the vendored *unmodified* files
must never pass through a text-mode copy tool either.

## Upstream drift at adoption

`main` was **83 commits past the pin** (tip `candidate-68`, 2026-11-22). Those are candidate
branches (LSA, CopyDrafter, adaptive verification, c32) — never a deployment branch. Deploy
only a promoted `spark-ds41f-*` tag.