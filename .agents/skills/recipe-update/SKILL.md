---
name: recipe-update
description: Use when reviewing a recipe in the spark-recipes repo against its upstream sources for updates, fixes, or improvements — checking the NVIDIA forum, the recipe's upstream GitHub repo(s) and sibling repos, HF model/image revisions, and other sources — then starting an update branch, committing, and pushing. Triggers: "review recipe <name> for updates", "check <recipe> upstream", "is there a new fix for <model>", "update <recipe> from upstream", "latest commits for <recipe>". Does NOT bring the recipe up on the sparks (that is bring-up-spark-recipe).
---

# Recipe Update

Research pass → evaluate → branch → commit → push for a `spark-recipes` recipe
(e.g. `glm-v53-flash`, `deepseek-v4-flash-*`, `mimo-v25-dflash-tonyd2wild`). This is
the standing forward-port task for this repo: find anything new in the recipe's
upstream sources, decide what's worth adopting on THIS cluster, and land it on an
update branch — while NEVER touching the running cluster.

## Scope boundary

This skill researches, evaluates, branches, commits, and PUSHES recipe changes. It
does NOT bring the recipe up on the sparks, rebuild images, restart containers, or
verify a live endpoint — that is the `bring-up-spark-recipe` skill's job, run AFTER
the update lands. Do not cross over: stop at "pushed an update branch".

## Overview / core principle

The repo's AGENTS.md (auto-loaded for this repo) carries the hard-won cluster
constraints and the detailed per-source query recipes. **Use AGENTS.md as the source
of truth for source URLs, Discourse API paths, GitHub endpoints, and image/HF
revision checks — this skill is the workflow that consumes them.** Every recipe's
README also documents its own invariants and changelog; read it before touching
anything.

Adoption rule — every candidate change must:
1. apply to our pinned image/revision (not a different image),
2. NOT contradict the recipe's validated invariants (start order worker→head,
   port 4000, offline serving, GMU/KV-pin/dtype/backend wiring),
3. be backed by a measured claim, not "in theory."

Reject anything that touches the fragile cross-recipe knobs casually
(`gpu_memory_utilization`, `max_num_seqs`, KV dtype, backend names are per-image and
NOT transferable between recipes — see AGENTS.md).

## The review loop

### 1. Snapshot current state

```bash
git log --oneline -8                      # recent commits on this branch
cat <recipe>/upstream/VENDORED-AT.md 2>/dev/null   # tonyd2wild pin (if vendored)
# note which overlays / fix files the recipe carries (encoding_dsv4.py, deepseek_v4_wrapper.py,
# detokenizer.py, deepseekv32_tool_parser.py, overlay-dflash2/*, patches/*)
```

### 2. Forum pass (NVIDIA Developer Forum) — highest-value source

Run the Discourse JSON API searches (no auth; send a browser-ish User-Agent). The
category IDs and thread IDs that matter are in AGENTS.md's "review how to query each
source" table. Generic sweep:

```python
# latest topics in the DGX Spark boards + full-text search
GET https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10/721.json?order=activity
GET https://forums.developer.nvidia.com/search.json?q=<recipe-model>%20after:<last-review-date>
GET https://forums.developer.nvidia.com/search.json?q=dspark%20category:721%20order:latest
```

Search terms per recipe (from AGENTS.md + the recipe's own README/research.md):
`deepseek v4 flash`, `dspark`, `b12x`, `nvfp4`, `sparkrun`, `mimo v2.5`,
`dflash`, plus the recipe's model name.

**Mandatory: a GENERAL search every run — do not limit the pass to the tracked
threads.** Known thread IDs (AGENTS.md table, recipe README) are for *catching up
on replies*; new fixes and regressions routinely land in threads whose titles
never mention the model. Every review MUST also:

1. Sweep the newest topics in the DGX Spark boards (categories `721` main +
   `723` projects) since the last review date — open every thread not previously
   reviewed and skim for recipe relevance:
   `GET .../721.json?order=created` / `GET .../723.json?order=created`.
2. Run multiple full-text searches with varied terms (model name, method names,
   backend names, error signatures) and an `after:<last-review-date>` filter.
3. Check forum-wide latest posts (`GET /posts.json`) for replies to any tracked
   thread.

Pull each promising thread's newest posts (paginate `/t/<id>.json` until the
posts are older than the last check; each post has `post_number`/`created_at`).
Look for: new image tags, new patches, new tuning knobs with measured numbers,
regression reports that could hit this cluster's config.

### 3. Upstream GitHub pass

- **Primary upstream repo** (tonyd2wild recipe: `tonyd2wild/DeepSeek-v4-Flash-...`;
  others referenced in the recipe README). List open PRs/issues + recent commits via
  the GitHub API; diff anything touching the recipe's core (dspark, encoder, loader,
  batching, drafter).
- **Sibling repos by the same author** — several times a missing/fixed module was
  found in a *different* repo of the same author. Check his DGX-Spark-serving family
  (AGENTS.md lists them).
- Compare a vendored `upstream/` against its pinned commit / current HEAD.

### 4. Image & model checkpoint pass

- **Container image tags** (e.g. aiden: `aidendle94/...` on Docker Hub tags API).
- **HF model revision** — compare `lastModified` / `siblings` on the recipe's pinned
  checkpoint; a newer encoder/template may exist.
- **vLLM upstream** — only adopt if the newer runtime boots on GB10/SM120 with the
  custom backends (stock vLLM often cannot — see AGENTS.md).

### 5. Evaluate before adopting

Filter candidates through the adoption rule above. For each accepted change, follow
the repo's surgical-edit conventions (AGENTS.md): recipe-level changes go in the
recipe dir (NOT `upstream/`); overlays copied between recipes stay in sync; respect
the `/opt/venv` vs `/opt/env` image-path split; keep start-order / port-4000 /
offline-serving / KV-pin semantics intact.

## Branch, commit, push (the deliverable)

Repository convention: `main` = stable recipes only. In-progress recipe work goes on
branches; the remote keeps merged feature branches (e.g. `glm-v53-flash`).

- If already on an update branch (one whose name / recent commits indicate active
  recipe work), stay on it. Otherwise create one:
  `git checkout -b <recipe>-<topic-or-date>-updates` (existing examples:
  `aiden-aug15-updates`, `tonyd2wild-upstream-update`).
- Commit with a descriptive one-liner and a `(#N)` PR number when it exists, matching
  repo history style. For notable tuning work, add a dated changelog block to the
  recipe README (the tonyd2wild "audit trail" section is the template): what changed,
  measured before/after, gotchas hit.
- Push: `git push -u origin <branch>`.
- Report the branch name + digest of what was adopted and why. Stop there — do NOT
  launch the recipe.

## References the skill leans on

- `AGENTS.md` in this repo — source URLs, Discourse API recipes, category/thread IDs,
  evaluation gates, failure-mode table, git conventions. Read it (auto-loaded).
- The recipe's own `README.md` and `research.md` — per-recipe invariants + audit trail.

## Common mistakes

- Editing `upstream/` vendored files — the parent recipe dir is the customization
  layer; `upstream/` is a pinned mirror.
- Committing directly to `main` — recipe changes go on a topic branch.
- Adopting a change tuned for a different image / GMU / KV dtype — invariants are
  recipe-specific.
- Adopting an "in theory" improvement that touches fragile cross-recipe knobs.
- Crossing into bring-up-spark-recipe territory (launching/restarting the cluster).
- Failing to update the OTHER recipe copies of a shared fix file (encoder/tool-arg
  files must stay in sync across all recipes that carry them).
