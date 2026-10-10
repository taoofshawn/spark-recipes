---
name: adopt-recipe
description: >-
  Use when creating a brand-new recipe (a new serving lane) in the spark-recipes repo
  from an NVIDIA forum thread and/or a GitHub repo — reviewing the linked sources and
  generating a new recipe directory in the 4x-switchless house style on a fresh branch
  for review. Triggers: "adopt <link>", "create a recipe for <model>", "new recipe
  from <thread/repo>", "port <author>'s recipe", "add <model> to spark-recipes". Does
  NOT update an existing recipe (update-recipe) and does NOT bring anything up on the
  sparks (bring-up-spark-recipe).
---

# Adopt a Recipe (4x-switchless house style)

Create a NEW `spark-recipes` serving lane from an external source — an NVIDIA DGX
Spark forum thread and/or a GitHub repo: review the links, extract the serving stack,
and generate a recipe directory in this repo's 4x-switchless house style, on a new
branch, left uncommitted for the user to review.

All 2-node recipes are archived under `.archived/` — do NOT model new work on them
(their docker-compose pair flows, `.env.node0/.env.node1`, and 192.168.0.x wiring are
the old cluster). The live reference for house style is the pair of 4x recipes:
`glm-v53-flash-4x-noswitch/` (FP8) and `glm-v53-flash-nvfp4-4x-noswitch/` (NVFP4).

## Scope boundary

This skill CREATES a new recipe directory and stops there. It does NOT update an
existing recipe (`update-recipe`), bring anything up on the sparks
(`bring-up-spark-recipe`), or commit/push/PR/merge. The deliverable is an untracked
directory on a fresh topic branch.

If the target looks like a variant of an existing 4x recipe (same model, different
quant/lane), CHECK WITH THE USER first — they may want it as an update instead.

## The loop

### 1. Review the target links (before writing anything)

Mine the sources the way AGENTS.md's "sources" section prescribes. Forum = Discourse
JSON API (no auth; browser-ish User-Agent; paginate `/t/<id>.json`). GitHub = raw
README/docs + the API for commits/PRs/issues. Extract — each item WITH a receipt
(post number, file path, commit SHA, log line):

| extract | feeds |
|---|---|
| model repo + pinned revision + quant method (+ license) | weight materialization; HF-cache invariant |
| serving image + how it is built (Dockerfile, digest pins) | the image build step |
| spec-decode method + k + drafter checkpoint + revision (license!) | the spec config |
| KV dtype / pin / pool, GPU mem util, max_num_seqs, batched tokens, block size | the tuning rows (per-image, from the source) |
| MoE/attention backend + why | boot-fatal on the wrong value |
| transport requirements (NCCL contract, HCAs, patches) | compatibility with the shared four-PF contract |
| patches / overlays / mods the stack carries | what must be vendored + documented |
| measured claims with their protocol | receipts for research.md; never adopt "in theory" numbers |
| boot markers + known failure modes | the README boot-marker section |

### 2. Decide the shape (4x-switchless house style)

A 4x lane is: a vendored/pinned upstream checkout (`upstream/`, pristine, with a
`VENDORED-AT.md` recording pin + every site modification and its re-apply recipe) +
a `site/` layer OUTSIDE the vendored tree (cluster env, netplan, weight
materialization, per-lance additions) + a serving profile derived from upstream's with
documented strip lists + a `README.md` (ops only) + `research.md` (provenance +
changelog). See `glm-v53-flash-nvfp4-4x-noswitch/` for the cleanest example (knapcio
vendored @ a pin, site env + profile + materializer) and `glm-v53-flash-4x-noswitch/`
for the pinned-checkout + site-mods variant.

Non-negotiables for any new lane:

- **TP4 across all four nodes**, switchless ring transport, four-PF NCCL contract
  (shared lib `~/.local/lib/nccl-patched-4pf`; a lane needing a different NCCL build
  stages its own dir and documents why — never swap the shared one in place).
- **Serving contract**: port 8000, `SERVED_MODEL_NAME` = lowercase-hyphenated model
  name; the model-name proxy's `BACKEND_MODEL` must match (see `model-name-proxy/`).
  If the name differs, the proxy change is part of the recipe.
- **Offline serving**: weights + drafter materialized from the HF cache with pinned
  revisions on ALL FOUR nodes (`site/materialize-weights.sh` pattern — hardlink
  snapshots; HF snapshot symlinks break inside container mounts).
- **`HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`** semantics: no runtime downloads.
- **LANE-SWITCHING.md integration**: distinct container names + overlay roots from the
  existing lanes; add the new lane's column to that runbook.
- **GPU contention**: the new lane must document the teardown requirement for the
  others.

### 3. Name it

`<model>-<distinguisher>` where the distinguisher is the author, quant, or defining
feature (existing pattern: `glm-v53-flash-4x-noswitch`, `...-nvfp4-4x-noswitch`).
Check for collisions; record alternatives in research.md.

### 4. Branch, then create the directory

`main` = stable only; direct pushes are blocked. The per-lane working branch carries
the active recipes — branch from it:

    git checkout glm-v53-flash-nvfp4-4x-noswitch && git pull
    git checkout -b adopt-<recipe-name>
    mkdir <recipe-name>

### 5. Generate, then finish uncommitted — and stop

Model the files on the two live recipes — copy STRUCTURE, not values. Every
non-cluster value traces to a reviewed source with a receipt; every cluster value
(nodes, ring addressing, HCAs, netplan, NCCL dirs) is THIS cluster's, from AGENTS.md.
README = active-running doc only (provisioning, build, deploy, verify, boot markers,
tuning rows); history/TODO/watch material goes to research.md.

    git status    # recipe dir untracked on adopt-<recipe-name>; nothing staged

Report: branch, directory, what was adopted from where (with receipts), deviations
forced by this cluster. Do NOT commit, push, PR, merge, build, or launch.

## Common mistakes

- Modeling on the archived 2-node recipes (docker-compose pairs, `.env.nodeN`,
  192.168.0.x crossover IPs — that era is gone).
- Committing/pushing the adoption — the deliverable is UNCOMMITTED.
- Copying upstream cluster wiring as-is (HCAs, IPs, netplan) — this ring's contract is
  in AGENTS.md; upstream values are receipts for the model stack only.
- Adopting measured claims without receipts, or "in theory" improvements that touch the
  fragile shared transport (NCCL contract, netplan).
- Inventing hardware or a different topology. Four fixed nodes, switchless ring.
- Writing changelog/watch material into README.md instead of research.md.
