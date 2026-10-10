---
name: update-recipe
description: >-
  Use when reviewing one of the two active 4x-switchless recipes (glm-v53-flash-4x-noswitch
  FP8, glm-v53-flash-nvfp4-4x-noswitch NVFP4) against its upstream sources for updates,
  fixes, or improvements — checking the NVIDIA forum, the upstream GitHub repos
  (knapcio, jnardiello, tonyd2wild), HF model revisions, and other sources — then
  committing and pushing on the working branch. Triggers: "review recipe <name> for
  updates", "check <recipe> upstream", "is there a new fix for <model>", "update
  <recipe> from upstream", "latest commits for <recipe>". Does NOT bring the recipe up
  on the sparks (that is bring-up-spark-recipe).
---

# Recipe Update (4x-switchless lanes)

Research pass → evaluate → commit → push for one of the two active GLM-5.3-Flash lanes
on the 4-node switchless ring. Find anything new in the upstream sources, decide what's
worth adopting on THIS cluster, and land it on the working branch — while NEVER touching
the running cluster.

## Scope boundary

This skill researches, evaluates, commits, and PUSHES recipe changes. It does NOT bring
the recipe up on the sparks, restart containers, or verify a live endpoint — that is
`bring-up-spark-recipe`. Do not cross over: stop at "pushed".

## The two active recipes and their upstreams

| recipe | upstream | pin (verify at review time) |
|---|---|---|
| `glm-v53-flash-nvfp4-4x-noswitch` | `github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4` (vendored, detached — see its `upstream/VENDORED-AT.md`) | `770d115` (2026-09-29 tip at last review) |
| `glm-v53-flash-4x-noswitch` | `github.com/jnardiello/GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless` (pinned checkout, its `upstream/VENDORED-AT.md` carries site mods to re-apply on refresh) | `ed365a6` (E36) |

Sibling repos worth checking (the GLM serving family): `tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark`,
`tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark`, plus kindlingai/eugr/technigma
stacks for cross-stack ideas. Model checkpoints: `nvidia/GLM-5.3-Flash-NVFP4` (pinned
`09b04e5e`), drafter `incoai/GLM-5.3-Flash-DFlash2` (pinned `bf582e4e`, **CC BY-NC-ND** —
no derivative re-encodes). Check HF `lastModified`/siblings for new revisions.

## Review loop

1. **Snapshot current state:** `git log --oneline -8` on the working branch; each
   recipe's `upstream/VENDORED-AT.md` (pin + site-mods list); note which site mods /
   derived-file layers the recipe carries.
2. **Forum pass (Discourse JSON API, no auth, browser-ish User-Agent):**
   - Tracked threads: `381350` (GLM-5.3-Flash main, 700+ posts), `382459` (jnardiello
     switchless), `382939` (nvidia-NVFP4 optimization), `385075` (kindling 3x) — page
     `/t/<id>.json` until posts are older than the last review date in research.md.
   - General sweep (mandatory every run): newest topics in categories `721` (main) and
     `723` (projects) since the last review, plus varied searches (`glm 5.3 flash`,
     `dflash`, `nvfp4`, `switchless`, `after:<last-review-date>`). Recipe-relevant info
     routinely lands in threads whose titles never mention the model.
3. **GitHub pass:** `git ls-remote` + commit lists per upstream; open PRs/issues via the
   API; diff anything touching transport (NCCL/switchless), scheduler, launcher, or the
   serving profile. For the FP8 recipe, classify upstream changes against the site-mods
   list in its `VENDORED-AT.md` (site paths, the NCCL SHA re-pin, the gid-checker fix,
   the netplan P2 extension — re-apply on refresh).
4. **Image/checkpoint pass:** both lanes build locally (`glm53-roce:v11-b58f34ea` from
   knapcio's `Dockerfile.roce`; the FP8 digest-pinned image per its build-record) — no
   Docker Hub tracking. HF checkpoint revisions per the table above.
5. **Evaluate before adopting.** Every candidate must (a) apply to our pinned
   image/revision, (b) not contradict the lane's validated invariants (start order,
   port 8000, offline serving, KV pins, the four-PF NCCL contract, `FABRIC_P2_VIEWS=1`),
   (c) be backed by a measured claim, not "in theory". Reject casual changes to fragile
   knobs (KV_BYTES/GPU_UTIL/max_num_seqs/batched-tokens are per-lane, tuned, and not
   transferable). Record rejections with reasons in research.md — rejections are knowledge.
6. **Adopt surgically:** recipe-level changes go in the recipe dir / its site layer
   (never inside vendored `upstream/` except as documented VENDORED-AT site mods);
   upstream changes to `profiles/current.env` need a stock-compat review of new switches
   against the NVFP4 strip list (`stock-nvfp4.env`) before inheriting.
7. **Document:** dated changelog entry in the lane's `research.md` (what changed,
   measured before/after, gotchas). README = active ops content only.

## Branch conventions

Work happens on the per-lane working branch (`glm-v53-flash-nvfp4-4x-noswitch` — it
carries BOTH recipes since the 2026-10-10 reorg). Never commit to `main` directly;
branch protection blocks it. Changes reach the nodes via git pull + the derived-file
mirror steps in `LANE-SWITCHING.md`.
