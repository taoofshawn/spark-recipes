# Archived skills + agent docs (2026-10-10)

Snapshot of `.agents/skills/` and the root `AGENTS.md` as they were BEFORE the
4x-switchless-only cleanup (owner decision: only the 4x-noswitch lanes stay active;
all 2-node recipes archived). These copies retain the 2-node knowledge — docker-compose
pair flows, the 192.168.0.x crossover era, DeepSeek/MiMo/Qwen recipe specifics, sparkrun,
per-recipe tuning tables — so nothing was lost, it just no longer loads automatically.

- `*/SKILL.md` — the five skills as they were (2-node flows, old cluster topology).
- `AGENTS.pre-4x-cleanup-2026-10-10.md` — the full pre-cleanup AGENTS.md (DeepSeek
  recipe cluster, forum source tables, image/checkpoint pins for archived recipes).
- `bench-recipe-update/scripts/bench_recipe.py` — the generic bench harness (still
  used by the FP8 4x lane; the live skill carries the current version).

The LIVE versions under `.agents/skills/` and the root `AGENTS.md` are trimmed to the
4x-switchless deployments only.
