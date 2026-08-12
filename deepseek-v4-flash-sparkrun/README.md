# deepseek-v4-flash-sparkrun

DeepSeek-V4-Flash-0731 served on a **2-node DGX Spark cluster**, built and
launched **entirely through `sparkrun`** (no manual eugr wrapper scripts).

This supersedes the old `dsv4f-sparkrun-v6` branch (removed). It is the
sparkrun-native port of the working `deepseek-v4-flash-eugr` recipe.

## How it works (build + run all via sparkrun)

Sparkrun's **eugr builder** owns the whole lifecycle — it does *not* require a
pre-built image:

1. `sparkrun` reads this recipe, sees `build_args: [--exp-b12x]` + `mods:`, and
   routes to the eugr builder.
2. The builder runs eugr's `build-and-copy.sh -t vllm-node-b12x --exp-b12x`
   (the **same full source build** used by the manual eugr workflow; ~20–40 min
   on first build). If `vllm-node-b12x:latest` already exists on the cluster it
   is reused.
3. `mods:` are resolved **adjacent to this file** (the `mods/` dir here) and
   applied inside the containers before `vllm serve`.
4. sparkrun distributes the image + model to both nodes and launches the
   cluster, detaching by default.

## Run (on the leader node)

```bash
# from the spark-recipes repo
sparkrun run deepseek-v4-flash-sparkrun/deepseek-v4-flash-sparkrun.yaml
```

Sparkrun detaches by default (cluster keeps serving after the command returns).
To watch logs:

```bash
sparkrun logs deepseek-v4-flash-sparkrun
```

Stop gracefully:

```bash
sparkrun stop deepseek-v4-flash-sparkrun
```

Status:

```bash
sparkrun status
```

### Useful flags

```bash
# Force a rebuild even if the image exists
sparkrun run deepseek-v4-flash-sparkrun.yaml --force-build

# Target specific hosts / cluster
sparkrun run deepseek-v4-flash-sparkrun.yaml --cluster spark
# or
sparkrun run deepseek-v4-flash-sparkrun.yaml -H 192.168.0.170,192.168.0.171

# Dry run (see exactly what would happen, incl. generated serve command)
sparkrun run deepseek-v4-flash-sparkrun.yaml -n
```

## What's in this directory

```
deepseek-v4-flash-sparkrun/
├── README.md
├── deepseek-v4-flash-sparkrun.yaml   # the sparkrun recipe
└── mods/
    ├── dsv4-reasoning-effort-fix/run.sh            # reasoning-effort fix (low/high/max)
    └── instanttensor-hybrid-draft-loader/          # hybrid draft loader
        ├── README.md
        ├── patch_model_loader.py
        └── run.sh
```

The recipe is identical in *intent* to `deepseek-v4-flash-eugr` (same container
`vllm-node-b12x`, same `--exp-b12x` build, same mods, same B12X backend, same
1M-context DSpark config). The only change is the `command` templating, adapted
to sparkrun's substitution rules:

- JSON-valued flags (`--reasoning-config`, `--override-generation-config`,
  `--compilation-config`, `--speculative-config`) are **single `defaults` values**
  referenced as `{key}` — sparkrun substitutes them in one piece, unlike eugr's
  `{{...}}` str.format escapes.
- `--speculative-config` JSON matches the eugr recipe (DSpark, 5 draft tokens,
  `B12X_MLA_SPARSE`).

## Serve settings

| knob | value |
|---|---|
| model | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| served name | `deepseek-v4-flash` |
| container | `vllm-node-b12x` (build arg `--exp-b12x`) |
| port | 4000 |
| TP | 2 (across 2 nodes) |
| max context | 1M (`auto`) |
| KV cache | fp8, block 256 |
| `max_num_seqs` | 6 |
| spec tokens | 5 (DSpark) |
| load format | `instanttensor` |
| backends | B12X MoE / linear / `B12X_MLA_SPARSE` attention |
| HF | offline (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`) |

## Verify

```bash
curl -sS http://127.0.0.1:4000/v1/models | jq .
curl -sS http://127.0.0.1:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash",
       "messages":[{"role":"user","content":"Say hi"}],
       "thinking":true,"reasoning_effort":"high"}' \
  | jq '.choices[0].message'
```

## Reference

- [DSpark forum thread](https://forums.developer.nvidia.com/t/instructions-for-running-deepseek-v4-flash-with-dspark-using-eugrs-repo/376220)
  (post 18 = bernisse's solution)
- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) (branch `b12x`)
- `sparkrun` docs: `sparkrun --help`, `sparkrun run --help`
