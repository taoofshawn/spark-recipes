# glm-v53-flash-miaai — GLM-5.3-Flash EXL3 4bpw (2x DGX Spark)

**Adoption** of [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
(vendored at upstream HEAD, 2026-09-01) into this repo's docker-compose
conventions. The EXL3 lane: **4-bit-weight EXL3/TR3** checkpoint (176 GB,
~54% of the FP8 bytes) served through a prebuilt GHCR image with the overlay
baked — no local build, no `--moe-backend marlin`.

## What it serves

- **Weights:** `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` (mirror of
  `brandonmusic/GLM-5.3-Flash-tr3-4bpw` snapshot `5ab363a8…`, ~164 GiB,
  120 shards) — EXL3/TR3 uniform-K4, revision `25a44fd…`
- **Drafter:** DFlash2 `incoai/GLM-5.3-Flash-DFlash2`, k=7, draft TP=2
- **Served name:** `GLM-5.3-Flash-EXL3` | **Port:** 4000
- **Context:** 1M | **KV:** `fp8` → packed `fp8_ds_mla`, pool **1,754,237
  tokens** (1.75× @1M) @ GMU **0.87**
- **Vision:** native image + video (`--limit-mm-per-prompt {"image":10,"video":1}`),
  `--skip-mm-profiling`
- **Quality:** KLD 0.0246 vs official FP8 0.0206 (closer than NVFP4's 0.0605
  on the same harness) — see the upstream README's KLD table.

## Why EXL3 (vs the repo's NVFP4 glm recipe)

| | `glm-v53-flash/` (ours, NVFP4) | this recipe (EXL3 4bpw) |
|---|---|---|
| weights | NVFP4 W4A4 (compressed-tensors, 184 GiB) | EXL3 4bpw (176 GB) |
| fidelity | farther from FP8 (W4A4) | measured near FP8 parity (KLD) |
| context | 262,144 (TP2 ceiling) | 1M on 2 nodes |
| image | `ghcr.io/miaai-lab/…:exl3` (pulled) | — |
| runtime | locally built image | prebuilt GHCR, overlay baked |

## Deploy

```bash
# 0) one-time on BOTH nodes (TP=2 reads weights + drafter on every rank)
hf download Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw --revision 25a44fdbf16862a46b7cc9921142c6c81350af2f
hf download incoai/GLM-5.3-Flash-DFlash2

# 1) worker (node 1) FIRST, then leader ~35 s later.
#    worker: docker compose --env-file .env --env-file .env.node1 up -d
#    leader: docker compose --env-file .env --env-file .env.node0 up -d

# 2) verify
curl http://127.0.0.1:4000/v1/models     # -> "id":"GLM-5.3-Flash-EXL3", max_model_len 1000000
```

## Cluster deviations from upstream

| knob | upstream | here | why |
|---|---|---|---|
| `PORT` | 8888 | 4000 | cluster convention |
| `MASTER_ADDR` | 10.0.0.1 | 192.168.0.170 | wired RoCE IPs |
| `MASTER_PORT` | 29521 | 29521 (unchanged) | no collision on 4000/25000 |
| JIT caches | host paths in `.env` | node-local `/vllm-cache` | repo convention |
| start order | `./start.sh` orchestrator | compose `.env.node0/1` | repo convention; worker first |
| `NCCL_IB_HCA` / socket ifs | per-node CX7 pins | `${IB_PORTS}` / `${ETH_IF},${ETH_IF2}` | repo convention (same NICs on this cluster) |

Unchanged from upstream's validated profile: GMU 0.87, MNBT 7168, DFlash2
k=7/draft-TP2, `EXL3_FAT_KERNEL=1`, `GLM53_*` patch knobs, `SKIP_MM_PROFILING=1`.

## Notes

- `HF_HOME` is `/root/.cache/huggingface` in this image (mount lands there);
  the DS4/Anemll image uses `/cache/huggingface` instead.
- The GHCR image carries a KV-accounting inconsistency that kills 1M boots;
  `hotfix_kv_check_glm5.py` patches it at boot. See `research.md` for the
  full analysis, all deployment problems/workarounds, and what to re-visit
  on future updates (incl. Entrpi's GLM/vLLM-fork repos).
- Upstream also supports a rebuild-from-source path (`BUILD=1 ./start.sh`);
  not vendored here — the prebuilt image is the default.

## References

- [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
  (README carries the full constraint matrix + benchmark tables)
- Weights: [Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw)
  / [brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)

## GPU contention

Serves on all 2 GPUs per node. Tear down any other model container before
starting. One recipe at a time.
