# qwen-3.8-flash-next-tonyd2wild — Qwen3.8-Flash-Next NVFP4 (2× DGX Spark)

**Adoption** of the TP2 lane from
[forum 382476](https://forums.developer.nvidia.com/t/qwen3-8-flash-next-on-1-2-and-4-dgx-sparks-with-nvidias-official-nvfp4-quant-64-tok-s-peak-single-stream/382476)
(ToNYD2WiLD, post 1 + the TP2 sections) into this repo's docker-compose
conventions: serve NVIDIA's official NVFP4 quant
[`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
(125B-A6B hybrid MoE + 51B n-gram PLE table + 4B MTP head; used byte for byte,
no requantizing) on the **stock vLLM nightly** `8a728663` with bind-mounted
overlays, TP=2 across the two Sparks. Upstream repo:
[tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark)
(`single-spark-vllm-tp1/launch/qwen38fn-nvidia-tp2.sh` = the reference
launcher; HEAD `6ad1c8f…` 2026-09-07).

**This is a NEW model family** (`qwen4_exp`) — not a DeepSeek/GLM variant. The
overlays in `patches/` are REQUIRED: without them fp8 KV and the MTP head fail
to load on this nightly.

## What it serves

| | |
|---|---|
| Model | Qwen3.8-Flash-Next, 125B total / 6B active hybrid MoE + 51B-row n-gram PLE table (47.68 GiB FP8) + 4B MTP head + vision tower |
| Quant | NVIDIA ModelOpt NVFP4 (experts W4A16 NVFP4, MTP experts W8 FP8 block-scaled, n-gram embedding FP8; lm_head/embeddings/attention/shared experts unquantized) |
| Context | **262,144 — the model's native ceiling** (`text_config.max_position_embeddings`), not a serving choice |
| Served name | `qwen3.8-flash-next` \| Port: **8000** (repo convention; model-name proxy fronts :4000) |
| Spec decoding | native **MTP3** (`{"method":"mtp","num_speculative_tokens":3}`); MTP4 measured trailing at every load |
| KV | `fp8_e4m3` via the #54846 QSA overlay (stock vLLM refuses fp8 KV on this model) |
| Speed (TP2 SPEED receipt, post 1) | median **53.7** / prose 37.2 / peak 63.7 tok/s single stream; **97.9 aggregate @6**; TTFT 180 ms; cold prefill 2,784 tok/s @28K; KV pool **1.97M tokens** (7 full 262K contexts) |
| Thinking | OFF server-side (upstream default; `THINKING` env flips it) |
| Tool calling | `qwen3_xml` parser (upstream default; `qwen3_coder` alternative) |

## Profiles: where the 47.7 GB n-gram table lives

The one deployer decision (upstream "SPEED vs CONTEXT", 2026-09-05 evening):

| Profile | `.env` | What you get | Cost |
|---|---|---|---|
| **SPEED (default)** | defaults (`PLE_MODE=none GRAPHS=nocompile`) | table resident, half/rank; 53.7 median, 1.97M pool | 7 full 262K contexts |
| **CONTEXT** | `PLE_MODE=mmap GRAPHS=piecewise` | table on NVMe, rows gathered per step; KV pool **5.87M** (22 contexts) | 35.8 median, 65.5 agg @6 |

`PLE_MODE=resident|staged` are experimental upstream (resident measured 8–15
tok/s) — the compose refuses them. Do NOT set `DRAFT_VOCAB` unless A/B'ing
(prose +10%, TTFT slightly worse under load; not default upstream).

## Deploy

Pre-launch ritual (per node — GB10 unified memory swap-wedges instead of
OOMing; run before `up`, as with every recipe in this repo):
```bash
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
```

```bash
# 0) one-time provisioning (BOTH nodes — each rank reads its local copy):
python3 -m venv /tmp/hfvenv && /tmp/hfvenv/bin/pip install -q -U huggingface_hub
/tmp/hfvenv/bin/hf download nvidia/Qwen3.8-Flash-Next-NVFP4 \
    --revision fc694b54fb0174e0913e6adf86691ef85a4ead47
# Then set .env HF_CACHE to your default HF cache location (already the
# default in the shipped .env); the boot block resolves
#   hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/<rev>
# inside the container at the default cache location.

# 1) worker (rank 1) FIRST, leader ~30 s later:
docker compose --env-file .env --env-file .env.node1 up -d
docker compose --env-file .env --env-file .env.node0 up -d

# 2) verify (leader):
curl -s http://127.0.0.1:8000/v1/models   # "id":"qwen3.8-flash-next"
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health   # 200
```

Boot: weights ~5–6 min (TP2 splits the load) + MTP draft head + graphs —
upstream measured ~10 min launch-to-serve at TP2.

## Boot markers (leader log)

```bash
docker logs qwen38fn-nvfp4 2>&1 | grep -F "GID auto-detect"
#   -> "GID auto-detect: NCCL_IB_GID_INDEX=N" (not the error dump)
docker logs qwen38fn-nvfp4 2>&1 | grep -F "speculative_config"
#   -> method='mtp', num_speculative_tokens=3
# KV pool receipt (SPEED): ~1.97M tokens at GMU 0.70
```

## Gotchas (all upstream-measured; do not "fix" blind)

- **NEVER pair the 4096-token chunk with torch.compile on** — measured 8–15
  tok/s decode at TP2. SPEED always uses `{"mode":0,"cudagraph_mode":
  "FULL_DECODE_ONLY"}` (= `GRAPHS=nocompile`). CONTEXT (mmap + piecewise)
  keeps the default chunk, which is why mmap must switch GRAPHS.
- **Compile duplicates the resident PLE table** (~24 GB/rank at TP2) —
  starved/rebooted three Sparks on 2026-09-05 before the cause was found
  (gau-nernst's open vLLM PR #55272 removes compile for this reason).
- **GMU 0.70 is deliberate.** Head sits near the ~10 GB MemAvailable danger
  line; 0.85 drifts into swap after a day, 0.875 OOM-killed on a 300K prefill
  (blazux). CONTEXT (mmap) boots at 0.80 upstream — 5.87M pool, ~10.5 GB
  available; `GMU=0.75` is the documented stable fallback.
- **Prefix caching OFF** until vLLM #54173 (GDN prefix-cache crash) is
  verified fixed in this nightly. m0l0 (post 14) measured it working when
  enabled (TTFT 3–4 s @ ~200K input) — flip `PREFIX_CACHE=1` to test, at your
  own crash risk.
- **MTP head load needs the modelopt.py overlay** (draft-local layer index +
  FP8_BLOCK_SCALES branch + `FP8_PB_WO` alias). The pinned checkpoint stores
  exactly one MTP expert layer with `quant_algo=FP8_PB_WO`
  (`mtp.layers.0.mlp.experts`, group 128; vLLM-side `mtp.layers.48.*`); that
  branch routes it to the generic block-FP8 MoE method. Vendored 2026-09-08
  (forum 382476 posts 15/17/22; same fix as MiaAI-Lab #39).
- **Rejected-at-TP2 knobs** (measured worse upstream, kept off): expert
  parallel (-4%), `index_share_for_mtp_iteration` (+3% median, prose -6%),
  `--async-scheduling` (-3%), NCCL 8 channels (-3%).
- **No custom image**: the stock nightly is used as-is; the `patches/`
  overlays bind-mount in. Any vLLM-package path drift inside a future nightly
  breaks the mounts loudly at boot (per-file `:ro` mounts fail on missing
  targets) — that is the intended fail-closed behavior; re-vendor the overlays
  from upstream when adopting a new nightly.
- **Driver**: this cluster is on 580.173.02; don't upgrade blindly (repo-wide
  watch item).

## Deviations from upstream

| | upstream TP2 launcher | here | why |
|---|---|---|---|
| mechanism | `docker run` shell script | docker-compose (`.env`/`.env.node0/1`) | repo convention |
| port | 8000 | 8000 | same; repo convention (proxy fronts :4000) |
| rendezvous | 29531, IPs 192.168.192.x (their kit) | 29531, `MASTER_ADDR=192.168.0.170` | this cluster's RoCE rail |
| GID | hardcoded `3` | sysfs auto-detect in the command block | GID renumbers across reboots |
| NICs | `rocep1s0f0` only | both (`IB_PORTS`) | this cluster's validated set |
| image | tag-pinned nightly `8a728663` | same tag in `.env` — **digest-pin before production** | upstream ships no digest |
| cache | `/var/tmp/qwen38fn-vllm-cache` | `CACHE_HOST_PATH` (.env) | repo convention |
| thinking | off (hardcoded) | `THINKING` env (default false = upstream) | repo parity |
| sampling | none shipped | explicit `TEMPERATURE=1.0`/`TOP_P=0.95` pins | repo parity (no-ops vs upstream) |
| PLE modes | none/mmap/resident/staged | none/mmap only (refuse resident/staged) | experimental upstream, measured slow |
| overlays | `PATCH_DIR` bind-mounts at run | `patches/` bind-mounts in compose | same mechanism, repo layout |

## References

- Forum: [382476 — Qwen3.8-Flash-Next on 1/2/4 DGX Sparks with NVIDIA's official NVFP4 quant](https://forums.developer.nvidia.com/t/qwen3-8-flash-next-on-1-2-and-4-dgx-sparks-with-nvidias-official-nvfp4-quant-64-tok-s-peak-single-stream/382476)
  (post 1 = TP2 SPEED receipts; post 3 = clint25 cross-engine A/B: vLLM 86 vs
  SGLang 90 tool-eval; posts 15–17 = MTP-load errors → the modelopt overlay)
- Upstream repo: [tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark)
  (Apache-2.0; `single-spark-vllm-tp1/patch/` = the vendored overlays +
  diffs; `PROVENANCE.md` in `patches/upstream-overlays/` documents the
  upstream-PR vs own-fix split per file with sha256s)
- Weights: [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
  (rev `fc694b54…`, 2026-09-05, 124 GB) — base `Qwen/Qwen3.8-Flash-Next`
- Upstream vLLM PRs the overlays carry: [#55375](https://github.com/vllm-project/vllm/pull/55375)
  (merged 09-05), [#54846](https://github.com/vllm-project/vllm/pull/54846)
  (open); gotchas reference #54173, #54125, #55272
- Sibling recipes: [T-Klug/Qwen3.8-Flash-Next-NVFP4-2xDGX-Spark-vLLM](https://github.com/T-Klug/Qwen3.8-Flash-Next-NVFP4-2xDGX-Spark-vLLM),
  [sfxnz/Qwen3.8-Flash-Next-NVFP4-vLLM-2x-DGX-Spark](https://github.com/sfxnz/Qwen3.8-Flash-Next-NVFP4-vLLM-2x-DGX-Spark)
  (independent modelopt fixes), [MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks)

## GPU contention

Serves on all 2 GPUs per node. Tear down any other model container (DS4 vision,
GLM lanes, etc.) on BOTH nodes before starting. One recipe at a time. The
model-name proxy (`model-name-proxy/.env`) needs `BACKEND_MODEL=qwen3.8-flash-next`
when this recipe is active.
