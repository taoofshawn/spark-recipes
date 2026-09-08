# mtp3-pmu128 — vendored upstream lane (florianbrede-ayet)

Verbatim copy of
[`florianbrede-ayet/spark-recipes/tp2_glm53flash_autoround_mtp3_pmu128`](https://github.com/florianbrede-ayet/spark-recipes/tree/main/tp2_glm53flash_autoround_mtp3_pmu128)
(tree pinned 2026-09-07; forum thread
[382632](https://forums.developer.nvidia.com/t/glm-5-3-flash-intel-autoquant-w4a16-tp2-mtp3-concurrent-agentic-use/382632)),
with two files dropped because this recipe does not use them:

- `launch.sh` — florianbrede's own launcher; the parent recipe's
  `docker-compose.yml` replaces it with the same flags in repo conventions.
- `validate.sh` — florianbrede's offline/network validation harness; kept
  upstream, not vendored.

`SHA256SUMS` here covers only the vendored files (upstream's list minus the two
dropped ones); verify with `sha256sum -c SHA256SUMS` inside this directory.

## What this lane is

Same Intel W4A16 quant, same digest-pinned base image
(`ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef6…`, the sm121-v11-dflash2
image the parent recipe runs) — but:

- **Native MTP3** (`--speculative-config
  '{"method":"mtp","num_speculative_tokens":3,"disable_eagle_block_drop":true}'`)
  instead of the DFlash2 k=7 drafter. No drafter checkpoint download needed.
- **PMU128** prefix matching (`--prefix-match-unit 128` +
  `--enable-prompt-tokens-details`), backed by a 3-patch vLLM series applied at
  build time by `apply_runtime_patches.py` (fail-closed, exact before/after
  SHA-256 gates):
  1. `0001-vllm-53388-native-mtp-block-drop.patch` — upstream PR #53388
     (`disable_eagle_block_drop`): MTP no longer drops the trailing
     prefix-cache block. Includes the SlidingWindowManager fine-grained-partial
     assert removal.
  2. `0002-vllm-53906-coordinator-partial-hits.patch` — upstream #53906
     hunk: hybrid coordinator partial-hash-hit check only considers groups
     that participate in prefix caching.
  3. `0003-vllm-scheduler-lcm-mamba-block-align.patch` — one-liner: scheduler
     `_mamba_block_aligned_split` uses the scheduler LCM block size, not
     `cache_config.block_size` (which PMU128 shrinks to the 128 hash
     granularity).
  plus the **current full SM121 kpool indexer overlay**
  (`sparse_attn_indexer_kpool_sm121.py`, sha `8a3ecfb0…`) — byte-identical to
  tonyd2wild's latest and to this recipe's
  `../patches/sparse_attn_indexer_kpool.py` (bar our provenance header).
- **Profile**: GMU 0.85, KV pin 13.5 GB/rank (fp8 e4m3) → **1,920,956-token
  logical pool** @ 1M ctx; max_num_seqs **6**; MNBT 8192; block 2304 (resolved
  scheduler block 4608, Mamba 2304); Marlin MoE; vision image 4 / video 0;
  port 8888 / head `10.0.7.1` / rendezvous 29531 in the upstream launcher —
  the parent compose maps these to the cluster's :8000 / 192.168.0.170 /
  29521.

## Why the parent recipe tracks this

Florianbrede's measured receipts (forum 382632): tool-eval-bench 2.6.x
hardmode seed 42 **91/100** (74 pass / 13 partial / 1 fail; TC-43
safety-gate fail) at t=1.0/top-p=0.95 — vs miken's 90/100 with DFlash2 k=7;
up to **108 tok/s generation @ C6** code-heavy; cold prefill 1,100–1,500
tok/s; 200K-context exact replay 199,936 cached / 64 computed; **PMU128 cuts
average prompt reprocessing to 82 tokens/turn** (vs ~2,304 at 4,608-token
granularity); >1B input / ~5M output tokens multi-day soak with zero errors.
Upstream claims MTP3 "performs significantly better than DFlash2 for
reasoning-heavy workloads" — the parent recipe keeps DFlash2 k=7 as the
default lane until this is A/B'd here, and this directory is the machine to
run that A/B against.

## Build + run

The image is built FROM the digest-pinned base (see `Dockerfile`) — the
patches are baked at build, no boot-time bind mounts:

```bash
cd glm-v53-flash-intel-w4a16/mtp3-pmu128
docker build -t glm53-intel-mtp3-pmu128:20260907 .
# load on both nodes (or build there); then bring up the parent recipe with
# IMAGE=glm53-intel-mtp3-pmu128:20260907 and the LANE=mtp3 (see .env).
```

Verification inside the built image (portable pin check, from upstream):

```bash
docker run --rm -v "$PWD:/recipe:ro" --entrypoint python3 \
  glm53-intel-mtp3-pmu128:20260907 \
  /recipe/apply_runtime_patches.py \
  --root /usr/local/lib/python3.12/dist-packages --verify-only
```

The surgery (`../prepare-model.sh`) is unchanged — the checkpoint and the
GPTQ metadata transformation are identical (florianbrede's
`adapt_autoround_to_gptq.py` asserts the same source config sha
`d4deaf40…`, 679 exclusion rules, and produces the same output config sha
`958beaf7…` as our `prepare-model.sh`; kept in the same repo conventions).

## Upstream validation status

`validate.sh --no-network` was NOT run for this vendoring; the SHA256SUMS
verify above is the integrity check for the copy. The full upstream validation
gate (base-fixture patch replay) runs implicitly at build time: the Dockerfile
runs the installer twice (apply + `--verify-only`) and fails the build on any
hash mismatch.
