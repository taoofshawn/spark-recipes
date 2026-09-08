# dflash2-pmu128 — vendored lane (external DFlash2 k7 + PMU128)

Verbatim vendor (SHA256SUMS-verified) of
[florianbrede-ayet/spark-recipes/tp2_glm53flash_autoround_dflash2_k7_pmu128](https://github.com/florianbrede-ayet/spark-recipes/tree/main/tp2_glm53flash_autoround_dflash2_k7_pmu128)
at upstream commit `fd508a5c` ("Publish sanitized Spark deployment recipes",
2026-09-07). Upstream thread: [forum 382632](https://forums.developer.nvidia.com/t/glm-5-3-flash-intel-autoquant-w4a16-tp2-mtp3-concurrent-agentic-use/382632).

**What this lane is:** the recipe's default DFlash2 k=7 drafter, but on an
image with florianbrede's PMU128 patch series baked in
(#53388 block-drop + #53906 coordinator partial hits + scheduler LCM/mamba
block alignment + `dflash2-pmu128-swa-fine-hits.patch`, derived from draft PR
#54397). PMU128 (`--prefix-match-unit 128`) removes the 2304-token block
granularity on prefix hits; florianbrede measured warm TTFT dropping from
seconds to ~0.2 s and ~82-token average recompute per turn on agentic traffic.

**Same base image as the default lane** — digest `4def0ef6…` (sm121-v11).
The SM121 indexer overlay in `patches/` is code-identical to the recipe's
`../patches/sparse_attn_indexer_kpool.py` (upstream ships it with a 12-line
provenance header; bodies match).

## Build

```bash
cd dflash2-pmu128
./validate.sh --no-network            # repo-only gate
docker build -t glm53-intel-dflash2-pmu128:20260908 .
./validate.sh --image glm53-intel-dflash2-pmu128:20260908   # full gate (network + docker)
```

Drafter note: florianbrede's validated pin is
`incoai/GLM-5.3-Flash-DFlash2@bf582e4e…` (2026-08-31 upload). Our default
lane pins `dc77ff1c…` (2026-08-28). For this lane, download the `bf582e4e`
revision and set `DFLASH_REVISION=bf582e4eacc1810f76656d1811693ff6c6737d2a`
in `.env` (the snapshot resolver accepts any revision present in the cache).

## Run (this repo's compose conventions)

`.env`:

```bash
LANE=dflash2pmu
IMAGE=glm53-intel-dflash2-pmu128:20260908
DFLASH_REVISION=bf582e4eacc1810f76656d1811693ff6c6737d2a
KV_CACHE_MEMORY=13500000000   # florianbrede's 13.5 GB pin -> 1.81M-token pool
MAX_SEQS=6
PMU=1
```

Then the standard flow: worker (`--env-file .env.node1`) first, leader ~35 s
later. The compose command block gates the boot-time hybrid-APC patch to the
plain `dflash2` lane only — the pmu-baked images carry the equivalent patches
inside the vLLM tree (the boot-time anchors would not match).

Served profile (matches upstream launcher): MNBT 8192, retention 0, PMU128,
block 2304 / scheduler LCM 4608, pool 1,814,557 tokens @ 13.5 GB pin, GMU
0.85, `disable_eagle_block_drop=true` on the dflash spec config.

## Upstream receipts (2026-09-04 measurements, florianbrede's node)

- tool-eval-bench hardmode: **90/100** (158/176 points; TC-33/TC-43 fails)
- llama-benchy PP2048/TG128: cold d0 C1 1,678.9/33.1, C4 1,653.5/54.7 tok/s;
  d32K warm C1 1,097.2/24.0, C4 1,198.1/36.0
- Prefix: 32,384/32,385 hit on a 32K warm replay; ~82 tokens recomputed on
  uniform residues; 8×200K prefixes retained (retention 8/8; peak admission
  only 2 concurrent — not a C8 proof)
- Known carve-out: prompt length exactly 4,608-aligned does not materialize
  the replay-cap Mamba checkpoint (0 hit / 36,864 compute at N=36,864);
  no scheduler fix upstream — deliberate
- DFlash acceptance 33.9–37.2% on their workloads (lower than miken's
  0.68–0.71 code-acceptance figure; different workload mix)

## Provenance

- `SHA256SUMS` (in this dir) verifies every vendored file; the two manifests
  under `patches/` bind the base/final in-image runtime states.
- Upstream `launch.sh`/`validate.sh` are kept non-executable by design
  (upstream: "accidental direct execution must fail"); invoke via `bash`.
- Vendoring did NOT run `validate.sh` against a built image — that happens at
  first bring-up (bring-up-spark-recipe territory).
