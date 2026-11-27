# deepseek-v41-flash-christopherowen — operations README

DeepSeek-V4.1-Flash **1017 release**
([`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
@ `1bae0e10`, MIT) served on the **4-node switchless ConnectX-7 ring** by
[`christopherowen/spark-ds41f`](https://github.com/christopherowen/spark-ds41f) — vLLM
(TP4) + sparknet one-shot ring collectives + the TileLang kernel family. Stock weights, zero
weight modification, no separate drafter (DSpark/Engram live inside the checkpoint).

This is the **second** DeepSeek-V4.1-Flash lane in this repo. It differs from
`deepseek-v41-flash-tp4-switchless` (SGLang/knapcio) in engine, transport
(`oneshot-ring4` + in-image NCCL vs the host four-PF lib) and shape. **Both serve the same
model name**, so switching between them needs no proxy edit.

| | |
|---|---|
| upstream pin | **`spark-ds41f-1120`** → commit `d3d40d94` (`VENDORED-AT.md`) |
| vendored | in-repo under `upstream/` (closure-only tree; 1,468 files after the site layer) |
| image | `spark-ds41f:1120`, built locally by `upstream/bin/spark build image` |
| shape | `tp4-ring-4k` — 1M ctx / 16 seqs / 8,192 batched / KV 10.5 GiB, stock 4 KiB kernel |
| control | `./bin/start.sh` → `upstream/bin/spark --cluster-config config/cluster.json cluster start` |
| endpoint | `http://10.69.43.170:8000/v1`, model `deepseek-v41-flash` |
| container | **one name on every node: `spark-ds41f`** (not per-rank) |

## This lane is not the qualified upstream shape

Upstream's promoted four-node shape is `tp4-ring` (32 seqs / 16.5 GiB KV / 64 KiB kernel +
memory saver). This site runs the **externally reported** 4 KiB variant that
`docs/setup.md` records for our exact host (`6.17.0-1032-nvidia`, Secure Boot on, no memory
saver): 16 seqs, 8,192 batched, 10.5 GiB KV. Consequence: **no release benchmark qualifies
it**, so `cluster start` needs `--unqualified-shape` until our own run is promoted (that flag
is baked into `bin/start.sh` via `site/lane.env`). See `research.md` D5.

## Cluster map

| rank | node | mgmt IP | role |
|---|---|---|---|
| 0 | `spark-0f0b` | 10.69.43.170 | head — API `:8000`, `cluster start` runs here |
| 1 | `spark-6d14` | 10.69.43.171 | worker |
| 2 | `spark-6d90` | 10.69.43.172 | worker |
| 3 | `spark-6d24` | 10.69.43.173 | worker |

Ring fabric: the shared four-PF netplan (10.10.1.x–10.10.4.x + P2 views), MTU 9000. **This
lane advertises two of the four addressed PFs** (`rocep1s0f0`, `rocep1s0f1`) — one HCA per
neighbour, which is what sparknet's ring4 map allows. The host keeps all four (shared netplan;
`site/fabric/preflight.sh` still verifies the four-PF contract).

## Start / stop

Both run **on the head node**, from this recipe's directory in the nodes' checkout
(`~/code/spark-recipes/deepseek-v41-flash-christopherowen`).

```bash
./bin/start.sh --dry-run     # guard + site-drift assertion + the exact command
./bin/start.sh               # guard, then cluster start --apply --replace --unqualified-shape
./bin/stop.sh                # capture 4 logs -> cluster stop --remove --apply --parallel -> verify
```

`bin/start.sh` refuses while anything else serves (`bin/guard.sh`, fail-closed; it knows all
four lanes' container names). `bin/stop.sh` captures `docker logs spark-ds41f` from **every**
node into `logs/` *before* removing the containers — this lane keeps its only log there.

`cluster start` defragments memory, arms a startup memory guard (5 GiB / 0.25 s), launches
**rank 3 → 2 → 1 → 0**, waits for `/v1/models` (timeout 3600 s), then swaps in the steady guard
(3 GiB / 2 s). The guards are systemd units: `spark-ds41f-memguard.service` per node.

## Gates (before any measurement)

```bash
site/gate-gid-slot.sh          # every routed PF carries exactly ONE IPv4 RoCE v2 GID
site/fabric/preflight.sh       # host fabric contract (read-only; safe while serving)
```

Then, after `/v1/models` is 200: identity (`deepseek-v41-flash`, ctx 1,048,576), a chat
round-trip, a forced tool call with thinking on, an emoji/CJK corruption probe, 0 `NCCL WARN`
in all four logs, and one ~993K-token prefill. `research.md` holds the measured receipts.

## Things that bite

| symptom | cause / fix |
|---|---|
| `cluster start` refuses: "no release benchmark qualifies this shape" | expected — this lane is `--unqualified-shape` until our own run is promoted (`site/lane.env` `SHAPE_UNQUALIFIED=1`) |
| this lane starts beside a GLM lane | upstream's own interlock only knows `spark-ds41f`/`^dsv41-`. The refusal is **this recipe's** `bin/guard.sh`; never run `upstream/bin/spark` by hand |
| `cluster sync` says "the deployed commit does not carry it" | the `profile_committed()` site patch was lost in a re-vendor — see `VENDORED-AT.md` |
| `doctor` says "tracked secret-like path" | the node map was renamed to `*.local.json`; it must stay `config/nodes-ring4.json` |
| `build prepare` reports a missing patch | a re-vendor dropped part of the closure — regenerate with the closure script, never hand-copy |
| NCCL "invalid usage" right after start | a routed PF lost or gained an IPv4 RoCE v2 GID — run `site/gate-gid-slot.sh` |
| image build refuses | needs ≥64 GiB MemAvailable and no `spark-ds41f`/`^dsv41-` container: stop the lane first |
| boot log says the engine found a *different* container | `cluster start` refuses to replace an engine from another profile; that is correct — use that profile's own stop |

## Docs in this recipe

| file | contents |
|---|---|
| `research.md` | provenance, dated changelog, decisions, gate receipts, watchlist |
| `site/FROM-ZERO.md` | four wiped nodes → served lane, from this recipe alone |
| `upstream/VENDORED-AT.md` | the pin, the four site patches, and how to re-apply them |
| `upstream/docs/` | upstream's own docs (`setup.md`, `switchless-topology.md`, `memory-profiles.md`, …) |