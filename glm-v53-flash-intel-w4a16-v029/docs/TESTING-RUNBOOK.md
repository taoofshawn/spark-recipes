# TESTING-RUNBOOK — glm53-intel-w4a16-v029:20260920 bring-up + validation

**For the executing agent:** you are running on a remote backend. The agent
that built this image was running FROM the currently-serving vLLM instance on
the cluster (the `spark-llm` proxy on :4000 fronts the legacy GLM recipe).
Once you tear that instance down in Step 3, that agent is gone — do not expect
it to respond. This runbook is the handoff. Follow it top to bottom; every
gate has expected output. If a gate fails irrecoverably, execute the rollback
(§6) and write up what happened.

Branch: `glm53-w4a16-v029-stock` (all commits pushed). Recipe dir:
`glm-v53-flash-intel-w4a16-v029/`. Read `docs/BUILD-RECEIPT.md`,
`docs/FLAGS-AUDIT.md`, and `docs/PATCH-REBASE-NOTES.md` in that dir first.

## 0. What was built

`glm53-intel-w4a16-v029:20260920` (image ID `sha256:1e987123…7d286b048`) —
official `vllm/vllm-openai:v0.29.0-aarch64` base + baked SM121 patch stack
(#53388 backport, LCM align, PR #53969 NoPE/topk, PDL gate). Serves
`Intel/GLM-5.3-Flash-W4A16-AutoRound@5eee1846…` **raw** — no GPTQ surgery, no
runtime overlays, mtp3 lane (native MTP k=3 + PMU128) only.

Cluster facts: node0 = `spark-0f0b.shawndo.intra` (leader, rank 0, API
:8000, RoCE 192.168.0.170), node1 = `spark-6d14.shawndo.intra` (worker, rank
1, headless, 192.168.0.171). SSH from the workstation via the hostnames only
(never the 192.168.0.x IPs). Repo on the workstation:
`C:\Users\sdrew\code\github.com\taoofshawn\spark-recipes` (branch checked out).
Do NOT edit files on the nodes; all recipe changes flow through git.

## 1. Preconditions (verify, don't assume)

- [ ] Image exists on node0: `ssh spark-0f0b.shawndo.intra 'docker images glm53-intel-w4a16-v029 --format "{{.Tag}} {{.ID}}"'` → `20260920 sha256:1e987123…`
- [ ] Image on node1 — it is NOT there yet. Transfer (runs from node0, ~20 GiB over LAN, several minutes):
  ```bash
  ssh spark-0f0b.shawndo.intra 'docker save glm53-intel-w4a16-v029:20260920 | gzip' | ssh spark-6d14.shawndo.intra 'gunzip | docker load'
  ```
  Verify on node1 with the same `docker images` command.
- [ ] Weights on BOTH nodes (should already be there — the legacy lane uses the same snapshot):
  `ssh <node> 'jq -r ".quantization_config.quant_method" /home/sdrew/.cache/huggingface/hub/models--Intel--GLM-5.3-Flash-W4A16-AutoRound/snapshots/5eee1846f0321058ed73745f9aa16f2aaf0fc0a0/config.json'` → expect `auto-round`.
- [ ] Recipe files on the nodes come from the branch: `git pull origin glm53-w4a16-v029-stock` on node0 (the nodes have their own clones under the user's home; locate the existing spark-recipes clone with `ls ~/*/spark-recipes` on node0 — ask the user if ambiguous).
- [ ] If jq is missing on a node: `python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['quantization_config']['quant_method'])" <config.json>`.

## 2. Snapshot the legacy lane before touching it (rollback insurance)

```bash
ssh spark-0f0b.shawndo.intra 'docker ps --format "{{.Names}} {{.Image}} {{.Status}}"'
ssh spark-6d14.shawndo.intra 'docker ps --format "{{.Names}} {{.Image}} {{.Status}}"'
```
Record output. The legacy lane is `glm53-intel-w4a16` on image
`glm53-intel-mtp3-pmu128:20260907` (or similar). Its compose dir on the nodes
is the `glm-v53-flash-intel-w4a16` checkout — do not modify it.

## 3. Tear down the legacy serving stack (BOTH nodes)

```bash
ssh spark-0f0b.shawndo.intra 'docker ps -q --filter name=glm53 | xargs -r docker rm -f'
ssh spark-6d14.shawndo.intra 'docker ps -q --filter name=glm53 | xargs -r docker rm -f'
```
**From here on the build agent's backend is down.** Everything below runs from
your (remote) session.

## 4. Pre-launch ritual per node (GB10 swaps instead of OOMing)

```bash
ssh <node> 'sync; echo 3 | sudo tee /proc/sys/vm/drop_caches'
# optional, measured +5-7% decode on the legacy lane:
ssh <node> 'bash <recipe-dir-on-node>/tools/clocks.sh'   # only if present
```

## 5. Bring up the new recipe — WORKER FIRST

From the recipe dir on each node (`glm-v53-flash-intel-w4a16-v029/` inside
the node's repo clone):

```bash
# node1 (~35 s before node0):
ssh spark-6d14.shawndo.intra 'cd <recipe-dir> && docker compose --env-file .env --env-file .env.node1 up -d'
# node0, ~35 s later:
ssh spark-0f0b.shawndo.intra 'cd <recipe-dir> && docker compose --env-file .env --env-file .env.node0 up -d'
```

Wrong order hangs rendezvous (`DistStoreError: 1/2 clients`) — always
`docker compose down` on BOTH nodes between relaunches. Cold boot ~8–12 min.

## 6. Gates

### Gate A — boot markers (leader log)

```bash
ssh spark-0f0b.shawndo.intra 'docker logs glm53-intel-w4a16-v029 2>&1 | grep -F "GID auto-detect"'        # → NCCL_IB_GID_INDEX=N (not an error dump)
ssh spark-0f0b.shawndo.intra 'docker logs glm53-intel-w4a16-v029 2>&1 | grep -F "method='"'"'mtp'"'"'"'   # → SpeculativeConfig(method='mtp', ... num_speculative_tokens': 3
ssh spark-0f0b.shawndo.intra 'docker logs glm53-intel-w4a16-v029 2>&1 | grep -iE "inc|auto-round"'        # → the INC/auto-round load path (NO-SURGERY PROOF — record the exact line)
ssh spark-0f0b.shawndo.intra 'docker logs glm53-intel-w4a16-v029 2>&1 | grep -iE "marlin"'                # → Marlin MoE backend selected
ssh spark-0f0b.shawndo.intra 'docker logs glm53-intel-w4a16-v029 2>&1 | grep -F "GPU KV cache size"'      # → ~1.9M tokens @1M ctx
ssh spark-0f0b.shawndo.intra 'docker logs glm53-intel-w4a16-v029 2>&1 | grep -iE "B16|bf16|unquantized"'  # → excluded modules (attn/router/shared experts/visual/norms) load BF16
```
Also grep the load log for which attention backend was chosen
(`FLASHINFER_MLA_SPARSE_SM120` expected) and save the full leader boot log to
the recipe's research notes. **KV pool check:** if the pool is materially
below ~1.9M tokens (MRV2 accounting change suspects), boot once with
`KV_CACHE_MEMORY=` (empty) in `.env` — profiler-sized — and record both
numbers; keep whichever works for the bench.

### Gate B — API sanity (leader)

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://192.168.0.170:8000/health    # 200
curl -s http://192.168.0.170:8000/v1/models | jq -r '.data[0]|(.id,.root)'   # glm-5.3-flash / Intel/GLM-5.3-Flash-W4A16-AutoRound
curl -s http://192.168.0.170:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"Say hi in one word"}],"max_tokens":32}' | jq -r '.choices[0].message.content'
```
(Run curl from node0 itself via ssh if 192.168.0.170:8000 is not routable from
your session: `ssh spark-0f0b.shawndo.intra "curl -s …"`.)

### Gate C — PMU128 + spec decode + long-context indexer

- PMU: send the same >128-token prompt twice with
  `"stream_options":{"include_usage":true}`; response 2 must report
  `usage.prompt_tokens_details.cached_tokens` ≈ prompt_tokens rounded down to
  a 128 multiple. Prompts <128 tokens → `cached_tokens=0` (expected).
- Spec decode: a handful of agent-style prompts (few hundred tokens, code-ish);
  no `EngineDeadError`, no `DistStoreError`; check log
  `acceptance`/draft stats if present.
- Indexer/long-context: one ~30K-token prompt (e.g. a long pasted doc), then
  repeat it — decode must survive past ~24K ctx (the legacy kpool crash
  territory; PR #53969 + unified indexer under test here).

### Gate D — record everything

Append a dated entry to `glm-v53-flash-intel-w4a16-v029/research.md`: boot log
excerpts (INC line, backend selection, KV pool), gate results, timings,
failures. Commit + push the branch. Do NOT merge.

## 7. Failure playbook

| symptom | first response |
|---|---|
| warmup death in `FLASHINFER_MLA_SPARSE_SM120` (NaN / cudaFuncSetAttribute) | the PR #53969 fix was insufficient on real hw — capture the traceback; check flashinfer version in-image (`pip show flashinfer-python`); try `EAGER=1` boot to isolate CUDA graphs; do NOT improvise kernel patches — report back |
| quantization load error mentioning `auto-round`/`INC` | capture full traceback; as a diagnostic only, a local gptq-surgery copy can prove/disprove the config path (the legacy `prepare-model.sh` is lane-agnostic) — but the goal is raw load; report the exact error |
| marlin MoE rejected (`ValueError … WNA16`) | try `MOE_BACKEND=triton` boot (known ~2× slower) to isolate; report |
| `ValueError: DSpark currently requires uniform effective per-request target context lengths`-class spec-decode error | v0.29 regression class — capture, report |
| NCCL/RoCE hang at init | verify GID detect line first; compare NCCL env vs the legacy lane's; the legacy lane is the reference |
| KV pool tiny / OOM at first long prompt | unset KV_CACHE_MEMORY (profiler-sized) and re-boot; record |
| engine dead / `CUDA_ERROR_NOT_PERMITTED` under image+text concurrency | known watch item (forum 381350); record as watch item, don't debug |

**Rollback (§6 of the plan):** stop new containers on both nodes, bring the
legacy recipe back up (its dir, worker first), cluster restored. Nothing about
the legacy recipe was modified.

## 8. A/B bench — HARD PRE-MERGE GATE

After all gates pass: bench the new image vs the legacy lane per repo
discipline (`bench-recipe-update` skill): same prompts/harness, restart
between lanes, warm-cache discipline (never bench right after boot), read
`usage.completion_tokens` with `stream:false`. Regression >10% c4 aggregate
vs the legacy lane's recorded numbers (`glm-v53-flash-intel-w4a16/research.md`)
→ do NOT merge; investigate. Record both lanes' numbers in research.md.

## 9. When done

1. All gates + bench recorded in `research.md`, committed and pushed.
2. Leave the BETTER lane serving (user decides after seeing numbers; default
   expectation: legacy stays serving until the user merges the PR).
3. Report: gate results, bench table, and any watch items discovered. The
   user will merge the PR and push the image to ghcr
   (`ghcr.io/taoofshawn/vllm-glm53-intel-w4a16:v0.29.0-pmu128-mtp3`).
