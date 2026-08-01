# Kai — DS4 update (2026-07-08): concurrency crash fixed

## TL;DR
A new bug was found and fixed: **DS4 died the instant 2+ requests hit it at once.** The fix is in the repo. To pull it onto any DS4 deployment, from the **head node's checkout dir**:

```bash
bash update-and-restart.sh
```

That one command pulls the latest fixes, preserves your node-specific `.env.dspark`, and relaunches both nodes. (First time only: the node still needs its `.env.dspark` set up — that's already done on all our nodes.)

---

## What the bug was
- **Symptom:** with speculative decoding on, any time two or more requests were in a mixed prefill+decode batch, the DSpark proposer raised `ValueError: DSpark currently requires uniform flattened per-request inputs`, which **killed the worker and the whole engine.** After that, every request returned **HTTP 500** until a manual restart. Single-request traffic was fine, which is why it hid — it only bites under real concurrent load.
- **How to recognize it in the wild:** instant `500`s on every call + `uniform flattened` in the worker container logs (`docker logs <worker-container> 2>&1 | grep "uniform flattened"`).
- **Present on:** current `main` *even with* the earlier PR #4 slot-clamp guards active — this is a separate, deeper issue in the draft path.

## The fix (commit `90ab8e2`)
At the proposer's `propose()` entry we detect the non-uniform batch and **skip speculation for just that step** (one step of plain, correct decode) instead of crashing. Returns a full-shape dummy draft the verifier rejects naturally. Costs nothing on the healthy path — verified c1 unchanged at ~38 tok/s; c2/c4 which used to instant-500 now run clean (59.5 / 85.2 tok/s aggregate). Delivered as a runtime bind-mount (`recipe/vllm/v1/spec_decode/dspark_proposer.py`), no image rebuild.

## Two DS4 deployments — both now have the fix
| deployment | nodes | checkout | status |
|---|---|---|---|
| **DS4-1M** | Bluey + Reddie | `~/ds4-1m-repo` | fixed + running (serving now) |
| **DS4 (patch3)** | Asusi + Spark4 | `~/ds4-repo-patch3` | fix **applied to files**; takes effect on next boot (those nodes currently run GLM) |

Both checkouts point `start-deepseek-v4-flash-dspark.sh` at the same repo, so `update-and-restart.sh` works on either.

## Starting DS4 from cold (reminder)
- **Always** launch via `bash start-deepseek-v4-flash-dspark.sh` (or `update-and-restart.sh`) from the head node — never raw `docker compose up` (it won't read `.env.dspark`, pulls the wrong image, and skips the worker-file sync).
- After **any** failed boot: full recreate (`stop-` script on both nodes, then start fresh) — never retry-start into a half-dead state, that's the silent-hang trap.
- The `vllm_tri` container on Asusi is a **decoy** — it resolves the model from the HF hub cache (a stub) and always dies with `LocalEntryNotFoundError`. Not prod DS4. Safe to `docker rm`.

## Update (2026-07-08, later): fix delivery moved into the image
The bind-mount delivery above bit back the same day: a same-tag image rebuild
moved to a newer vLLM whose runner passes `req_ids=` to the proposer, and the
older mounted copy then crashed **every** request with `propose() got an
unexpected keyword argument 'req_ids'`. The overlay in `recipe/overlay/` (baked
into the image at build) already carried the concurrency fix, so the bind-mount
is now gone from compose, and the start script verifies the image against
`recipe/overlay/` and rebuilds automatically when stale. Practical effect for
you: `bash update-and-restart.sh` works unchanged, but the first run after this
update may trigger an image rebuild on rigs whose image predates the current
overlay — let it finish; that is the fix being baked in.

## What's still open (not blocking)
- **Issue #6 — soft-failure:** completions that return real tokens but parse to empty/thinking-only content, episodically under sustained agent load. Engine survives (HTTP 200 throughout), so it's a quality bug, not a crash. Under active investigation on the Bluey/Reddie rig; if you see empty agent responses that a retry fixes, that's this — capture the raw response body and drop it in issue #6.

## Endpoints
- DS4-1M: `http://100.92.77.51:8888/v1` (Bluey head), model `deepseek-v4-flash-dspark`, 1M context
- DS4 patch3 (when booted): `http://100.90.25.78:8888/v1` (Asusi head), 350K context
