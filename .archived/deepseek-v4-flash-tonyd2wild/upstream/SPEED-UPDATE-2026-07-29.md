# Speed + deployment update — 2026-07-29: 83.4 tok/s peak, and four deployment bugs fixed

Re-deployed this recipe from scratch onto a **different node pair** than the original
Bluey+Reddie, which surfaced four real defects that only appear when the node layout changes.
Also re-measured decode properly (peak *and* mean, by content type) and beat the previous
70.3 tok/s figure.

## Measured (single-stream, temperature 0, warm, TP=2, k=5 + probabilistic draft)

| workload | tok/s |
|---|---:|
| **count to 300** (maximally predictable) | **83.4** |
| 20x20 multiplication table | 76.1 |
| 12x12 multiplication table | 78.2 |
| count to 150 | 78.0 |
| JSON rows x60 | 77.2 |
| 24 boilerplate functions | 74.9 – 75.4 |
| **prime-sum math** (repo's old 70.5 test) | **71.7** |
| BST implementation (repo's old 70.3 test) | 64.1 |

**Peak 83.4 · mean 74.1** across the high-acceptance suite. Both of the previous update's
prompts reproduce: prime-math **71.7** (was 70.5); BST **64.1** (was 70.3 — see caveat below).

> **Read these numbers correctly.** Decode speed here is *acceptance-driven*, and acceptance is
> content-driven. The same server measures **83 tok/s on counting and 64 on a BST implementation**
> — a ~30% spread with zero config change. Quoting a single number without the workload is
> meaningless. Report peak **and** mean **and** the content type.

## Config that produced it (no tuning wins were found — the baked config is already optimal)

`--kv-cache-dtype nvfp4_ds_mla --block-size 256 --max-model-len 1048576 --max-num-seqs 6
--max-num-batched-tokens 8192 --gpu-memory-utilization 0.78`
`--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}'`

Experiments that did **not** help (all measured, all rejected):

| lever tried | result |
|---|---|
| `max-model-len` 1M → 200K | 65.2 peak (no gain) |
| `max-num-seqs` 6 → 2 | 66.3 peak (no gain) |
| `--max-cudagraph-capture-size 36` | 65.4 peak (no gain) |
| `num_speculative_tokens` 7 | **rejected at boot** — must be divisible by `n_predict=5` |
| `num_speculative_tokens` 10 | boots, then **every generation crashes** (CUDA error) |

**`k` is effectively locked at 5** on this drafter: values must be multiples of `n_predict=5`,
and 10 crashes at request time. Note this also means the previously documented `MTP_NUM_TOKENS=3`
is **not valid** against this image — it fails config validation.

## Four deployment bugs — only visible on a different node layout

The recipe worked on the original Bluey(head)+Reddie(worker) pair partly by luck: several
node-specific values are baked into the image and were correct only for that pairing.

1. **`VLLM_HOST_IP=192.168.192.1` is baked into the image ENV.** That is "my own address" for the
   process. On any other node the worker tries to bind an IP it does not own and dies with
   `zmq.error.ZMQError: Cannot assign requested address (addr='tcp://192.168.192.1:...')`.
   **Fix: set `-e VLLM_HOST_IP=<this node's fabric IP>` per node.**
2. **The same image tag has a DIFFERENT baked `CMD` on different nodes.** One node's copy carries
   `--node-rank 0`, another's carries `--node-rank 1 --headless`. A launcher that reuses the baked
   command inherits the wrong identity, both nodes come up as rank 1, and the cluster **hangs
   silently at distributed init with no error**. **Fix: normalize `--node-rank`/`--headless`
   explicitly; never trust the baked values.**
3. **`NCCL_IB_HCA` is hardcoded** (`rocep1s0f0`) but RoCE device names are not uniform across
   nodes. **Fix: auto-detect the Up RoCE device whose netdev carries the fabric IP.**
4. **The image's baked `CMD` is missing BOTH halves of the 2026-07-03 garble fix** — no
   `draft_sample_method` (so it runs a **greedy draft**, the documented garble trigger) and it
   carries an `--override-generation-config`. Anyone launching from the baked command rather than
   compose gets garble. **Fix: inject `"draft_sample_method":"probabilistic"` and strip
   `--override-generation-config`.** Verified clean afterwards by a gate that checks for CJK drift,
   repetition loops, template/XML leakage, and empty-content-with-tokens-billed: **5/5 clean**.

## Also

- Engine deaths that exit `0` (see issue #8) defeat Docker restart policies. Run containers with
  `--restart unless-stopped`.
- `MTP_NUM_TOKENS` default in `.env`/compose was corrected from the stale `3` to `5` (+24%).
