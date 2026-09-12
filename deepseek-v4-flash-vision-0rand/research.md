# research.md — deepseek-v4-flash-vision-0rand: maintenance notes

Working notes for future update/maintenance sessions on this recipe. The
README is the deploy doc (active-running content only); this file is the
memory — dated changelog entries, update-pass findings, and TODO/watch
items live here (AGENTS.md convention).

## Changelog

### 2026-09-12 — adopt upstream PR #1 mm-prefix span fix (opt-in mod)

Upstream merged oselivanov's PR #1 (`d0c8584`, 2026-09-12T16:13Z): the V2
model runner derived DSV4 vision mm-prefix bidirectional ranges from
`PlaceholderRange.extract_embeds_range()` (per-row-pair IMAGE-embed runs of
the N-layout) instead of the full sentinel block `[pad + IMAGE_START …
IMAGE_END]` — with the N-layout interleaving rows, y leaked into the causal
128-token window → y-only grounding bias ("center collapse"). Measured on
our exact image/rev: ball-sweep y 416/501/501 → 250/350/750, interior
errors ≤0.006 (PR #1 comments + INVESTIGATION-vision-grounding-bias.md
§12.7).

Adopted as an OPT-IN knob mirroring upstream (`FIX_MM_PREFIX_SPAN`, default
0 — upstream's default too):

- vendored `mods/fix-dsv4-mm-prefix-span/` verbatim from upstream @
  `d0c8584` (`dsv4-mm-prefix-span.patch` + `run.sh`, SHA256SUMS recorded).
  Python-only patch of `vllm/v1/worker/gpu/attn_utils.py`
  (`compute_mm_prefix_ranges`) + `vllm/v1/worker/gpu/model_states/default.py`
  (`DefaultModelState.prepare_attn`); idempotent `patch -p1 -N` with dry-run
  guards; applied on BOTH nodes at boot (compose mounts `./mods:/mods:ro`
  and runs the mod via `bash` — upstream stores `run.sh` non-executable,
  100644 — before `exec vllm serve` when `FIX_MM_PREFIX_SPAN=1`).
- patch failure is fatal by design: an explicitly requested fix must not be
  silently missing. If a future image changes the two files' context, boot
  stops loudly — re-vendor from upstream or flip the flag off.
- `.env` ships `FIX_MM_PREFIX_SPAN=0`; README documents the flip + both-node
  recreate requirement.

Caveat carried from the PR review (unanswered before merge): the V2 port
reads `mm_prefix_clamp_sliding_window` only from `hf_text_config` (V1 also
reads the module — "so it can't silently no-op later"). Harmless at our
pinned rev (attributes exist; the V1 path proves the mechanism); re-verify
on any image rev bump. Watch-list entry for PR #1 closed; vLLM #56141
(parser follow-up) remains the top image-refresh watch item.

### 2026-09-11 — upstream review: no runtime changes adopted

Sources swept (window 2026-09-10 → 09-11): NVIDIA forum thread 381911
(all posts >#214) + board sweep (categories 721/723) + full-text searches;
GitHub (0rand primary + siblings, PilcoTHINK lane, Dickson, vLLM
`deepseek_v4` parser); Docker Hub tags; HF model revisions.

**Pins verified current (no action):**

- Image `dicksondickson/vllm_spark_dsv4:0.29-b12x` @ `sha256:899d174e…` is
  still the only tag (count=1 at page_size 25 and 100), digest unmoved
  since its 2026-09-10T06:26Z push; 0rand confirmed this exact pin on the
  forum (#229).
- 0rand's own hub image (`0rand/vllm_spark_dsv4-0.29-b12x:latest` @
  `85eb91ee`) unchanged since 2026-09-08 — still the superseded build.
- HF `DeepSeek-V4-Flash-Vision-Exp`: HEAD `6821d6ad` (lastModified
  2026-09-01) vs pinned `86f746b3` — git-tree OID diff shows `README.md` +
  `.eval_results/` only; `encoding/encoding_dsv4.py`, tokenizer/config
  files, and all 48 shards byte-identical (same LFS sha256s). Keep
  `86f746b3`; a pin==HEAD bump would buy nothing.
- PilcoTHINK Dockerfile lane: zero commits after `38db013b` (pin = HEAD);
  no 0.29/0.30 lane directory exists.

**Upstream delta since our pin (`74e9f11` → `b94f1fbd`):** one functional
commit, `0cf86aa` (2026-09-10): image → Dickson's build (already adopted
09-10), `MAX_NUM_SEQS` 8→4, and `--async-scheduling` (commit message:
"was mistakingly omitted" — i.e. intended config, absent at our pin by
accident).

**Not adopted (re-evaluated):** `--async-scheduling` + seqs=4 pairing.
Still no measured claim at our seqs=8 / batch-4096 profile; the seqs flip
contradicts our validated agent-serving profile; the DSpark +
BREAKABLE_CUDAGRAPH interaction is unverified on this image. Revisit if
numbers land at a comparable profile.

**Watch list (adoption triggers):**

- **vLLM #56141 (`9e25706`, merged 2026-09-10T07:04Z)** — "[Bugfix]
  Tolerate misspelled DSML tool_calls wrapper": production-observed DSML
  opener misspellings (`<｜DSML｜tool>` vs `<｜DSML｜toolcalls>`) emitted by
  DSv4-Flash itself; direct follow-up to the `d98c8c03` parser fix baked
  into our image. Merged 38 min AFTER Dickson's image push → NOT in any
  published vision image; PilcoTHINK's newer engine lane (vLLM
  `0.28.1rc1.dev637+g9e2570656`, forum 382957) contains it but has no
  DSv4-vision image yet. → Adopt on the next image refresh that contains
  `9e25706`; do NOT hand-backport the multi-file parser change into the
  digest-pinned image (untestable without a boot; outside the update
  skill's no-launch scope).
- **0rand PR #1** (oselivanov) — DSV4 vision mm-prefix span fix in the V2
  model runner (bidirectional range derived from the full sentinel block,
  not just the IMAGE-embed runs); measured grounding repair (ball-sweep y
  416/501/501 → 250/350/750, interior error ≤0.006), gated by
  `FIX_MM_PREFIX_SPAN=1`. **MERGED 2026-09-12 (`d0c8584`) — adopted here as
  the opt-in `FIX_MM_PREFIX_SPAN` knob; see the 2026-09-12 entry.**
- **vLLM #52865** (open) — DSv4/V3.2 tool-argument streaming: long string
  args buffered until `</parameter>` instead of streaming incrementally;
  touches our pinned parser file.
- **vLLM #50753** (open) — surface unterminated reasoning as content
  (opt-in `force_nonempty_content`); author validated on 2× GB10 TP=2.
- **0rand's B12X bug hunt** (forum 382939 #2/#4) — B12X bugs "affect both
  GLM and DS4FVE with high probability"; quality testing so far shows no
  impact ("it just folds to a different path"). Results pending — could
  touch the B12X env family.
- **K=5 DSpark** (forum 382675 + `hxfhd/deepseek-v4-dspark-k5-workaround`)
  — validator-patched k=5 measured on aiden `production-3.73-vision`
  (draft work −15–20%, avg 47.5 tok/s): NOT our image, author calls it
  unsupported, and #171 is precedent for off-standard-k thinking damage.
  HOLD. Side-receipt: k=6 is the smallest stock-legal k for this
  checkpoint (n_predict=3, dspark_block_size=5) — validates our k=6 pin.
- **DSv4.1-Flash successor lane** — vLLM #56208 frontend merged;
  tonyd2wild 4× TP4 measured 77.2 tok/s peak / 1M ctx (forum 382897);
  2-Spark feasibility unknown (the "552B" is int8 double-packed per
  381911 #217–#225; 2–3-bit quant would be needed). Track as a future
  lane, not this recipe.

**Recipe hygiene in this pass:**

- compose command-block fallback for `MAX_NUM_BATCHED_TOKENS` aligned
  2048 → 4096 (matches `.env`, the `environment:` block, and the README
  shipped value; no behavior change — `.env` always sets it).
- The 2026-09-10 changelog block moved here from the README per the
  AGENTS.md convention (README = active-running doc only).
- Ops knowledge for the running cluster distilled into the README's new
  "Ops notes" section (prefill headroom, client max-output-tokens
  footgun, GB10 clock latch, GB10 cgroups gap).

### 2026-09-10 — switch to Dickson image (upstream tool-call parser fix)

*(moved from the README per the AGENTS.md convention)*

**What changed:** the image pin moved from `0rand/vllm_spark_dsv4-0.29-b12x`
@ `85eb91ee` (PilcoTHINK `9df77f2e` lane) to `dicksondickson/vllm_spark_dsv4`
@ `899d174e` (Dickson's build of the PilcoTHINK `38db013b` lane). Same vLLM
base `6fbb00b1`, same model pin, same `o_proj` donor — the only functional
delta is that `38db013b` **bakes in the upstream DeepSeek V4 tool-call parser
fix** (`vllm/parser/deepseek_v4.py` from vLLM `d98c8c03`, "[Bugfix] Parse DSML
tool calls when the model omits the tool_calls wrapper" #55954). The earlier
0rand image predates that commit.

**Why:** the parser fix is the measured delta behind the higher tool-call
quality on this image. stu.miller's locked v1.8.0 comparison (forum #212):
**pilco/0rand 86 vs Dickson 93** tool-eval-bench (Dickson 93/100 post #176,
92/100 post #194). 0rand himself switched to the same image (forum #214) but
has not yet pushed his own hub tag.

**Not adopted:** `--async-scheduling` (Dickson's only other change, post #194,
paired with `MAX_NUM_SEQS=4`, post #200). No measured claim at our seqs=8 /
batch-4096 profile, and it wasn't shipped in 0rand's `.env.sample` — left off
to avoid an un-validated knob on a fragile cross-recipe dimension.
*(2026-09-11 note: `0cf86aa` has since shipped it upstream paired with
seqs=4; still holding — see the 2026-09-11 entry.)*

**Gotchas:** image is arm64 (GB10-correct), ~24.2 GiB pulled; cold-boot JIT is
unchanged. No flag/env changes — kv-cache-dtype fp8, FLASHINFER_MLA_SPARSE_DSV4,
AOT=0/BREAKABLE=1, GMU 0.87, k=6, port 8000, worker-first all preserved.
