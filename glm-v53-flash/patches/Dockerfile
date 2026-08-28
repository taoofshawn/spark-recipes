# GLM-5.3-Flash-NVFP4 patched vLLM image for DGX Spark (GB10/SM121).
#
# Built from the vLLM PR authors' day-0 image, pinned by digest. The stock
# image works on B200 but fails on GB10/SM121 in five separate ways; this
# Dockerfile consolidates tonyd2wild's patch chain (v1 -> v8, which is the
# validated STABLE stack) into a single build. v9 (InstantTensor loader) is
# deliberately NOT included — it is unstable in 2-node TP2 (a rank dies
# silently ~60-90 s after load in every tested boot).
#
# Patch layers (each guarded — refuses to apply if the base source drifted):
#  1. SM121 NoPE MLA backend  (v1)  — only stock SM12x sparse backend forces
#     DeepSeek's packed fp8_ds_mla cache (pe_dim=64); GLM-5.3 is NoPE MLA
#     (qk_rope_head_dim=0) -> assert death in warmup. Extend the SM90 NoPE
#     backend (plain bf16 cache, FlashInfer BatchMLAPagedAttentionWrapper) to
#     capability 12, select its FA2 path off-Hopper.
#  2. FlashInfer 0.6.18 nightly (v3) — 0.6.17's FA2 MLA scheduler emits NaN
#     for 64-256-row batches on SM121 (normal prompts land exactly there).
#  3. NCCL 2.30.7 re-pin        (v4) — the nightly silently downgrades
#     nvidia-nccl-cu13 to 2.29.7, which fails on the Spark IB fabric.
#  4. cutlass-dsl 4.6.2 re-pin  (v5) — the nightly also left a mixed
#     4.7.0/4.6.2 install that ICEs the CuTeDSL warmup.
#  5. PDL gated off SM12x       (v6) — Programmatic Dependent Launch races the
#     KDA state kernels on unvalidated SM121 (boots NaN or don't by timing).
#  6. Indexer top-k hardening   (v7) — kpool top-k dest torch.empty ->
#     torch.full(-1) + pool-id clamp (uninitialized ids -> NaN lottery).
#  7. fp8 KV cache on SM12x     (v8) — cap the FA2 fp8 CTA_TILE_KV instead of
#     forcing Hopper's 32 (over-requests smem on GB10's ~101KB max), and
#     relax the FlashInfer fp8 MLA gate to SM12x. First fp8 KV for a NoPE-MLA
#     model on consumer Blackwell.
#  8. persistent_topk SM121 gate (crash forensics) — the DSA indexer's
#     persistent_topk hard-crashes ANY decode past ~24K context on GB10 (its
#     FilteredTopK fallback needs 128KB smem/block; SM121 has ~101KB). The
#     overlay routes small-SM parts to top_k_per_row_decode instead.
#
# Upstream: https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-2x-DGX-Spark
#   (docker/Dockerfile.glm53-sm121* + patch_v7.py + patch_v8_fp8.py +
#    sparse_attn_indexer_kpool_sm121.py)
FROM vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce

# ---- 1) SM121 NoPE MLA backend (v1) ---------------------------------------
RUN python3 - <<'PY'
from pathlib import Path

base = Path("/usr/local/lib/python3.12/dist-packages/vllm")

p = base / "platforms/cuda.py"
s = p.read_text()
old = """        elif device_capability.major == 12:
            return [
                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
            ]"""
new = """        elif device_capability.major == 12:
            return [
                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM90,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
            ]"""
if s.count(old) != 1:
    raise SystemExit("unexpected cuda.py capability-12 MLA candidate list; refusing to patch")
p.write_text(s.replace(old, new))

p = base / "v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py"
s = p.read_text()

old = "    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 9
"
new = "    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major in (9, 12)
"
if s.count(old) != 1:
    raise SystemExit("unexpected sm90 capability gate; refusing to patch")
s = s.replace(old, new)

old = '            backend="fa3",
'
new = '            backend=("fa3" if torch.cuda.get_device_capability()[0] == 9 else "fa2"),
'
if s.count(old) != 1:
    raise SystemExit("unexpected sm90 wrapper backend literal; refusing to patch")
s = s.replace(old, new)

old = """        if not has_flashinfer_sm90_nope_mla():
            return (
                "FLASHINFER_MLA_SPARSE_SM90 requires FlashInfer with SM90 "
                "MLA support (ckv_scale_arr in "
                "BatchMLAPagedAttentionWrapper.run, FlashInfer >= 0.6.18)"
            )"""
new = """        if kv_cache_dtype in ("fp8", "fp8_e4m3") and not has_flashinfer_sm90_nope_mla():
            return (
                "FLASHINFER_MLA_SPARSE_SM90 fp8 KV requires FlashInfer with "
                "SM90 MLA support (ckv_scale_arr in "
                "BatchMLAPagedAttentionWrapper.run, FlashInfer >= 0.6.18)"
            )"""
if s.count(old) != 1:
    raise SystemExit("unexpected sm90 flashinfer version gate; refusing to patch")
s = s.replace(old, new)

p.write_text(s)
print("sm121 NoPE MLA patches applied")
PY

# ---- 2) FlashInfer 0.6.18 nightly (v3) --------------------------------------
# Fixes the 0.6.17 FA2 MLA NaN on 64-256-row SM121 batches; removes the stale
# 0.6.17 AOT jit-cache so everything JIT-compiles for 12.1a.
RUN pip install -q --pre \
      flashinfer-python==0.6.18.dev20260819 \
      flashinfer-cubin==0.6.18.dev20260819 \
      --extra-index-url https://flashinfer.ai/whl/nightly/ \
    && pip uninstall -q -y flashinfer-jit-cache \
    && python3 -c "import flashinfer; print('flashinfer', flashinfer.__version__)"

# ---- 3) NCCL 2.30.7 re-pin (v4) ----------------------------------------------
# The flashinfer nightly silently downgraded nvidia-nccl-cu13 2.30.7 -> 2.29.7,
# which fails ncclCommInitRank on the Spark IB fabric.
RUN pip install -q nvidia-nccl-cu13==2.30.7 && pip show nvidia-nccl-cu13 | grep -q "Version: 2.30.7" && echo NCCL-RESTORED

# ---- 4) cutlass-dsl 4.6.2 re-pin (v5) ----------------------------------------
# The nightly also bumped nvidia-cutlass-dsl to a mixed 4.7.0/4.6.2 state that
# ICEs the CuTeDSL warmup. The FA2 MLA path does not use cutlass-dsl.
RUN pip install -q nvidia-cutlass-dsl==4.6.2 && pip show nvidia-cutlass-dsl | grep -q "Version: 4.6.2" && echo DSL-RESTORED

# ---- 5) PDL gated off SM12x (v6) ---------------------------------------------
RUN python3 - <<'PY'
from pathlib import Path

p = Path("/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py")
s = p.read_text()
old = """    @classmethod
    def is_arch_support_pdl(cls) -> bool:
        try:
            device = torch.cuda.current_device()
            major, _ = torch.cuda.get_device_capability(device)
        except Exception:
            return False
        return major >= 9
"""
new = """    @classmethod
    def is_arch_support_pdl(cls) -> bool:
        try:
            device = torch.cuda.current_device()
            major, _ = torch.cuda.get_device_capability(device)
        except Exception:
            return False
        # PDL lowering is unvalidated on SM12x (GB10) and races on KDA
        # state kernels there; keep it to Hopper/Blackwell-datacenter.
        return major in (9, 10)
"""
if s.count(old) != 1:
    raise SystemExit("unexpected is_arch_support_pdl source; refusing to patch")
p.write_text(s.replace(old, new))
print("PDL gated off on SM12x")
PY

# ---- 6) Indexer top-k hardening (v7) -----------------------------------------
COPY patch_v7.py /tmp/patch_v7.py
RUN python3 /tmp/patch_v7.py && rm /tmp/patch_v7.py

# ---- 7) fp8 KV cache on SM12x (v8) --------------------------------------------
COPY patch_v8_fp8.py /tmp/patch_v8_fp8.py
RUN python3 /tmp/patch_v8_fp8.py && rm /tmp/patch_v8_fp8.py

# ---- 8) persistent_topk SM121 gate (crash forensics) --------------------------
# Full-file overlay of vLLM's sparse_attn_indexer_kpool.py with the SM121
# small-SM gate (routes persistent_topk to top_k_per_row_decode below 78 SMs).
COPY sparse_attn_indexer_kpool.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py

# ---- syntax-check everything we touched --------------------------------------
RUN python3 -m py_compile \
      /usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py \
      /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py \
      /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py
