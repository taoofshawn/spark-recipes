# SPDX-License-Identifier: Apache-2.0
"""Runtime hook that makes VLLM_USE_FASTOKENS actually work on this image.

The base image is a vLLM 0.21.x fork: its vendored ``envs.py`` overlay defines
the ``VLLM_USE_FASTOKENS`` env var but nothing in the 0.21.x code path consumes
it. vLLM 0.21 only ships the separate ``--tokenizer-mode fastokens``, which
would REPLACE the ``deepseek_v4`` tokenizer mode and drop the reasoning-effort /
tool-arg overlays this recipe depends on.

So this module does the same job vLLM main does lazily in ``get_tokenizer``,
but at interpreter start: ``fastokens.patch_transformers()`` is process-global
and idempotent, and swaps the inner Rust BPE backend of every HF fast tokenizer
loaded AFTER this point (``hf``, ``deepseek_v32``, ``deepseek_v4``, ``qwen_vl``,
...), rebinding ``tokenizers.decoders.DecodeStream`` for the streaming
detokenizer. This recipe's ``detokenizer.py`` overlay already looks the class
up on the module, so the rebind is honored regardless of import order — the
``deepseek_v4`` wrapper/encoder overlays keep working unchanged.

Python imports ``sitecustomize`` automatically at interpreter start (unless
``-S``), so every ``vllm`` / worker process applies the shim before any
tokenizer loads. Fail-fast: if the env var is on but ``fastokens`` is missing
or too old, raise a clear error instead of silently serving without it.
"""

import os
from importlib.metadata import PackageNotFoundError, version

_MIN_FASTOKENS_VERSION = "0.2.0"


def _enabled() -> bool:
    # Mirror vllm/envs.py semantics: bool(int(os.getenv(...))).
    try:
        return bool(int(os.environ.get("VLLM_USE_FASTOKENS", "0")))
    except ValueError:
        return False


if _enabled():
    try:
        import fastokens
    except ImportError as e:
        raise ImportError(
            "VLLM_USE_FASTOKENS=1 requires the 'fastokens' package "
            "(https://github.com/crusoecloud/fastokens, >= "
            + _MIN_FASTOKENS_VERSION
            + "). Install it with /opt/env/bin/python -m pip install "
            "'fastokens>=0.2.0' or unset VLLM_USE_FASTOKENS."
        ) from e

    try:
        installed = version("fastokens")
    except PackageNotFoundError:
        installed = None

    from packaging.version import Version

    if installed is None or Version(installed) < Version(_MIN_FASTOKENS_VERSION):
        raise ImportError(
            "VLLM_USE_FASTOKENS=1 requires fastokens >= "
            + _MIN_FASTOKENS_VERSION
            + " (found "
            + (installed or "unknown")
            + ")."
        )

    fastokens.patch_transformers()
