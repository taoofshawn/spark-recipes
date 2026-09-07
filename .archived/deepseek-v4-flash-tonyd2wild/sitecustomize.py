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
tokenizer loads.

Why this module warns instead of raising: it runs in EVERY python process,
including the compose boot script's own "is fastokens installed?" probe, which
runs before the package is installed on first boot. A raise there would print a
scary ``Error in sitecustomize`` traceback for a condition that the boot script
is about to fix. The authoritative guard therefore lives in the compose boot
script (``docker-compose.yml``): it installs ``fastokens`` when the env var is
on and hard-fails the boot if the install fails. Here, a missing/too-old
package just logs a warning and the process continues — the shim is simply not
applied in that process.
"""

import os
import sys
from importlib.metadata import PackageNotFoundError, version

_MIN_FASTOKENS_VERSION = "0.2.0"


def _enabled() -> bool:
    # Mirror vllm/envs.py semantics: bool(int(os.getenv(...))).
    try:
        return bool(int(os.environ.get("VLLM_USE_FASTOKENS", "0")))
    except ValueError:
        return False


def _warn(message: str) -> None:
    # Greppable in container logs next to the boot script's [fastokens] lines.
    print(f"[fastokens] WARNING: {message}", file=sys.stderr, flush=True)


if _enabled():
    try:
        import fastokens
    except ImportError:
        _warn(
            "VLLM_USE_FASTOKENS=1 but the 'fastokens' package is not installed; "
            "the shim will NOT be applied. The compose boot script installs it "
            "before serving, so this only shows in helper processes that ran "
            "before the install."
        )
    else:
        try:
            installed = version("fastokens")
        except PackageNotFoundError:
            installed = None
        from packaging.version import Version

        if installed is None or Version(installed) < Version(_MIN_FASTOKENS_VERSION):
            _warn(
                "VLLM_USE_FASTOKENS=1 but fastokens "
                + (installed or "unknown")
                + " is older than "
                + _MIN_FASTOKENS_VERSION
                + "; the shim will NOT be applied."
            )
        else:
            fastokens.patch_transformers()
