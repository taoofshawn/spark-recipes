#!/usr/bin/env python3
"""Host-side tests for the DSV41_MAX_NEW_TOKENS default/cap."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'adapter'))

from encoding_compat import (  # noqa: E402
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MAX_NEW_TOKENS_FLOOR,
    apply_request_max_tokens,
    configured_max_new_tokens,
    configured_max_new_tokens_floor,
    install_serving_chat,
)


def _with_env(value, fn, key='DSV41_MAX_NEW_TOKENS'):
    old = os.environ.get(key)
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value
    try:
        return fn()
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


def test_default_limit():
    assert _with_env(None, configured_max_new_tokens) == DEFAULT_MAX_NEW_TOKENS
    assert _with_env('65536', configured_max_new_tokens) == 65536
    assert _with_env('0', configured_max_new_tokens) == 0


def test_omitted_fills_default():
    req = SimpleNamespace(max_tokens=None, max_completion_tokens=None)
    _with_env(None, lambda: apply_request_max_tokens(req))
    assert req.max_tokens == DEFAULT_MAX_NEW_TOKENS


def test_under_cap_is_kept():
    req = SimpleNamespace(max_tokens=256, max_completion_tokens=None)
    _with_env(None, lambda: apply_request_max_tokens(req))
    assert req.max_tokens == 256


def test_over_cap_is_clamped():
    req = SimpleNamespace(max_tokens=1_000_000, max_completion_tokens=None)
    _with_env(None, lambda: apply_request_max_tokens(req))
    assert req.max_tokens == DEFAULT_MAX_NEW_TOKENS


def test_max_completion_tokens_wins_and_clamps():
    req = SimpleNamespace(max_tokens=16, max_completion_tokens=999_999)
    _with_env('1024', lambda: apply_request_max_tokens(req))
    assert req.max_completion_tokens == 1024
    assert req.max_tokens == 16  # unused sibling left alone


def test_zero_disables():
    req = SimpleNamespace(max_tokens=None, max_completion_tokens=None)
    _with_env('0', lambda: apply_request_max_tokens(req))
    assert req.max_tokens is None


def test_install_convert_applies_cap():
    class FakeServing:
        class OpenAIServingChat:
            def _process_messages(self, request, is_multimodal):
                return request.chat_template_kwargs

            def _convert_to_internal_request(self, request, raw_request=None):
                return request.max_tokens, request

    def run():
        install_serving_chat(FakeServing)
        serving = FakeServing.OpenAIServingChat()
        req = SimpleNamespace(max_tokens=None, max_completion_tokens=None,
                              chat_template_kwargs={'thinking': False})
        return serving._convert_to_internal_request(req)

    tokens, out = _with_env('4096', run)
    assert tokens == 4096
    assert out.max_tokens == 4096


def _req(max_tokens=None, kwargs=None, completion=None):
    return SimpleNamespace(max_tokens=max_tokens, max_completion_tokens=completion,
                           chat_template_kwargs=kwargs)


def test_floor_raises_starved_thinking_request():
    req = _req(256, {'thinking': True})
    _with_env(str(DEFAULT_MAX_NEW_TOKENS_FLOOR),
              lambda: apply_request_max_tokens(req),
              key='DSV41_MAX_NEW_TOKENS_FLOOR')
    assert req.max_tokens == DEFAULT_MAX_NEW_TOKENS_FLOOR


def test_floor_reads_unaliased_enable_thinking():
    req = _req(256, {'enable_thinking': True})
    _with_env('8192', lambda: apply_request_max_tokens(req),
              key='DSV41_MAX_NEW_TOKENS_FLOOR')
    assert req.max_tokens == 8192


def test_floor_leaves_ample_budget_alone():
    req = _req(20000, {'thinking': True})
    _with_env('16384', lambda: apply_request_max_tokens(req),
              key='DSV41_MAX_NEW_TOKENS_FLOOR')
    assert req.max_tokens == 20000


def test_floor_ignores_non_thinking_requests():
    req = _req(256, {'thinking': False})
    _with_env('16384', lambda: apply_request_max_tokens(req),
              key='DSV41_MAX_NEW_TOKENS_FLOOR')
    assert req.max_tokens == 256


def test_floor_ignores_engine_default_thinking():
    """The engine default renders reasoning; it must not rewrite the budget.

    llama-benchy's throughput probes and tool-eval-bench ``--spec-bench`` send
    ``max_tokens`` and no thinking kwarg at all. Folding the default into the
    floor decision raised those fixed budgets to 16384, so the probes stopped
    at EOS instead of at their budget and the cell stopped being the
    measurement its `tg128` label advertises (2026-10-23 fix1 run).
    """
    req = _req(128, None)

    def run():
        return _with_env('true', lambda: apply_request_max_tokens(req),
                         key='SGLANG_DEFAULT_THINKING')

    _with_env('16384', run, key='DSV41_MAX_NEW_TOKENS_FLOOR')
    assert req.max_tokens == 128


def test_floor_ignores_engine_default_for_max_completion_tokens():
    req = _req(16, None, completion=128)

    def run():
        return _with_env('true', lambda: apply_request_max_tokens(req),
                         key='SGLANG_DEFAULT_THINKING')

    _with_env('16384', run, key='DSV41_MAX_NEW_TOKENS_FLOOR')
    assert req.max_completion_tokens == 128
    assert req.max_tokens == 16


def test_floor_raises_explicit_thinking_when_engine_default_off():
    req = _req(256, {'thinking': True})

    def run():
        return _with_env('0', lambda: apply_request_max_tokens(req),
                         key='SGLANG_DEFAULT_THINKING')

    _with_env('16384', run, key='DSV41_MAX_NEW_TOKENS_FLOOR')
    assert req.max_tokens == 16384


def test_floor_treats_explicit_none_as_absent():
    """An explicit ``thinking: null`` is not an opt-in."""
    req = _req(128, {'thinking': None})

    def run():
        return _with_env('true', lambda: apply_request_max_tokens(req),
                         key='SGLANG_DEFAULT_THINKING')

    _with_env('16384', run, key='DSV41_MAX_NEW_TOKENS_FLOOR')
    assert req.max_tokens == 128


def test_install_keeps_probe_budget_when_engine_default_thinks():
    """End-to-end ordering guard for the 2026-10-23 contamination.

    SGLang's ``_convert_to_internal_request`` calls ``_process_messages``
    itself, so the floor wrapper (outermost) sees the client's raw kwargs. The
    injection that makes the default explicit happens afterwards — it must not
    feed back into the floor decision, or a probe with no kwargs gets the
    reasoning parser's rendering *and* a 16384-token budget again.
    """

    class FakeServing:
        class OpenAIServingChat:
            def _process_messages(self, request, is_multimodal):
                return 'processed'

            def _convert_to_internal_request(self, request, raw_request=None):
                self._process_messages(request, False)
                return request

    def run():
        install_serving_chat(FakeServing)
        serving = FakeServing.OpenAIServingChat()
        req = SimpleNamespace(max_tokens=128, max_completion_tokens=None,
                              chat_template_kwargs=None)
        return serving._convert_to_internal_request(req)

    out = _with_env('true', run, key='SGLANG_DEFAULT_THINKING')
    assert out.max_tokens == 128
    # the parser still gets an explicit kwarg, so the think block is split
    assert out.chat_template_kwargs == {'thinking': True, 'enable_thinking': True}


def test_floor_is_bounded_by_cap():
    req = _req(256, {'thinking': True})

    def run():
        return _with_env('32768', lambda: apply_request_max_tokens(req),
                         key='DSV41_MAX_NEW_TOKENS_FLOOR')

    _with_env('8192', run)
    assert req.max_tokens == 8192


def test_floor_applies_to_max_completion_tokens():
    req = _req(16, {'thinking': True}, completion=256)
    _with_env('4096', lambda: apply_request_max_tokens(req),
              key='DSV41_MAX_NEW_TOKENS_FLOOR')
    assert req.max_completion_tokens == 4096
    assert req.max_tokens == 16


def test_floor_zero_keeps_request_budget():
    req = _req(256, {'thinking': True})
    _with_env('0', lambda: apply_request_max_tokens(req),
              key='DSV41_MAX_NEW_TOKENS_FLOOR')
    assert req.max_tokens == 256


def test_floor_off_by_default_env_reading():
    assert _with_env(None, configured_max_new_tokens_floor,
                     key='DSV41_MAX_NEW_TOKENS_FLOOR') == DEFAULT_MAX_NEW_TOKENS_FLOOR
    assert _with_env('garbage', configured_max_new_tokens_floor,
                     key='DSV41_MAX_NEW_TOKENS_FLOOR') == DEFAULT_MAX_NEW_TOKENS_FLOOR
    assert _with_env('0', configured_max_new_tokens_floor,
                     key='DSV41_MAX_NEW_TOKENS_FLOOR') == 0


if __name__ == '__main__':
    tests = [v for k, v in list(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f'[OK]   {fn.__name__}')
        except Exception as exc:
            failed += 1
            print(f'[FAIL] {fn.__name__}: {exc}')
    print(f'{len(tests) - failed}/{len(tests)} passed')
    sys.exit(1 if failed else 0)
