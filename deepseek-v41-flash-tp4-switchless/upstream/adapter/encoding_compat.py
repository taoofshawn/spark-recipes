"""Publisher-aligned V4.1 prompt encoding: reasoning-effort table, thinking kwargs,
and a completion-token cap.

SGLang's OpenAI chat path only reads ``chat_template_kwargs.thinking``. vLLM and the
EXL3 recipe use ``enable_thinking``. Without an alias, ``enable_thinking: true`` is
ignored (SGLang's default thinking flag is off) and a client that only sends the
vLLM name cannot turn thinking on.

The V4.1 encoder's named effort tiers also disagree with DeepSeek's
``encoding/encoding.py`` (SGLang: low/high = 25/50; publisher: 50/75). The
sitecustomize hook remaps them; ``xhigh`` stays as 75.

Chat completions that omit ``max_tokens`` fill the remaining context (1M on TP4).
That is how a racing-animation trial ran to 714k tokens. The 2x EXL3 recipe does
not install a server cap either — vLLM does the same fill — but every smoke and
README example there sends ``max_tokens``. This adapter fills *and* clamps at
``DSV41_MAX_NEW_TOKENS`` (default 32768; 0 restores the uncapped engine behaviour).
"""
import hashlib
import inspect
import logging
import os
import textwrap

logger = logging.getLogger(__name__)

# Publisher encoding/encoding.py REASONING_EFFORT_MAPPINGS, plus SGLang's xhigh alias.
PUBLISHER_REASONING_EFFORT = {'low': 50, 'high': 75, 'xhigh': 75, 'max': 100}
DEFAULT_MAX_NEW_TOKENS = 32768
# Staged SGLang AnthropicThinkingParam._validate_thinking_shape, dedented inspect source.
ANTHROPIC_THINKING_SHAPE_SHA256 = 'a94a9e59de88daa15d7ff86ea1b3c616d3b1b550ca90a5203c21b0aed363740e'
# Thinking-mode completion floor. A thinking request that asks for a small
# max_tokens spends the whole budget inside the reasoning segment and returns
# no answer at all (observed tool-eval-bench MMLU: 50/500 questions truncated at
# max_tokens=256, re-run with headroom 42/50 answered correctly; IFEval 26/541).
# Raising the *floor* keeps the request's own value when it is already large
# enough and never shortens output the way a thinking cap does. 0 disables.
#
# The floor applies only to requests that *ask* for thinking. A request that
# rides the engine-wide SGLANG_DEFAULT_THINKING without saying so keeps its own
# budget: that default sets how a request renders, not how much output the
# client is entitled to. Folding the default in here rewrote measurement
# clients' fixed budgets (llama-benchy's `max_tokens=128` throughput probes and
# tool-eval-bench `--spec-bench` both omit every thinking kwarg) into 16384, so
# the probes stopped at EOS instead of at their budget and reported 366-8,578
# generated tokens for a `tg128` cell (2026-10-23 fix1 run). Explicit-off and
# no-kwarg requests are unaffected by design; explicit-on is what the accuracy
# plugins and the scenario traffic send.
DEFAULT_MAX_NEW_TOKENS_FLOOR = 16384


def normalize_chat_template_kwargs(kwargs):
    """Copy ``enable_thinking`` onto ``thinking`` (and the reverse) in a new dict.

    ``thinking`` wins when both are present and disagree. ``None`` / empty input
    is returned unchanged so SGLang's ``SGLANG_DEFAULT_THINKING`` still applies.
    """
    if not kwargs:
        return kwargs
    out = dict(kwargs)
    has_thinking = 'thinking' in out
    has_enable = 'enable_thinking' in out
    if has_thinking and has_enable:
        thinking = bool(out['thinking'])
        enable = bool(out['enable_thinking'])
        if thinking != enable:
            logger.warning(
                'chat_template_kwargs thinking=%r disagrees with enable_thinking=%r; '
                'using thinking', out['thinking'], out['enable_thinking'])
        out['thinking'] = thinking
        out['enable_thinking'] = thinking
    elif has_enable:
        out['thinking'] = bool(out['enable_thinking'])
        out['enable_thinking'] = out['thinking']
    elif has_thinking:
        out['thinking'] = bool(out['thinking'])
        out['enable_thinking'] = out['thinking']
    return out


def configured_max_new_tokens():
    """Return the completion-token default/cap, or 0 to leave the engine uncapped."""
    raw = os.environ.get('DSV41_MAX_NEW_TOKENS', str(DEFAULT_MAX_NEW_TOKENS)).strip()
    try:
        return int(raw)
    except ValueError:
        logger.warning('DSV41_MAX_NEW_TOKENS=%r is not an int; using %s',
                       raw, DEFAULT_MAX_NEW_TOKENS)
        return DEFAULT_MAX_NEW_TOKENS


def configured_max_new_tokens_floor():
    """Return the thinking-mode completion floor, or 0 when the floor is off."""
    raw = os.environ.get('DSV41_MAX_NEW_TOKENS_FLOOR',
                         str(DEFAULT_MAX_NEW_TOKENS_FLOOR)).strip()
    try:
        return int(raw)
    except ValueError:
        logger.warning('DSV41_MAX_NEW_TOKENS_FLOOR=%r is not an int; using %s',
                       raw, DEFAULT_MAX_NEW_TOKENS_FLOOR)
        return DEFAULT_MAX_NEW_TOKENS_FLOOR


def request_enables_thinking_explicitly(request):
    """True when the client itself asked for the reasoning segment.

    Only an explicit ``thinking`` / ``enable_thinking`` key counts. The
    engine-wide ``SGLANG_DEFAULT_THINKING`` is deliberately not consulted: it
    decides whether a request renders reasoning, not whether the client's own
    ``max_tokens`` may be overridden (see ``DEFAULT_MAX_NEW_TOKENS_FLOOR``).

    Runs ahead of ``_process_messages``, so ``enable_thinking`` may still be
    unaliased here; ``thinking`` wins when both keys are present, matching
    ``normalize_chat_template_kwargs``. ``None`` values are treated as absent.
    """
    kwargs = getattr(request, 'chat_template_kwargs', None)
    if isinstance(kwargs, dict):
        if kwargs.get('thinking') is not None:
            return bool(kwargs['thinking'])
        if kwargs.get('enable_thinking') is not None:
            return bool(kwargs['enable_thinking'])
    return False


def apply_request_max_tokens(request):
    """Fill omitted max_tokens / max_completion_tokens and clamp to the cap.

    Mutates ``request`` in place and returns it. ``limit <= 0`` is a no-op.
    ``max_completion_tokens`` wins when both fields are set, matching SGLang.
    A request that explicitly enables thinking whose budget is below the floor
    is raised to it: the reasoning segment otherwise consumes the entire
    budget and the response ends with no answer (``finish_reason=length``,
    empty content). A request that only inherits ``SGLANG_DEFAULT_THINKING``
    keeps its own budget — see ``request_enables_thinking_explicitly``.
    """
    if request is None:
        return request
    limit = configured_max_new_tokens()
    floor = configured_max_new_tokens_floor()
    if floor > 0 and request_enables_thinking_explicitly(request):
        completion = getattr(request, 'max_completion_tokens', None)
        tokens = getattr(request, 'max_tokens', None)
        current = completion if completion is not None else tokens
        if current is None:
            if limit > 0:
                request.max_tokens = limit
        elif current < floor:
            target = min(floor, limit) if limit > 0 else floor
            if completion is not None:
                request.max_completion_tokens = target
            else:
                request.max_tokens = target
    if limit <= 0:
        return request
    completion = getattr(request, 'max_completion_tokens', None)
    tokens = getattr(request, 'max_tokens', None)
    current = completion if completion is not None else tokens
    if current is None:
        request.max_tokens = limit
        return request
    if current > limit:
        if completion is not None:
            request.max_completion_tokens = limit
        else:
            request.max_tokens = limit
    return request


def install_encoder(module):
    """Point encoding_dsv41.REASONING_EFFORT_MAPPINGS at the publisher table."""
    module.REASONING_EFFORT_MAPPINGS.clear()
    module.REASONING_EFFORT_MAPPINGS.update(PUBLISHER_REASONING_EFFORT)


def install_serving_chat(module):
    """Rewrite thinking kwargs and apply the completion-token cap on every chat request."""
    cls = module.OpenAIServingChat
    original = cls._process_messages

    def _process_messages(self, request, is_multimodal):
        request.chat_template_kwargs = normalize_chat_template_kwargs(
            request.chat_template_kwargs)
        kwargs = request.chat_template_kwargs
        if not isinstance(kwargs, dict) or 'thinking' not in kwargs:
            # Surface SGLANG_DEFAULT_THINKING as an explicit kwarg: read directly
            # from the env it only arms prompt encoding — the reasoning parser
            # keys off chat_template_kwargs, so without this a default-on request
            # thinks but the  think block leaks into ``content`` unsplit
            # (verified live 2026-10-23). Explicit client values still win.
            default = os.environ.get('SGLANG_DEFAULT_THINKING', '0').strip().lower() in (
                '1', 'true', 'yes', 'on')
            merged = dict(kwargs) if isinstance(kwargs, dict) else {}
            merged['thinking'] = default
            merged['enable_thinking'] = default
            request.chat_template_kwargs = merged
        return original(self, request, is_multimodal)

    cls._process_messages = _process_messages

    original_convert = getattr(cls, '_convert_to_internal_request', None)
    if original_convert is not None:
        def _convert(self, request, *args, **kwargs):
            apply_request_max_tokens(request)
            return original_convert(self, request, *args, **kwargs)

        cls._convert_to_internal_request = _convert

    limit = configured_max_new_tokens()
    logger.warning(
        'DSV41 chat_template_kwargs: enable_thinking is accepted as an alias for thinking'
        + (f'; omitted max_tokens defaults and caps at {limit}' if limit > 0
           else '; DSV41_MAX_NEW_TOKENS=0 leaves completions uncapped')
        + (f'; requests that explicitly enable thinking below '
           f'{configured_max_new_tokens_floor()} tokens are raised to it '
           '(DSV41_MAX_NEW_TOKENS_FLOOR; 0 = off; implicit default-thinking '
           'requests keep their own budget)'
           if configured_max_new_tokens_floor() > 0 else ''))


def install_anthropic_protocol(module):
    """Accept Anthropic ``thinking: {"type": "enabled"}`` without ``budget_tokens``.

    SGLang's Messages protocol requires ``budget_tokens`` on ``enabled`` (the SDK's
    discriminated shape), but its serving layer never enforces a budget: anything other
    than ``disabled`` turns reasoning on, and ``output_config.effort`` sets the effort.
    Clients that send ``enabled`` with ``output_config.effort`` and no budget (DeepSeek
    Harness's DeepSeek provider does) got a 400 on every request. Without a budget,
    ``enabled`` now passes like ``adaptive``; a budget, when given, is still checked, and
    ``disabled`` and ``adaptive`` keep their rules.
    """
    cls = getattr(module, 'AnthropicThinkingParam', None)
    validators = getattr(getattr(cls, '__pydantic_decorators__', None), 'model_validators', {})
    entry = validators.get('_validate_thinking_shape')
    if entry is None:
        logger.warning('DSV41 Anthropic thinking: AnthropicThinkingParam shape check not found; unchanged')
        return False
    if getattr(entry.func, '_dsv41_optional_budget', False):
        return False
    original = entry.func
    try:
        source = textwrap.dedent(inspect.getsource(original))
        if hashlib.sha256(source.encode()).hexdigest() != ANTHROPIC_THINKING_SHAPE_SHA256:
            raise ValueError('AnthropicThinkingParam._validate_thinking_shape source drift')
    except (TypeError, OSError, ValueError) as exc:
        logger.warning('DSV41 Anthropic thinking: refusing to patch; %s; unchanged', exc)
        return False

    def _validate_thinking_shape(self):
        if self.type == 'enabled' and self.budget_tokens is None:
            return self
        return original(self)

    _validate_thinking_shape._dsv41_optional_budget = True
    entry.func = _validate_thinking_shape
    cls.model_rebuild(force=True)
    for name in ('AnthropicMessagesRequest', 'AnthropicCountTokensRequest'):
        model = getattr(module, name, None)
        if model is not None:
            model.model_rebuild(force=True)
    logger.warning('DSV41 Anthropic thinking: "enabled" is accepted without budget_tokens '
                   '(effort comes from output_config.effort; budgets are not enforced)')
    return True
