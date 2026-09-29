# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-4.7 parser for reasoning and tool calls.

GLM-4.7 uses XML-like tool calls::

    <tool_call>func_name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>

The function name can be followed directly by the first ``<arg_key>`` tag,
and tool calls may have no arguments.

SITE PATCH (2026-09-29): this file is a site override of the image's
``vllm/parser/glm47_moe.py`` (pristine source extracted from
``ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache``, content ID ``5e32aaa1...``;
vLLM ``0.11.2.dev279+eldritch.final.fcc6141``). It adds structural-tag
enforcement for ``tool_choice="required"`` / named function choice (TC-45 fix).
Do not overwrite this file wholesale on upstream/image updates -- see
``research.md`` section 2.6 (local mods 6-7).
"""

from __future__ import annotations

import functools
import json
from typing import TYPE_CHECKING

import regex as re

from openai.types.responses import ToolChoiceFunction

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
import vllm.envs as envs
from vllm.sampling_params import StructuredOutputsParams
from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

THINK_START = "<think>"
THINK_END = "</think>"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
ARG_KEY_START = "<arg_key>"
ARG_KEY_END = "</arg_key>"
ARG_VALUE_START = "<arg_value>"
ARG_VALUE_END = "</arg_value>"

_ARG_RE = re.compile(
    r"<arg_key>(?P<key>.*?)</arg_key>\s*"
    r"<arg_value>(?P<value>.*?)</arg_value>",
    re.DOTALL,
)
_PARTIAL_ARG_RE = re.compile(
    r"<arg_key>(?P<key>.*?)</arg_key>\s*"
    r"<arg_value>(?P<value>.*)$",
    re.DOTALL,
)


def _glm47_arg_converter(raw_args: str, partial: bool) -> str:
    params: dict[str, object] = {}

    for match in _ARG_RE.finditer(raw_args):
        params[match.group("key").strip()] = match.group("value")

    if partial:
        remaining = _ARG_RE.sub("", raw_args)
        match = _PARTIAL_ARG_RE.search(remaining)
        if match:
            key = match.group("key").strip()
            if key:
                params[key] = match.group("value")

    return json.dumps(params, ensure_ascii=False)


@functools.cache
def glm47_moe_config(thinking: bool = True) -> ParserEngineConfig:
    arg_tag_transitions = {
        (ParserState.TOOL_ARGS, terminal): Transition(
            ParserState.TOOL_ARGS,
            (EventType.ARG_VALUE_CHUNK,),
        )
        for terminal in (
            "ARG_KEY_START",
            "ARG_KEY_END",
            "ARG_VALUE_START",
            "ARG_VALUE_END",
        )
    }

    reasoning_terminals = (
        {
            "THINK_START": THINK_START,
            "THINK_END": THINK_END,
        }
        if thinking
        else {}
    )
    reasoning_token_id_terminals = (
        {
            "THINK_START": THINK_START,
            "THINK_END": THINK_END,
        }
        if thinking
        else {}
    )
    reasoning_transitions = (
        {
            (ParserState.CONTENT, "THINK_START"): Transition(
                ParserState.REASONING,
                (EventType.REASONING_START,),
            ),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                (EventType.REASONING_END,),
            ),
            (ParserState.CONTENT, "THINK_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
        }
        if thinking
        else {}
    )

    return ParserEngineConfig(
        name="glm47_moe",
        initial_state=ParserState.REASONING if thinking else ParserState.CONTENT,
        terminals={
            **reasoning_terminals,
            "TOOL_START": TOOL_CALL_START,
            "TOOL_END": TOOL_CALL_END,
            "ARG_KEY_START": ARG_KEY_START,
            "ARG_KEY_END": ARG_KEY_END,
            "ARG_VALUE_START": ARG_VALUE_START,
            "ARG_VALUE_END": ARG_VALUE_END,
        },
        # SITE PATCH (2026-09-29, TC-45 follow-up): TOOL_START/TOOL_END REMOVED from
        # token_id_terminals. StreamingParserEngine._process_lex_tokens demotes any
        # TEXT-matched terminal whose name is in _token_id_terminal_names to plain
        # content once the stream has delivered token ids ("strict" mode): a text
        # <tool_call> is treated as prose because it did not arrive as the special-token id.
        # Under the forced structural tag (xgrammar) the tool tags arrive as ordinary
        # tokens, so keeping them here made every forced tool call unparseable (raw
        # <tool_call>…</tool_call> envelope leaked into `content`). The text lexer matches them fine
        # (adjust_request forces skip_special_tokens=False). THINK ids stay: they drive
        # is_reasoning_end/extract_content_ids and the scanner's THINK_END pre-lex; the
        # (REASONING, TOOL_START) transition still yields an implicit reasoning end.
        token_id_terminals={
            **reasoning_token_id_terminals,
        },
        transitions={
            **reasoning_transitions,
            (ParserState.REASONING, "THINK_START"): Transition(
                ParserState.REASONING,
                (),
            ),
            (ParserState.REASONING, "TOOL_START"): Transition(
                ParserState.TOOL_NAME,
                (EventType.REASONING_END, EventType.TOOL_CALL_START),
            ),
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_NAME, "ARG_KEY_START"): Transition(
                ParserState.TOOL_ARGS,
                (EventType.ARG_VALUE_CHUNK,),
            ),
            (ParserState.TOOL_NAME, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
            **arg_tag_transitions,
        },
        arg_converter=_glm47_arg_converter,
        stream_arg_deltas=True,
        tool_args_json=False,
        validate_tool_names=True,
    )


class Glm47MoeParser(ParserEngine):
    """GLM-4.7 parser backed by the declarative parser engine."""

    # SITE PATCH (2026-09-29): structural tag model, matching the registered
    # Glm47MoeModelToolParser.structural_tag_model in the image's
    # vllm/tool_parsers/glm47_moe_tool_parser.py.
    STRUCTURAL_TAG_MODEL = "glm_4_7"

    # --- SITE PATCH (2026-09-29): structural-tag enforcement for
    # tool_choice="required" / named function choice. Mirror of
    # DelegatingParser._apply_structural_tag plus the envs gate from
    # ToolParser.get_structural_tag; the engine-based path otherwise
    # never applies the model's structural tag. ---
    def adjust_request(self, request):
        request = super().adjust_request(request)
        return self._apply_structural_tag(request)

    def _apply_structural_tag(self, request):
        if not request.tools:
            return request
        need_tool_calling = (
            request.tool_choice == "auto"
            or request.tool_choice == "required"
            or isinstance(
                request.tool_choice,
                (ChatCompletionNamedToolChoiceParam, ToolChoiceFunction),
            )
        )
        if not need_tool_calling:
            return request
        if not envs.VLLM_ENFORCE_STRICT_TOOL_CALLING:
            return request
        from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag

        structure_tag = get_model_structural_tag(
            model=self.STRUCTURAL_TAG_MODEL,
            tools=request.tools,
            tool_choice=request.tool_choice,
            reasoning=False,
        )
        if structure_tag is None:
            return request
        request.structured_outputs = StructuredOutputsParams(
            structural_tag=json.dumps(structure_tag.model_dump()),
        )
        if isinstance(request, ResponsesRequest):
            request.text = None
        else:
            request.response_format = None
        return request
    # --- end SITE PATCH ---

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        thinking = chat_kwargs.get("thinking", None)
        enable_thinking = chat_kwargs.get("enable_thinking", None)
        self.thinking_enabled = (
            True
            if thinking is None and enable_thinking is None
            else bool(thinking) or bool(enable_thinking)
        )
        kwargs.setdefault(
            "parser_engine_config",
            glm47_moe_config(thinking=self.thinking_enabled),
        )
        super().__init__(tokenizer, tools, **kwargs)

    def _emit_name_delta(self, idx: int, deltas, name: str | None) -> None:
        if name is not None:
            name = name.strip()
        super()._emit_name_delta(idx, deltas, name)

    def _handle_tool_end(self, event, deltas) -> None:
        idx = event.tool_index
        if 0 <= idx < len(self._tool_slots):
            self._tool_slots[idx].name = self._tool_slots[idx].name.strip()
        super()._handle_tool_end(event, deltas)

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        if not self.thinking_enabled:
            return True
        return super().is_reasoning_end(input_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if not self.thinking_enabled:
            return input_ids
        return super().extract_content_ids(input_ids)

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        if not self.thinking_enabled:
            return None, model_output
        return super().extract_reasoning(model_output, request)
