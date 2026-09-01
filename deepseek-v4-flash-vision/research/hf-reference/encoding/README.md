# DeepSeek-V4 text and vision encoding

`encoding_dsv4.py` is the standalone prompt-format reference. It supports
multi-turn conversations, tool calls, thinking modes, and interleaved image
content blocks without importing the inference implementation.

## OpenAI-style messages

```python
from encoding_dsv4 import encode_messages

messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "第一张图"},
        {
            "type": "image_url",
            "image_url": {"url": "examples/images/carrots.jpeg"},
        },
        {"type": "text", "text": "有什么内容？"},
    ],
}]

# non-thinking
prompt, media = encode_messages(
    messages,
    thinking_mode="chat",
    return_multi_modal_data=True,
)
# prompt:
# '<｜begin▁of▁sentence｜><｜User｜>第一张图\n\n<｜deepseek_image｜>\n\n有什么内容？<｜Assistant｜></think>'


# # thinking with `max` reasoning_effort
# prompt, media = encode_messages(
#     messages,
#     thinking_mode="thinking",
#     reasoning_effort="max",
#     return_multi_modal_data=True,
# )
# prompt:
# <｜begin▁of▁sentence｜>Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\nYou MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\nDo not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n<｜User｜>第一张图\n\n<｜deepseek_image｜>\n\n有什么内容？<｜Assistant｜><think>

```

Images are represented in the prompt by `<｜deepseek_image｜>`. `media["images"]`
contains the corresponding image records in exactly the same order. Pixel
loading and expansion into model image tokens are handled by
`inference/image_processor.py`.

## Compact TXT notation

`parse_tagged_text()` converts a compact prompt such as

```text
第一张图<image>examples/images/carrots.jpeg</image>有什么内容？
```

into the same standard content blocks. It is an input convenience layer, not a
second encoding implementation.

## Tests

From the repository root:

```bash
python -m pytest -q encoding/test_encoding_dsv4.py
```

The tests include a check that the TXT and JSON examples encode to the same
prompt and preserve the same two-image ordering.
