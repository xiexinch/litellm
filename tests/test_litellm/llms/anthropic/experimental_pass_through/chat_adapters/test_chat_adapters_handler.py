"""
Tests for the chat_adapters handler — end-to-end verification that
Anthropic /v1/messages -> OpenAI Chat Completions -> Anthropic response
correctly handles thinking output.
"""

import json
import os
import sys

import pytest
from unittest.mock import MagicMock, patch

from litellm.types.utils import (
    Choices,
    Message,
    ModelResponse,
    Usage,
)


# ---------------------------------------------------------------------------
# Helper: build a minimal ModelResponse
# ---------------------------------------------------------------------------


def _make_response(
    *,
    content=None,
    thinking_blocks=None,
    reasoning_content=None,
    tool_calls=None,
    finish_reason="stop",
    model="openai/gpt-5.2",
) -> ModelResponse:
    msg_kwargs = {"role": "assistant"}
    if content is not None:
        msg_kwargs["content"] = content
    if thinking_blocks is not None:
        msg_kwargs["thinking_blocks"] = thinking_blocks
    if reasoning_content is not None:
        msg_kwargs["reasoning_content"] = reasoning_content
    if tool_calls is not None:
        msg_kwargs["tool_calls"] = tool_calls
    return ModelResponse(
        id="test-id",
        model=model,
        choices=[
            Choices(
                finish_reason=finish_reason,
                message=Message(**msg_kwargs),
            )
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=5),
    )


# ---------------------------------------------------------------------------
# End-to-end tests through the new chat_adapters handler
# ---------------------------------------------------------------------------


class TestChatAdaptersThinkingOutput:
    """Verify thinking content from chat/completions is converted to Anthropic blocks."""

    def test_should_convert_thinking_blocks_to_anthropic_thinking(self):
        """thinking_blocks[type=thinking] -> Anthropic content[type=thinking] with signature."""
        from litellm.llms.anthropic.experimental_pass_through.chat_adapters.handler import (
            LiteLLMMessagesToChatCompletionHandler,
        )

        mock_resp = _make_response(
            content="The answer is 4.",
            thinking_blocks=[
                {
                    "type": "thinking",
                    "thinking": "Let me calculate 2+2.",
                    "signature": "sig_abc123",
                },
            ],
        )

        with patch("litellm.completion", return_value=mock_resp):
            result = LiteLLMMessagesToChatCompletionHandler.anthropic_messages_handler(
                max_tokens=1024,
                messages=[{"role": "user", "content": "What is 2+2?"}],
                model="openai/gpt-5.2",
            )

        content = result["content"]
        assert len(content) == 2
        assert content[0]["type"] == "thinking"
        assert content[0]["thinking"] == "Let me calculate 2+2."
        assert content[0]["signature"] == "sig_abc123"
        assert content[1]["type"] == "text"
        assert content[1]["text"] == "The answer is 4."

    def test_should_convert_redacted_thinking_blocks(self):
        """thinking_blocks[type=redacted_thinking] -> Anthropic content[type=redacted_thinking]."""
        from litellm.llms.anthropic.experimental_pass_through.chat_adapters.handler import (
            LiteLLMMessagesToChatCompletionHandler,
        )

        mock_resp = _make_response(
            content=None,
            thinking_blocks=[
                {
                    "type": "thinking",
                    "thinking": "Step 1 analysis.",
                    "signature": "sig_step1",
                },
                {"type": "redacted_thinking", "data": "REDACTED_DATA"},
            ],
        )

        with patch("litellm.completion", return_value=mock_resp):
            result = LiteLLMMessagesToChatCompletionHandler.anthropic_messages_handler(
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hello"}],
                model="openai/gpt-5.2",
            )

        content = result["content"]
        assert len(content) == 2
        assert content[0]["type"] == "thinking"
        assert content[0]["thinking"] == "Step 1 analysis."
        assert content[0]["signature"] == "sig_step1"
        assert content[1]["type"] == "redacted_thinking"
        assert content[1]["data"] == "REDACTED_DATA"

    def test_should_convert_reasoning_content_to_thinking_block(self):
        """reasoning_content (flat string) -> Anthropic thinking block with signature=None."""
        from litellm.llms.anthropic.experimental_pass_through.chat_adapters.handler import (
            LiteLLMMessagesToChatCompletionHandler,
        )

        mock_resp = _make_response(
            content="There are 3 r's in strawberry.",
            reasoning_content="Let me count: s-t-r-a-w-b-e-r-r-y. That's 3.",
        )

        with patch("litellm.completion", return_value=mock_resp):
            result = LiteLLMMessagesToChatCompletionHandler.anthropic_messages_handler(
                max_tokens=1024,
                messages=[{"role": "user", "content": "How many r's in strawberry?"}],
                model="openai/gpt-5.2",
            )

        content = result["content"]
        assert len(content) == 2
        assert content[0]["type"] == "thinking"
        assert "Let me count" in content[0]["thinking"]
        assert content[0]["signature"] is None
        assert content[1]["type"] == "text"
        assert "3 r's" in content[1]["text"]

    def test_should_order_thinking_text_tool_use(self):
        """When thinking + text + tool_calls, order should be: thinking -> text -> tool_use."""
        from litellm.llms.anthropic.experimental_pass_through.chat_adapters.handler import (
            LiteLLMMessagesToChatCompletionHandler,
        )
        from litellm.types.utils import ChatCompletionMessageToolCall, Function

        mock_resp = _make_response(
            content="Let me look that up.",
            thinking_blocks=[
                {
                    "type": "thinking",
                    "thinking": "I should use the search tool.",
                    "signature": "sig_xx",
                },
            ],
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="call_abc",
                    type="function",
                    function=Function(
                        name="web_search",
                        arguments='{"query": "weather in Paris"}',
                    ),
                )
            ],
            finish_reason="tool_calls",
        )

        with patch("litellm.completion", return_value=mock_resp):
            result = LiteLLMMessagesToChatCompletionHandler.anthropic_messages_handler(
                max_tokens=1024,
                messages=[{"role": "user", "content": "What's the weather?"}],
                model="openai/gpt-5.2",
            )

        content = result["content"]
        assert len(content) == 3
        assert content[0]["type"] == "thinking"
        assert content[1]["type"] == "text"
        assert content[2]["type"] == "tool_use"
        assert content[2]["id"] == "call_abc"

    def test_should_not_fabricate_thinking_when_absent(self):
        """When chat response has no thinking fields, no thinking block should appear."""
        from litellm.llms.anthropic.experimental_pass_through.chat_adapters.handler import (
            LiteLLMMessagesToChatCompletionHandler,
        )

        mock_resp = _make_response(content="The answer is 42.")

        with patch("litellm.completion", return_value=mock_resp):
            result = LiteLLMMessagesToChatCompletionHandler.anthropic_messages_handler(
                max_tokens=1024,
                messages=[{"role": "user", "content": "What is the answer?"}],
                model="openai/gpt-5.2",
            )

        content = result["content"]
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "The answer is 42."

    def test_should_preserve_thinking_with_text_ordering(self):
        """When thinking + text, thinking comes first, then text."""
        from litellm.llms.anthropic.experimental_pass_through.chat_adapters.handler import (
            LiteLLMMessagesToChatCompletionHandler,
        )

        mock_resp = _make_response(
            content="Final answer.",
            thinking_blocks=[
                {
                    "type": "thinking",
                    "thinking": "Deep thought.",
                    "signature": "sig_deep",
                },
            ],
        )

        with patch("litellm.completion", return_value=mock_resp):
            result = LiteLLMMessagesToChatCompletionHandler.anthropic_messages_handler(
                max_tokens=1024,
                messages=[{"role": "user", "content": "Think about it."}],
                model="openai/gpt-5.2",
            )

        content = result["content"]
        assert len(content) == 2
        assert content[0]["type"] == "thinking"
        assert content[0]["thinking"] == "Deep thought."
        assert content[0]["signature"] == "sig_deep"
        assert content[1]["type"] == "text"
        assert content[1]["text"] == "Final answer."


class TestChatAdaptersRouting:
    """Verify the new handler routes through litellm.completion, not litellm.responses."""

    def test_should_call_completion_not_responses_for_openai(self):
        """OpenAI models should call litellm.completion, not litellm.responses."""
        from litellm.llms.anthropic.experimental_pass_through.chat_adapters.handler import (
            LiteLLMMessagesToChatCompletionHandler,
        )

        mock_resp = _make_response(content="Hello!")

        with patch("litellm.completion", return_value=mock_resp) as mock_comp, patch(
            "litellm.responses"
        ) as mock_resp_api:
            LiteLLMMessagesToChatCompletionHandler.anthropic_messages_handler(
                max_tokens=100,
                messages=[{"role": "user", "content": "Hi"}],
                model="openai/gpt-5.2",
            )
            mock_comp.assert_called_once()
            mock_resp_api.assert_not_called()

    def test_should_not_prefix_model_with_responses(self):
        """Model name should NOT be auto-prefixed with 'responses/' even with thinking."""
        from litellm.llms.anthropic.experimental_pass_through.chat_adapters.handler import (
            LiteLLMMessagesToChatCompletionHandler,
        )

        mock_resp = _make_response(content="Ok")

        with patch("litellm.completion", return_value=mock_resp) as mock_comp:
            LiteLLMMessagesToChatCompletionHandler.anthropic_messages_handler(
                max_tokens=1024,
                messages=[{"role": "user", "content": "Think"}],
                model="openai/gpt-5.2",
                thinking={"type": "enabled", "budget_tokens": 5000},
            )

            call_kwargs = mock_comp.call_args.kwargs
            assert not call_kwargs["model"].startswith(
                "responses/"
            ), f"model should not have responses/ prefix, got {call_kwargs['model']}"

    def test_should_pass_reasoning_effort_for_thinking(self):
        """When thinking is enabled for non-Claude model, reasoning_effort should be in kwargs."""
        from litellm.llms.anthropic.experimental_pass_through.chat_adapters.handler import (
            LiteLLMMessagesToChatCompletionHandler,
        )

        mock_resp = _make_response(content="Ok")

        with patch("litellm.completion", return_value=mock_resp) as mock_comp:
            LiteLLMMessagesToChatCompletionHandler.anthropic_messages_handler(
                max_tokens=1024,
                messages=[{"role": "user", "content": "Think"}],
                model="openai/gpt-5.2",
                thinking={"type": "enabled", "budget_tokens": 5000},
            )

            call_kwargs = mock_comp.call_args.kwargs
            assert "reasoning_effort" in call_kwargs
            assert "thinking" not in call_kwargs

    def test_should_forward_api_key_and_base(self):
        """api_key and api_base should be forwarded to litellm.completion."""
        from litellm.llms.anthropic.experimental_pass_through.chat_adapters.handler import (
            LiteLLMMessagesToChatCompletionHandler,
        )

        mock_resp = _make_response(content="Ok")

        with patch("litellm.completion", return_value=mock_resp) as mock_comp:
            LiteLLMMessagesToChatCompletionHandler.anthropic_messages_handler(
                max_tokens=100,
                messages=[{"role": "user", "content": "Hi"}],
                model="openai/gpt-5.2",
                api_key="sk-test-key",
                api_base="https://custom.endpoint.com",
            )

            call_kwargs = mock_comp.call_args.kwargs
            assert call_kwargs["api_key"] == "sk-test-key"
            assert call_kwargs["api_base"] == "https://custom.endpoint.com"
