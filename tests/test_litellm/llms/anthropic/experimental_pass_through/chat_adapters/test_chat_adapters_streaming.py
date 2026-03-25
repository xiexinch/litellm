"""
Tests for streaming through the chat_adapters handler.

Covers: text streaming, tool call streaming, thinking streaming,
usage merging, and sync/async parity for the
Anthropic /v1/messages -> OpenAI Chat Completions -> Anthropic SSE path.
"""

import asyncio
import json
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Delta,
    Function,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)


# ---------------------------------------------------------------------------
# Helpers: build mock streaming chunks
# ---------------------------------------------------------------------------


def _text_chunk(
    text: str, finish_reason: Optional[str] = None
) -> ModelResponseStream:
    return ModelResponseStream(
        id="chatcmpl-test",
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(content=text, role="assistant"),
                finish_reason=finish_reason,
            )
        ],
    )


def _tool_call_start_chunk(
    tool_id: str, tool_name: str
) -> ModelResponseStream:
    """First chunk for a tool call: carries the function name."""
    return ModelResponseStream(
        id="chatcmpl-test",
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(
                    tool_calls=[
                        ChatCompletionDeltaToolCall(
                            id=tool_id,
                            type="function",
                            index=0,
                            function=Function(name=tool_name, arguments=""),
                        ),
                    ]
                ),
                finish_reason=None,
            )
        ],
    )


def _tool_call_args_chunk(partial_json: str) -> ModelResponseStream:
    """Subsequent chunks with partial JSON arguments."""
    return ModelResponseStream(
        id="chatcmpl-test",
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(
                    tool_calls=[
                        ChatCompletionDeltaToolCall(
                            id=None,
                            type="function",
                            index=0,
                            function=Function(name=None, arguments=partial_json),
                        ),
                    ]
                ),
                finish_reason=None,
            )
        ],
    )


def _finish_chunk(
    reason: str = "stop",
    usage: Optional[Usage] = None,
) -> ModelResponseStream:
    chunk = ModelResponseStream(
        id="chatcmpl-test",
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(),
                finish_reason=reason,
            )
        ],
    )
    if usage is not None:
        chunk.usage = usage
    return chunk


def _usage_chunk(
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> ModelResponseStream:
    """A trailing usage-only chunk (stream_options.include_usage=True)."""
    chunk = ModelResponseStream(
        id="chatcmpl-test",
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(),
                finish_reason=None,
            )
        ],
        usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )
    return chunk


def _thinking_chunk(
    thinking: str = "", signature: str = ""
) -> ModelResponseStream:
    return ModelResponseStream(
        id="chatcmpl-test",
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(
                    thinking_blocks=[
                        {
                            "type": "thinking",
                            "thinking": thinking,
                            "signature": signature,
                        }
                    ]
                ),
                finish_reason=None,
            )
        ],
    )


# ---------------------------------------------------------------------------
# Helpers: collect SSE events from sync/async iterators
# ---------------------------------------------------------------------------


def _collect_sync_events(iterator: Iterator[bytes]) -> List[Dict[str, Any]]:
    """Parse SSE byte payloads into a list of dicts."""
    events = []
    for raw in iterator:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        for line in text.strip().split("\n"):
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: "):]))
    return events


async def _collect_async_events(
    aiterator: AsyncIterator[bytes],
) -> List[Dict[str, Any]]:
    """Parse async SSE byte payloads into a list of dicts."""
    events = []
    async for raw in aiterator:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        for line in text.strip().split("\n"):
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: "):]))
    return events


# ---------------------------------------------------------------------------
# Sync streaming tests
# ---------------------------------------------------------------------------


class TestSyncStreaming:
    """Verify sync streaming through chat_adapters handler."""

    def _make_sync_stream(self, chunks: List[ModelResponseStream]):
        """Create a mock sync iterator from a list of chunks."""
        return iter(chunks)

    def test_should_stream_text_events(self):
        """Basic text streaming: message_start -> content_block_start -> deltas -> stop."""
        from litellm.llms.anthropic.experimental_pass_through.chat_adapters.handler import (
            LiteLLMMessagesToChatCompletionHandler,
        )

        chunks = [
            _text_chunk("Hello"),
            _text_chunk(" world"),
            _finish_chunk("stop"),
            _usage_chunk(10, 5),
        ]

        with patch(
            "litellm.completion", return_value=self._make_sync_stream(chunks)
        ):
            result = LiteLLMMessagesToChatCompletionHandler.anthropic_messages_handler(
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hi"}],
                model="openai/gpt-4",
                stream=True,
            )

        events = _collect_sync_events(result)
        types = [e["type"] for e in events]

        assert types[0] == "message_start"
        assert "content_block_start" in types
        # At least one content_block_delta
        deltas = [e for e in events if e["type"] == "content_block_delta"]
        assert len(deltas) >= 1
        # Check text content
        text_deltas = [
            e["delta"]["text"]
            for e in deltas
            if e["delta"].get("type") == "text_delta"
        ]
        assert "Hello" in text_deltas
        assert " world" in text_deltas
        assert "content_block_stop" in types
        assert "message_delta" in types
        assert types[-1] == "message_stop"

    def test_should_stream_tool_call_events(self):
        """Tool call streaming: text -> tool_use content block switch."""
        from litellm.llms.anthropic.experimental_pass_through.chat_adapters.handler import (
            LiteLLMMessagesToChatCompletionHandler,
        )

        chunks = [
            _text_chunk("Let me search."),
            _tool_call_start_chunk("call_123", "web_search"),
            _tool_call_args_chunk('{"query":'),
            _tool_call_args_chunk(' "weather"}'),
            _finish_chunk("tool_calls"),
            _usage_chunk(15, 8),
        ]

        with patch(
            "litellm.completion", return_value=self._make_sync_stream(chunks)
        ):
            result = LiteLLMMessagesToChatCompletionHandler.anthropic_messages_handler(
                max_tokens=1024,
                messages=[{"role": "user", "content": "Weather?"}],
                model="openai/gpt-4",
                stream=True,
            )

        events = _collect_sync_events(result)
        types = [e["type"] for e in events]

        # Should have at least two content_block_start events (text + tool_use)
        block_starts = [e for e in events if e["type"] == "content_block_start"]
        assert len(block_starts) >= 2

        # Find the tool_use content block start
        tool_starts = [
            e
            for e in block_starts
            if e["content_block"].get("type") == "tool_use"
        ]
        assert len(tool_starts) == 1
        assert tool_starts[0]["content_block"]["name"] == "web_search"

        # Should have input_json_delta deltas
        json_deltas = [
            e
            for e in events
            if e.get("type") == "content_block_delta"
            and e.get("delta", {}).get("type") == "input_json_delta"
        ]
        assert len(json_deltas) >= 1

        # message_delta should have stop_reason = tool_use
        msg_deltas = [e for e in events if e["type"] == "message_delta"]
        assert len(msg_deltas) >= 1
        assert msg_deltas[0]["delta"]["stop_reason"] == "tool_use"

    def test_should_stream_thinking_events(self):
        """Thinking streaming: thinking content_block -> text content_block."""
        from litellm.llms.anthropic.experimental_pass_through.chat_adapters.handler import (
            LiteLLMMessagesToChatCompletionHandler,
        )

        chunks = [
            _thinking_chunk(thinking="Let me think..."),
            _thinking_chunk(thinking=" step by step."),
            _thinking_chunk(signature="sig_abc"),
            _text_chunk("The answer is 42."),
            _finish_chunk("stop"),
            _usage_chunk(20, 10),
        ]

        with patch(
            "litellm.completion", return_value=self._make_sync_stream(chunks)
        ):
            result = LiteLLMMessagesToChatCompletionHandler.anthropic_messages_handler(
                max_tokens=4096,
                messages=[{"role": "user", "content": "Think hard"}],
                model="openai/gpt-4",
                stream=True,
                thinking={"type": "enabled", "budget_tokens": 5000},
            )

        events = _collect_sync_events(result)
        types = [e["type"] for e in events]

        # Should have thinking_delta events
        thinking_deltas = [
            e
            for e in events
            if e.get("type") == "content_block_delta"
            and e.get("delta", {}).get("type") == "thinking_delta"
        ]
        assert len(thinking_deltas) >= 1

        # Should have text_delta after thinking
        text_deltas = [
            e
            for e in events
            if e.get("type") == "content_block_delta"
            and e.get("delta", {}).get("type") == "text_delta"
            and e.get("delta", {}).get("text")
        ]
        assert len(text_deltas) >= 1

        assert "message_stop" in types

    def test_should_include_usage_in_message_delta(self):
        """Usage from the trailing chunk should be merged into message_delta."""
        from litellm.llms.anthropic.experimental_pass_through.chat_adapters.handler import (
            LiteLLMMessagesToChatCompletionHandler,
        )

        chunks = [
            _text_chunk("Hi"),
            _finish_chunk("stop"),
            _usage_chunk(prompt_tokens=25, completion_tokens=12),
        ]

        with patch(
            "litellm.completion", return_value=self._make_sync_stream(chunks)
        ):
            result = LiteLLMMessagesToChatCompletionHandler.anthropic_messages_handler(
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hi"}],
                model="openai/gpt-4",
                stream=True,
            )

        events = _collect_sync_events(result)
        msg_deltas = [e for e in events if e["type"] == "message_delta"]
        assert len(msg_deltas) >= 1
        usage = msg_deltas[0].get("usage", {})
        assert usage["input_tokens"] == 25
        assert usage["output_tokens"] == 12

    def test_should_restore_truncated_tool_names(self):
        """Tool names truncated for OpenAI's 64-char limit should be restored."""
        from litellm.llms.anthropic.experimental_pass_through.chat_adapters.handler import (
            LiteLLMMessagesToChatCompletionHandler,
        )

        long_name = "a_very_long_tool_name_that_exceeds_sixty_four_characters_and_must_be_truncated"

        chunks = [
            _tool_call_start_chunk("call_456", long_name[:55] + "_" + "abcd1234"),
            _tool_call_args_chunk('{"key": "val"}'),
            _finish_chunk("tool_calls"),
            _usage_chunk(5, 3),
        ]

        # Mock the translation to set up tool_name_mapping
        with patch(
            "litellm.completion", return_value=self._make_sync_stream(chunks)
        ), patch(
            "litellm.llms.anthropic.experimental_pass_through.chat_adapters.transformation.LiteLLMAnthropicToOpenAIChatAdapter.translate_request"
        ) as mock_translate:
            truncated_name = long_name[:55] + "_" + "abcd1234"
            mock_translate.return_value = (
                {
                    "model": "openai/gpt-4",
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 1024,
                },
                {truncated_name: long_name},
            )

            result = LiteLLMMessagesToChatCompletionHandler.anthropic_messages_handler(
                max_tokens=1024,
                messages=[{"role": "user", "content": "test"}],
                model="openai/gpt-4",
                stream=True,
                tools=[{"name": long_name, "type": "custom", "input_schema": {}}],
            )

        events = _collect_sync_events(result)
        tool_starts = [
            e
            for e in events
            if e.get("type") == "content_block_start"
            and e.get("content_block", {}).get("type") == "tool_use"
        ]
        assert len(tool_starts) == 1
        assert tool_starts[0]["content_block"]["name"] == long_name


# ---------------------------------------------------------------------------
# Async streaming tests
# ---------------------------------------------------------------------------


class TestAsyncStreaming:
    """Verify async streaming through chat_adapters handler."""

    @staticmethod
    async def _make_async_stream(chunks: List[ModelResponseStream]):
        """Create a mock async iterator from a list of chunks."""
        for chunk in chunks:
            yield chunk

    @pytest.mark.asyncio
    async def test_should_stream_text_events_async(self):
        """Basic async text streaming: message_start -> deltas -> stop."""
        from litellm.llms.anthropic.experimental_pass_through.chat_adapters.handler import (
            LiteLLMMessagesToChatCompletionHandler,
        )

        chunks = [
            _text_chunk("Hello"),
            _text_chunk(" async"),
            _finish_chunk("stop"),
            _usage_chunk(10, 5),
        ]

        with patch(
            "litellm.acompletion",
            return_value=self._make_async_stream(chunks),
        ):
            result = await LiteLLMMessagesToChatCompletionHandler.async_anthropic_messages_handler(
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hi"}],
                model="openai/gpt-4",
                stream=True,
            )

        events = await _collect_async_events(result)
        types = [e["type"] for e in events]

        assert types[0] == "message_start"
        assert "content_block_start" in types
        deltas = [e for e in events if e["type"] == "content_block_delta"]
        assert len(deltas) >= 1
        text_deltas = [
            e["delta"]["text"]
            for e in deltas
            if e["delta"].get("type") == "text_delta"
        ]
        assert "Hello" in text_deltas
        assert " async" in text_deltas
        assert types[-1] == "message_stop"

    @pytest.mark.asyncio
    async def test_should_stream_tool_call_events_async(self):
        """Async tool call streaming with content block transitions."""
        from litellm.llms.anthropic.experimental_pass_through.chat_adapters.handler import (
            LiteLLMMessagesToChatCompletionHandler,
        )

        chunks = [
            _tool_call_start_chunk("call_789", "calculator"),
            _tool_call_args_chunk('{"expr":'),
            _tool_call_args_chunk(' "2+2"}'),
            _finish_chunk("tool_calls"),
            _usage_chunk(10, 5),
        ]

        with patch(
            "litellm.acompletion",
            return_value=self._make_async_stream(chunks),
        ):
            result = await LiteLLMMessagesToChatCompletionHandler.async_anthropic_messages_handler(
                max_tokens=1024,
                messages=[{"role": "user", "content": "calc"}],
                model="openai/gpt-4",
                stream=True,
            )

        events = await _collect_async_events(result)

        tool_starts = [
            e
            for e in events
            if e.get("type") == "content_block_start"
            and e.get("content_block", {}).get("type") == "tool_use"
        ]
        assert len(tool_starts) == 1
        assert tool_starts[0]["content_block"]["name"] == "calculator"

        json_deltas = [
            e
            for e in events
            if e.get("type") == "content_block_delta"
            and e.get("delta", {}).get("type") == "input_json_delta"
        ]
        assert len(json_deltas) >= 1

    @pytest.mark.asyncio
    async def test_should_include_usage_in_message_delta_async(self):
        """Usage data should be merged into the message_delta event."""
        from litellm.llms.anthropic.experimental_pass_through.chat_adapters.handler import (
            LiteLLMMessagesToChatCompletionHandler,
        )

        chunks = [
            _text_chunk("Done"),
            _finish_chunk("stop"),
            _usage_chunk(prompt_tokens=30, completion_tokens=15),
        ]

        with patch(
            "litellm.acompletion",
            return_value=self._make_async_stream(chunks),
        ):
            result = await LiteLLMMessagesToChatCompletionHandler.async_anthropic_messages_handler(
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hi"}],
                model="openai/gpt-4",
                stream=True,
            )

        events = await _collect_async_events(result)
        msg_deltas = [e for e in events if e["type"] == "message_delta"]
        assert len(msg_deltas) >= 1
        usage = msg_deltas[0].get("usage", {})
        assert usage["input_tokens"] == 30
        assert usage["output_tokens"] == 15

    @pytest.mark.asyncio
    async def test_should_stream_thinking_then_text_async(self):
        """Async thinking streaming followed by text."""
        from litellm.llms.anthropic.experimental_pass_through.chat_adapters.handler import (
            LiteLLMMessagesToChatCompletionHandler,
        )

        chunks = [
            _thinking_chunk(thinking="Step 1."),
            _thinking_chunk(signature="sig_xyz"),
            _text_chunk("Result."),
            _finish_chunk("stop"),
            _usage_chunk(20, 10),
        ]

        with patch(
            "litellm.acompletion",
            return_value=self._make_async_stream(chunks),
        ):
            result = await LiteLLMMessagesToChatCompletionHandler.async_anthropic_messages_handler(
                max_tokens=4096,
                messages=[{"role": "user", "content": "Think"}],
                model="openai/gpt-4",
                stream=True,
                thinking={"type": "enabled", "budget_tokens": 5000},
            )

        events = await _collect_async_events(result)

        thinking_deltas = [
            e
            for e in events
            if e.get("type") == "content_block_delta"
            and e.get("delta", {}).get("type") == "thinking_delta"
        ]
        assert len(thinking_deltas) >= 1

        text_deltas = [
            e
            for e in events
            if e.get("type") == "content_block_delta"
            and e.get("delta", {}).get("type") == "text_delta"
            and e.get("delta", {}).get("text")
        ]
        assert len(text_deltas) >= 1


# ---------------------------------------------------------------------------
# AnthropicStreamWrapper unit tests (lower-level)
# ---------------------------------------------------------------------------


class TestAnthropicStreamWrapperSync:
    """Direct tests on the sync iteration of AnthropicStreamWrapper."""

    def test_should_emit_correct_event_sequence_for_text(self):
        """Verify the full event sequence for a simple text stream."""
        from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
            AnthropicStreamWrapper,
        )

        chunks = [
            _text_chunk("Hi"),
            _finish_chunk("stop"),
            _usage_chunk(10, 5),
        ]

        wrapper = AnthropicStreamWrapper(
            completion_stream=iter(chunks),
            model="test-model",
        )

        events = []
        for event in wrapper:
            events.append(event)

        types = [e["type"] for e in events]
        assert types[0] == "message_start"
        assert types[1] == "content_block_start"
        assert "content_block_delta" in types
        assert "content_block_stop" in types
        assert "message_delta" in types
        assert types[-1] == "message_stop"

    def test_should_emit_content_block_transitions_for_tool_calls(self):
        """Text -> tool_use should produce content_block_stop + content_block_start."""
        from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
            AnthropicStreamWrapper,
        )

        chunks = [
            _text_chunk("Searching..."),
            _tool_call_start_chunk("call_1", "search"),
            _tool_call_args_chunk('{"q": "test"}'),
            _finish_chunk("tool_calls"),
            _usage_chunk(10, 5),
        ]

        wrapper = AnthropicStreamWrapper(
            completion_stream=iter(chunks),
            model="test-model",
        )

        events = list(wrapper)
        block_starts = [e for e in events if e["type"] == "content_block_start"]
        block_stops = [e for e in events if e["type"] == "content_block_stop"]

        # At least 2 content block starts (text + tool_use)
        assert len(block_starts) >= 2
        # At least 2 content block stops
        assert len(block_stops) >= 2

        # Second block start should be tool_use
        tool_block = next(
            (
                e
                for e in block_starts
                if e["content_block"].get("type") == "tool_use"
            ),
            None,
        )
        assert tool_block is not None
        assert tool_block["content_block"]["name"] == "search"

    def test_should_merge_usage_into_message_delta_sync(self):
        """The trailing usage chunk should be merged into message_delta."""
        from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
            AnthropicStreamWrapper,
        )

        chunks = [
            _text_chunk("ok"),
            _finish_chunk("stop"),
            _usage_chunk(prompt_tokens=42, completion_tokens=7),
        ]

        wrapper = AnthropicStreamWrapper(
            completion_stream=iter(chunks),
            model="test-model",
        )

        events = list(wrapper)
        msg_deltas = [e for e in events if e["type"] == "message_delta"]
        assert len(msg_deltas) >= 1
        usage = msg_deltas[0].get("usage", {})
        assert usage.get("input_tokens") == 42
        assert usage.get("output_tokens") == 7

    def test_should_restore_tool_name_from_mapping(self):
        """Truncated tool names should be restored via tool_name_mapping."""
        from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
            AnthropicStreamWrapper,
        )

        original_name = "my_very_long_tool_name_that_exceeds_the_openai_limit_and_needs_truncation"
        truncated_name = "my_very_long_tool_name_that_exceeds_the_openai_limit_a_12345678"

        chunks = [
            _tool_call_start_chunk("call_x", truncated_name),
            _tool_call_args_chunk("{}"),
            _finish_chunk("tool_calls"),
            _usage_chunk(5, 3),
        ]

        wrapper = AnthropicStreamWrapper(
            completion_stream=iter(chunks),
            model="test-model",
            tool_name_mapping={truncated_name: original_name},
        )

        events = list(wrapper)
        tool_starts = [
            e
            for e in events
            if e.get("type") == "content_block_start"
            and e.get("content_block", {}).get("type") == "tool_use"
        ]
        assert len(tool_starts) == 1
        assert tool_starts[0]["content_block"]["name"] == original_name

    def test_should_handle_parallel_tool_calls(self):
        """Two sequential tool calls should produce separate content blocks."""
        from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
            AnthropicStreamWrapper,
        )

        chunks = [
            _tool_call_start_chunk("call_a", "tool_1"),
            _tool_call_args_chunk('{"a": 1}'),
            _tool_call_start_chunk("call_b", "tool_2"),
            _tool_call_args_chunk('{"b": 2}'),
            _finish_chunk("tool_calls"),
            _usage_chunk(10, 5),
        ]

        wrapper = AnthropicStreamWrapper(
            completion_stream=iter(chunks),
            model="test-model",
        )

        events = list(wrapper)
        tool_starts = [
            e
            for e in events
            if e.get("type") == "content_block_start"
            and e.get("content_block", {}).get("type") == "tool_use"
        ]
        # Should have 2 tool_use content block starts
        assert len(tool_starts) == 2
        names = [e["content_block"]["name"] for e in tool_starts]
        assert "tool_1" in names
        assert "tool_2" in names
