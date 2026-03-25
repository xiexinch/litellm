"""
Facade adapter for Anthropic /v1/messages -> OpenAI Chat Completions.

This module provides thin wrappers around the existing translation functions
in ``adapters.transformation`` so that the chat-completions path has a
stable, self-contained interface symmetric with ``responses_adapters``.
"""

from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

from litellm.types.llms.anthropic import AnthropicMessagesRequest
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.types.llms.openai import ChatCompletionRequest
from litellm.types.utils import ModelResponse

from ..adapters.transformation import AnthropicAdapter, LiteLLMAnthropicMessagesAdapter

# Single shared adapter instance (stateless, safe to reuse).
_ADAPTER = AnthropicAdapter()
_MSG_ADAPTER = LiteLLMAnthropicMessagesAdapter()


class LiteLLMAnthropicToOpenAIChatAdapter:
    """Translate between Anthropic Messages and OpenAI Chat Completions."""

    # -- request translation ---------------------------------------------------

    @staticmethod
    def translate_request(
        request_data: Dict[str, Any],
    ) -> Tuple[ChatCompletionRequest, Dict[str, str]]:
        """Convert an Anthropic-style request dict to an OpenAI ChatCompletionRequest.

        Returns:
            (openai_request, tool_name_mapping)
        """
        return _ADAPTER.translate_completion_input_params_with_tool_mapping(
            request_data
        )

    # -- response translation --------------------------------------------------

    @staticmethod
    def translate_response(
        response: ModelResponse,
        tool_name_mapping: Optional[Dict[str, str]] = None,
    ) -> AnthropicMessagesResponse:
        """Convert an OpenAI ModelResponse to an Anthropic MessagesResponse."""
        result = _ADAPTER.translate_completion_output_params(
            response, tool_name_mapping=tool_name_mapping
        )
        if result is None:
            raise ValueError("Failed to translate response to Anthropic format")
        return result

    # -- streaming translation -------------------------------------------------

    @staticmethod
    def translate_streaming_response(
        completion_stream: Any,
        model: str,
        tool_name_mapping: Optional[Dict[str, str]] = None,
    ) -> Union[AsyncIterator[bytes], None]:
        """Wrap an OpenAI streaming response as Anthropic SSE events."""
        return _ADAPTER.translate_completion_output_params_streaming(
            completion_stream,
            model=model,
            tool_name_mapping=tool_name_mapping,
        )
