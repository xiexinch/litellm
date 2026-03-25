"""
Handler for Anthropic /v1/messages -> OpenAI Chat Completions path.

This is the **default** adapter for non-native providers (e.g. OpenAI,
Azure, custom providers).  It calls ``litellm.completion`` /
``litellm.acompletion`` — never ``litellm.responses``.

Unlike the legacy ``adapters.handler``, this handler does NOT auto-upgrade
OpenAI+thinking requests to the Responses API.  Thinking parameters are
mapped to ``reasoning_effort`` for the chat endpoint; thinking *output*
from the response is faithfully converted back to Anthropic thinking blocks.
"""

from typing import (
    Any,
    AsyncIterator,
    Coroutine,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import litellm
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.types.utils import ModelResponse

from .transformation import LiteLLMAnthropicToOpenAIChatAdapter

_ADAPTER = LiteLLMAnthropicToOpenAIChatAdapter()


def _prepare_chat_completion_kwargs(
    *,
    max_tokens: int,
    messages: List[Dict],
    model: str,
    metadata: Optional[Dict] = None,
    stop_sequences: Optional[List[str]] = None,
    stream: Optional[bool] = False,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    thinking: Optional[Dict] = None,
    tool_choice: Optional[Dict] = None,
    tools: Optional[List[Dict]] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    output_format: Optional[Dict] = None,
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Build kwargs for ``litellm.completion`` / ``litellm.acompletion``.

    This is intentionally very similar to ``adapters.handler._prepare_completion_kwargs``
    but it **does not** call ``_route_openai_thinking_to_responses_api_if_needed``.

    Returns:
        (completion_kwargs, tool_name_mapping)
    """
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LiteLLMLoggingObject,
    )

    request_data: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }

    if metadata:
        request_data["metadata"] = metadata
    if stop_sequences:
        request_data["stop_sequences"] = stop_sequences
    if system:
        request_data["system"] = system
    if temperature is not None:
        request_data["temperature"] = temperature
    if thinking:
        request_data["thinking"] = thinking
    if tool_choice:
        request_data["tool_choice"] = tool_choice
    if tools:
        request_data["tools"] = tools
    if top_k is not None:
        request_data["top_k"] = top_k
    if top_p is not None:
        request_data["top_p"] = top_p
    if output_format:
        request_data["output_format"] = output_format

    openai_request, tool_name_mapping = _ADAPTER.translate_request(request_data)

    if openai_request is None:
        raise ValueError("Failed to translate request to OpenAI format")

    completion_kwargs: Dict[str, Any] = dict(openai_request)

    if stream:
        completion_kwargs["stream"] = stream
        completion_kwargs["stream_options"] = {
            "include_usage": True,
        }

    excluded_keys = {"anthropic_messages"}
    extra_kwargs = extra_kwargs or {}
    for key, value in extra_kwargs.items():
        if (
            key == "litellm_logging_obj"
            and value is not None
            and isinstance(value, LiteLLMLoggingObject)
        ):
            from litellm.types.utils import CallTypes

            setattr(value, "call_type", CallTypes.completion.value)
            setattr(value, "stream_options", completion_kwargs.get("stream_options"))
        if (
            key not in excluded_keys
            and key not in completion_kwargs
            and value is not None
        ):
            completion_kwargs[key] = value

    # NOTE: No _route_openai_thinking_to_responses_api_if_needed() call here.
    # Thinking input is already handled by translate_thinking_for_model() inside
    # the adapter's translate_request(), which maps thinking -> reasoning_effort
    # for non-Claude models.  We do NOT auto-upgrade to the Responses API.

    return completion_kwargs, tool_name_mapping


class LiteLLMMessagesToChatCompletionHandler:
    """Route Anthropic /v1/messages to OpenAI Chat Completions.

    This handler calls ``litellm.completion`` (sync) or ``litellm.acompletion``
    (async), converts both the request and response between Anthropic and
    OpenAI formats.  It faithfully passes through thinking output from the
    chat endpoint response.
    """

    @staticmethod
    async def async_anthropic_messages_handler(
        max_tokens: int,
        messages: List[Dict],
        model: str,
        metadata: Optional[Dict] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: Optional[bool] = False,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        thinking: Optional[Dict] = None,
        tool_choice: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        output_format: Optional[Dict] = None,
        **kwargs,
    ) -> Union[AnthropicMessagesResponse, AsyncIterator]:
        completion_kwargs, tool_name_mapping = _prepare_chat_completion_kwargs(
            max_tokens=max_tokens,
            messages=messages,
            model=model,
            metadata=metadata,
            stop_sequences=stop_sequences,
            stream=stream,
            system=system,
            temperature=temperature,
            thinking=thinking,
            tool_choice=tool_choice,
            tools=tools,
            top_k=top_k,
            top_p=top_p,
            output_format=output_format,
            extra_kwargs=kwargs,
        )

        completion_response = await litellm.acompletion(**completion_kwargs)

        if stream:
            transformed_stream = _ADAPTER.translate_streaming_response(
                completion_response,
                model=model,
                tool_name_mapping=tool_name_mapping,
            )
            if transformed_stream is not None:
                return transformed_stream
            raise ValueError("Failed to transform streaming response")

        anthropic_response = _ADAPTER.translate_response(
            cast(ModelResponse, completion_response),
            tool_name_mapping=tool_name_mapping,
        )
        return anthropic_response

    @staticmethod
    def anthropic_messages_handler(
        max_tokens: int,
        messages: List[Dict],
        model: str,
        metadata: Optional[Dict] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: Optional[bool] = False,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        thinking: Optional[Dict] = None,
        tool_choice: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        output_format: Optional[Dict] = None,
        _is_async: bool = False,
        **kwargs,
    ) -> Union[
        AnthropicMessagesResponse,
        AsyncIterator[Any],
        Coroutine[Any, Any, Union[AnthropicMessagesResponse, AsyncIterator[Any]]],
    ]:
        if _is_async:
            return (
                LiteLLMMessagesToChatCompletionHandler.async_anthropic_messages_handler(
                    max_tokens=max_tokens,
                    messages=messages,
                    model=model,
                    metadata=metadata,
                    stop_sequences=stop_sequences,
                    stream=stream,
                    system=system,
                    temperature=temperature,
                    thinking=thinking,
                    tool_choice=tool_choice,
                    tools=tools,
                    top_k=top_k,
                    top_p=top_p,
                    output_format=output_format,
                    **kwargs,
                )
            )

        completion_kwargs, tool_name_mapping = _prepare_chat_completion_kwargs(
            max_tokens=max_tokens,
            messages=messages,
            model=model,
            metadata=metadata,
            stop_sequences=stop_sequences,
            stream=stream,
            system=system,
            temperature=temperature,
            thinking=thinking,
            tool_choice=tool_choice,
            tools=tools,
            top_k=top_k,
            top_p=top_p,
            output_format=output_format,
            extra_kwargs=kwargs,
        )

        completion_response = litellm.completion(**completion_kwargs)

        if stream:
            transformed_stream = _ADAPTER.translate_sync_streaming_response(
                completion_response,
                model=model,
                tool_name_mapping=tool_name_mapping,
            )
            return transformed_stream

        anthropic_response = _ADAPTER.translate_response(
            cast(ModelResponse, completion_response),
            tool_name_mapping=tool_name_mapping,
        )
        return anthropic_response
