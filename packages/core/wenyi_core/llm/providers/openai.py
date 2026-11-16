"""Call models through the official OpenAI Chat Completions endpoint."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..transport import Messages, RequestContext, ResolvedModel
from ..usage import UsageSample
from ._openai_compatible import (
    OpenAICompatibleBaseClient,
    base_request_kwargs,
    deep_merge,
    normalize_openai_usage,
)

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"


class OpenAIOptions(BaseModel):
    """OpenAI-specific model request options."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    thinking: bool = True
    reasoning_effort: str = "high"
    extra_body: dict[str, Any] = Field(default_factory=dict)


def build_request_kwargs(
    model_config: ResolvedModel[OpenAIOptions],
    messages: Messages,
    *,
    json_mode: bool = False,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Build OpenAI request arguments and limit output with max_completion_tokens."""
    kwargs = base_request_kwargs(model_config.model, messages, json_mode=json_mode)
    kwargs["reasoning_effort"] = (
        model_config.options.reasoning_effort if model_config.options.thinking else "none"
    )
    if model_config.options.extra_body:
        kwargs["extra_body"] = deep_merge({}, model_config.options.extra_body)
    if max_tokens is not None:
        kwargs["max_completion_tokens"] = max_tokens
    return kwargs


class OpenAIClient(OpenAICompatibleBaseClient[OpenAIOptions]):
    supports_response_schema = True
    protocol_version = 2
    default_base_url = DEFAULT_BASE_URL
    default_api_key_env = DEFAULT_API_KEY_ENV
    requires_api_key = True

    def _request(
        self,
        messages: Messages,
        model: ResolvedModel[OpenAIOptions],
        *,
        json_mode: bool,
        context: RequestContext,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        schema = context.response_schema
        if not json_mode or schema is None or "anyOf" not in schema:
            return super()._request(
                messages,
                model,
                json_mode=json_mode,
                context=context,
                response_format=response_format,
            )

        # OpenAI permits unions as properties, but rejects an anyOf at the response root.
        # Keep local JSON Pointers relative to the original schema after adding the envelope.
        def relocate(value: Any) -> Any:
            if isinstance(value, dict):
                if "$id" in value:
                    raise ValueError(
                        "OpenAI root-union schemas with resource identifiers are not supported"
                    )
                return {
                    key: f"#/properties/response{item[1:]}"
                    if key == "$ref"
                    and isinstance(item, str)
                    and (item == "#" or item.startswith("#/"))
                    else deepcopy(item)
                    if key in {"const", "enum", "default", "examples"}
                    else relocate(item)
                    for key, item in value.items()
                }
            if isinstance(value, list):
                return [relocate(item) for item in value]
            return value

        inner_schema = relocate(schema)
        if inner_schema.get("type") == "object" and all(
            branch.get("type") == "object" for branch in inner_schema["anyOf"]
        ):
            # The branches already enforce object type. An otherwise empty object container
            # fails OpenAI's requirement that every explicit object declare closed properties.
            del inner_schema["type"]
        wire_schema = {
            "type": "object",
            "properties": {"response": inner_schema},
            "required": ["response"],
            "additionalProperties": False,
        }
        raw = super()._request(
            messages,
            model,
            json_mode=True,
            context=replace(context, response_schema=wire_schema),
            response_format=response_format,
        )
        try:
            envelope = json.loads(raw)
        except ValueError:
            raise ValueError("OpenAI structured response envelope is invalid") from None
        if not isinstance(envelope, dict) or set(envelope) != {"response"}:
            raise ValueError("OpenAI structured response envelope is invalid")
        return json.dumps(envelope["response"], ensure_ascii=False)

    def _normalize_usage(self, usage: Any) -> UsageSample | None:
        return normalize_openai_usage(usage, separate_cache_writes=False)

    def _build_request_kwargs(
        self,
        model_config: ResolvedModel[OpenAIOptions],
        messages: Messages,
        *,
        json_mode: bool,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        """Build final request arguments for the selected OpenAI tier."""
        return build_request_kwargs(
            model_config,
            messages,
            json_mode=json_mode,
            max_tokens=max_tokens,
        )
