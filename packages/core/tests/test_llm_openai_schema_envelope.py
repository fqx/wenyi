"""Native OpenAI transports root unions without changing the domain response protocol."""

import copy
import json
from contextlib import contextmanager

import httpx
import pytest
from openai import OpenAI
from wenyi_core.llm.configuration import LLMConfig
from wenyi_core.llm.limits import RequestStopped
from wenyi_core.llm.providers._openai_compatible import OpenAICompatibleBaseClient
from wenyi_core.llm.providers.openai import OpenAIClient
from wenyi_core.llm.retrying import ProviderRequestError
from wenyi_core.llm.router import RoutedLLMClient


def _config(*, kind="openai", retries=0, budget=None):
    return LLMConfig.model_validate(
        {
            "providers": {
                "default": {
                    "kind": kind,
                    "base_url": "https://schema.invalid/v1",
                    "max_retries": retries,
                }
            },
            "models": {
                "m": {
                    "provider": "default",
                    "model": "example",
                    "max_output_tokens": 128,
                    "options": {"thinking": False},
                }
            },
            "tiers": {tier: "m" for tier in ("strong", "cheap", "fast")},
            "budget": budget or {},
        }
    )


@pytest.fixture
def root_union():
    return {
        "type": "object",
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "const": action},
                    "complete": {"type": "boolean", "const": complete},
                },
                "required": ["action", "complete"],
                "additionalProperties": False,
            }
            for action, complete in [("final", True), ("request_evidence", False)]
        ],
    }


@contextmanager
def _sdk(monkeypatch, responses, bodies, *, enforce_openai_root=True):
    def valid_objects(value):
        if isinstance(value, dict):
            if value.get("type") == "object" and (
                "properties" not in value or value.get("additionalProperties") is not False
            ):
                return False
            return all(valid_objects(item) for item in value.values())
        if isinstance(value, list):
            return all(valid_objects(item) for item in value)
        return True

    def respond(request):
        body = json.loads(request.content)
        bodies.append(body)
        schema = body["response_format"]["json_schema"]["schema"]
        if enforce_openai_root and ("anyOf" in schema or not valid_objects(schema)):
            return httpx.Response(400, json={"error": {"message": "Invalid root schema"}})
        response = responses.pop(0)
        if isinstance(response, int):
            return httpx.Response(response, json={"error": {"message": "Transient failure"}})
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "example",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(response)},
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
        )

    with OpenAI(
        api_key="test-key",
        base_url="https://schema.invalid/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ) as sdk:
        monkeypatch.setattr(OpenAICompatibleBaseClient, "_ensure_client", lambda self: sdk)
        yield


@pytest.mark.parametrize("action,complete", [("final", True), ("request_evidence", False)])
def test_native_openai_wraps_root_union_and_restores_domain_payload(
    monkeypatch, root_union, action, complete
):
    payload = {"action": action, "complete": complete}
    original_schema = copy.deepcopy(root_union)
    messages = [{"role": "user", "content": "Return the Review action."}]
    original_messages = copy.deepcopy(messages)
    bodies = []
    client = RoutedLLMClient(_config())
    starts = []
    client.set_event_sink(
        lambda event, **data: starts.append(data) if event == "llm_request_started" else None
    )
    with _sdk(monkeypatch, [{"response": payload}], bodies):
        assert (
            client.complete_json(messages, operation="review.verify", response_schema=root_union)
            == payload
        )

    [body] = bodies
    wire = body["response_format"]["json_schema"]
    assert wire["strict"] is True
    assert wire["schema"] == {
        "type": "object",
        "properties": {"response": {"anyOf": root_union["anyOf"]}},
        "required": ["response"],
        "additionalProperties": False,
    }
    assert root_union == original_schema
    assert messages == original_messages
    # The existing fixed protocol allowance also covers the small envelope and JSON instructions.
    wire_input_bytes = sum(
        len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
        for value in [body["messages"], wire["schema"]]
    )
    assert starts[0]["estimated_tokens"] >= wire_input_bytes + body["max_completion_tokens"]
    assert starts[0]["adapter_protocol"] == OpenAIClient.protocol_version
    assert client.usage_summary()["totals"]["calls"] == 1
    assert client.usage_summary()["totals"]["total_tokens"] == 5


def test_native_openai_preserves_non_union_schemas(monkeypatch):
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    payload = {"ok": True}
    bodies = []
    with _sdk(monkeypatch, [payload], bodies):
        assert (
            RoutedLLMClient(_config()).complete_json(
                [], operation="polish.body", response_schema=schema
            )
            == payload
        )
    assert bodies[0]["response_format"]["json_schema"]["schema"] == schema


@pytest.mark.parametrize("kind", ["openai", "openai-compatible"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_array_uniqueness_is_local_for_native_openai_only(monkeypatch, kind, wrapped):
    from wenyi_core.agents.review_schemas import fixer_schema

    inner = fixer_schema("ch1:text0:seg0", "hash", ["issue-1"])
    literal = {"uniqueItems": True}
    inner["properties"]["literal"] = {"const": literal}
    inner["properties"]["uniqueItems"] = {"type": "boolean"}
    inner["required"].extend(["literal", "uniqueItems"])
    schema = {"anyOf": [inner]} if wrapped else inner
    original = copy.deepcopy(schema)
    payload = {"replacement": "corrected"}
    response = {"response": payload} if wrapped and kind == "openai" else payload
    bodies = []
    with _sdk(monkeypatch, [response], bodies, enforce_openai_root=kind == "openai"):
        result = RoutedLLMClient(_config(kind=kind)).complete_json(
            [], operation="review.fix", response_schema=schema
        )
    assert result == payload
    wire = bodies[0]["response_format"]["json_schema"]["schema"]
    if wrapped and kind == "openai":
        wire = wire["properties"]["response"]
    if wrapped:
        wire = wire["anyOf"][0]
    ids = wire["properties"]["issue_ids"]
    assert ("uniqueItems" in ids) == (kind != "openai")
    assert ids["minItems"] == ids["maxItems"] == 1
    assert ids["items"]["enum"] == ["issue-1"]
    assert wire["properties"]["literal"]["const"] == literal
    assert wire["properties"]["uniqueItems"] == {"type": "boolean"}
    assert schema == original


def test_compatible_endpoints_keep_the_original_root_union(monkeypatch, root_union):
    payload = {"action": "final", "complete": True}
    bodies = []
    with _sdk(monkeypatch, [payload], bodies, enforce_openai_root=False):
        assert (
            RoutedLLMClient(_config(kind="openai-compatible")).complete_json(
                [], operation="review.verify", response_schema=root_union
            )
            == payload
        )
    assert bodies[0]["response_format"]["json_schema"]["schema"] == root_union


def test_native_openai_relocates_local_schema_pointers(monkeypatch, root_union):
    schema = copy.deepcopy(root_union)
    schema["$defs"] = {
        "detail": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        }
    }
    schema["anyOf"][0]["properties"]["detail"] = {"$ref": "#/$defs/detail"}
    schema["anyOf"][0]["required"].append("detail")
    payload = {"action": "final", "complete": True, "detail": {"text": "Checked."}}
    bodies = []
    with _sdk(monkeypatch, [{"response": payload}], bodies):
        assert (
            RoutedLLMClient(_config()).complete_json(
                [], operation="review.verify", response_schema=schema
            )
            == payload
        )
    wire = bodies[0]["response_format"]["json_schema"]["schema"]
    reference = wire["properties"]["response"]["anyOf"][0]["properties"]["detail"]["$ref"]
    assert reference == "#/properties/response/$defs/detail"
    target = wire
    for component in reference[2:].split("/"):
        target = target[component]
    assert target == schema["$defs"]["detail"]
    assert schema["anyOf"][0]["properties"]["detail"]["$ref"] == "#/$defs/detail"


def test_native_openai_does_not_relocate_reference_like_literal_data(monkeypatch, root_union):
    literal = {"$ref": "#/$defs/detail", "$id": "literal identifier"}
    root_union["anyOf"][0]["properties"]["literal"] = {"const": literal}
    root_union["anyOf"][0]["required"].append("literal")
    payload = {"action": "final", "complete": True, "literal": literal}
    bodies = []
    with _sdk(monkeypatch, [{"response": payload}], bodies):
        assert (
            RoutedLLMClient(_config()).complete_json(
                [], operation="review.verify", response_schema=root_union
            )
            == payload
        )
    wire = bodies[0]["response_format"]["json_schema"]["schema"]
    assert wire["properties"]["response"]["anyOf"][0]["properties"]["literal"]["const"] == literal


def test_native_openai_does_not_relocate_refs_across_resource_identifiers(monkeypatch, root_union):
    root_union["$id"] = "https://schema.invalid/review"
    bodies = []
    with _sdk(monkeypatch, [], bodies):
        with pytest.raises(ValueError, match="resource identifiers are not supported"):
            RoutedLLMClient(_config()).complete_json(
                [], operation="review.verify", response_schema=root_union
            )
    assert bodies == []


def test_native_openai_root_envelope_keeps_one_retry_budget(monkeypatch, root_union):
    payload = {"action": "final", "complete": True}
    bodies = []
    client = RoutedLLMClient(_config(retries=1))
    monkeypatch.setattr(client.limits, "wait_for_retry", lambda delay: None)
    starts = []
    client.set_event_sink(
        lambda event, **data: starts.append(data) if event == "llm_request_started" else None
    )
    with _sdk(monkeypatch, [503, {"response": payload}], bodies):
        assert (
            client.complete_json([], operation="review.verify", response_schema=root_union)
            == payload
        )
    assert [event["attempt"] for event in starts] == [1, 2]
    assert len(bodies) == 2
    assert bodies[0]["response_format"] == bodies[1]["response_format"]
    assert client.usage_summary()["totals"]["calls"] == 1


def test_native_openai_root_envelope_does_not_bypass_request_limits(monkeypatch, root_union):
    bodies = []
    client = RoutedLLMClient(_config(retries=2, budget={"max_requests": 1}))
    monkeypatch.setattr(client.limits, "wait_for_retry", lambda delay: None)
    with _sdk(monkeypatch, [503], bodies):
        with pytest.raises(RequestStopped):
            client.complete_json([], operation="review.verify", response_schema=root_union)
    assert len(bodies) == 1


@pytest.mark.parametrize("payload", [{}, {"action": "final"}, {"response": {}, "extra": True}])
def test_native_openai_rejects_invalid_wire_envelopes_without_weaker_retry(
    monkeypatch, root_union, payload
):
    bodies = []
    with _sdk(monkeypatch, [payload], bodies):
        with pytest.raises(ValueError, match="structured response envelope"):
            RoutedLLMClient(_config(retries=2)).complete_json(
                [], operation="review.verify", response_schema=root_union
            )
    assert len(bodies) == 1


def test_native_openai_envelope_errors_keep_credential_snapshot_projection(monkeypatch, root_union):
    bodies = []
    client = RoutedLLMClient(_config(), credentials={"default": "test-key"})
    events = []
    client.set_event_sink(lambda event, **data: events.append({"event": event, **data}))
    with _sdk(monkeypatch, [{"untrusted": "PRIVATE model response"}], bodies):
        with pytest.raises(ProviderRequestError) as error:
            client.complete_json([], operation="review.verify", response_schema=root_union)
    assert "PRIVATE" not in str(error.value)
    assert "PRIVATE" not in json.dumps(events)
    assert len(bodies) == 1


def test_openai_adapter_protocol_invalidates_previous_request_identity(monkeypatch):
    from wenyi_core.llm.routing import inference_snapshot

    config = _config()
    assert OpenAIClient.protocol_version == 3
    current = inference_snapshot(config, ["review.verify"])
    with monkeypatch.context() as context:
        context.setattr(OpenAIClient, "protocol_version", 2)
        assert inference_snapshot(config, ["review.verify"]) != current
