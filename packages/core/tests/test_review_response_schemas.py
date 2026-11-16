"""Review requests carry task-specific server schemas without weakening local validation."""

import json
import re
from dataclasses import replace

import pytest
from wenyi_core.agents.review_arbiter import ReviewConflictArbiter
from wenyi_core.agents.review_fixer import ReviewFixer
from wenyi_core.agents.review_loop import ReviewAgentLoop
from wenyi_core.agents.reviewer import Reviewer
from wenyi_core.llm.providers.fake import FakeClient
from wenyi_core.llm.routing import inference_snapshot
from wenyi_core.review.conflicts import build_conflict_groups, normalize_review_issues

from tests.review_fixtures import MemoryTrace, _config, fix_json, review_evidence


def test_initial_review_requests_a_complete_block_schema():
    client = FakeClient(
        handler=lambda *args: json.dumps({"reviewed_segments": 2, "complete": True, "issues": []})
    )
    Reviewer(client, _config()).review(["A.", "B."], ["甲。", "乙。"])
    schema = client.calls[0]["response_schema"]
    assert schema is not None
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"reviewed_segments", "complete", "issues"}
    assert schema["properties"]["reviewed_segments"]["const"] == 2
    issue = schema["properties"]["issues"]["items"]
    assert issue["properties"]["index"] == {"type": "integer", "minimum": 0, "maximum": 1}
    assert issue["additionalProperties"] is False


@pytest.mark.parametrize("operation", ["review.fix", "autofix.fix"])
def test_fix_requests_bind_schema_to_the_exact_target_and_issues(operation):
    client = FakeClient(handler=lambda messages, *args: fix_json(messages[-1]["content"], "改译。"))
    patch = ReviewFixer(client, _config(), operation=operation).propose(
        1,
        "ch0:text0:seg0",
        0,
        0,
        "Ann arrived.",
        "安到了。",
        [{"issue_id": "issue-1", "detail": "译名", "suggestion": "修改"}],
    )
    call = client.calls[0]
    assert call["operation"] == operation
    schema = call["response_schema"]
    assert schema is not None
    assert schema["properties"]["segment_ref"]["const"] == patch.segment_ref
    assert schema["properties"]["before_hash"]["const"] == patch.before_hash
    assert schema["properties"]["issue_ids"]["items"]["enum"] == ["issue-1"]
    assert schema["properties"]["complete"]["const"] is True
    assert set(schema["required"]) == set(schema["properties"])


@pytest.mark.parametrize("operation", ["review.verify", "autofix.verify"])
def test_verification_schema_tracks_evidence_and_final_only_turns(operation):
    def respond(messages, *args):
        if len(messages) == 2:
            return json.dumps(
                {
                    "action": "request_evidence",
                    "complete": False,
                    "requests": [
                        {
                            "request_id": "style",
                            "tool": "book_context",
                            "arguments": {"section": "style_guide"},
                        }
                    ],
                }
            )
        return json.dumps(
            {
                "action": "final",
                "complete": True,
                "decisions": [
                    {
                        "candidate_id": "ch0-base0-candidate0",
                        "verdict": "dismissed",
                        "reason": "无需修改",
                        "evidence_refs": ["book:style_guide"],
                    }
                ],
                "new_issues": [],
            }
        )

    config = _config()
    config.pipeline.review_agent_max_evidence_rounds = 1
    client = FakeClient(handler=respond)
    outcome = ReviewAgentLoop(
        client, config, review_evidence(), MemoryTrace(), operation=operation
    ).review_chunk(
        chapter=0,
        chunk_base=0,
        sources=["Ann arrived."],
        targets=["安到了。"],
        initial_issues=[
            {"index": 0, "type": "terminology", "detail": "译名", "suggestion": "修改"}
        ],
    )
    assert not outcome.fallback_reason
    assert len(client.calls) == 2
    first, second = [call["response_schema"] for call in client.calls]
    assert first is not None and second is not None
    assert {branch["properties"]["action"]["const"] for branch in first["anyOf"]} == {
        "final",
        "request_evidence",
    }
    assert second["properties"]["action"]["const"] == "final"
    decision = next(
        branch
        for branch in second["properties"]["decisions"]["items"]["anyOf"]
        if branch["properties"]["verdict"]["const"] == "dismissed"
    )
    assert decision["properties"]["candidate_id"]["enum"] == ["ch0-base0-candidate0"]
    assert decision["properties"]["evidence_refs"]["items"]["enum"] == [
        "book:style_guide",
        "ch0:text0:seg0",
    ]
    assert second["properties"]["new_issues"]["items"]["properties"]["index"]["maximum"] == 0


@pytest.mark.parametrize("operation", ["review.verify", "autofix.verify"])
@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"detail": "", "suggestion": "修改"},
        {"detail": " \n", "suggestion": "修改"},
        {"detail": "原文姓名与译名不一致", "suggestion": "\t"},
    ],
)
def test_confirmed_decisions_require_their_own_substantive_guidance(operation, fields):
    response = {
        "action": "final",
        "complete": True,
        "decisions": [{"candidate_id": "ch0-base0-candidate0", "verdict": "confirmed", **fields}],
        "new_issues": [],
    }
    outcome = ReviewAgentLoop(
        FakeClient(handler=lambda *args: json.dumps(response)),
        _config(),
        review_evidence(),
        MemoryTrace(),
        operation=operation,
    ).review_chunk(
        chapter=0,
        chunk_base=0,
        sources=["Ann arrived."],
        targets=["安到了。"],
        initial_issues=[
            {
                "index": 0,
                "type": "terminology",
                "detail": "Initial allegation",
                "suggestion": "Initial proposed correction",
            }
        ],
    )
    assert outcome.fallback_reason == "confirmed_issue_missing_text"
    assert outcome.issues[0]["agent_fallback"] is True


@pytest.mark.parametrize("operation", ["review.verify", "autofix.verify"])
@pytest.mark.parametrize("reason", ["", " \n\t"])
def test_dismissed_decisions_require_their_own_reason(operation, reason):
    response = {
        "action": "final",
        "complete": True,
        "decisions": [
            {"candidate_id": "ch0-base0-candidate0", "verdict": "dismissed", "reason": reason}
        ],
        "new_issues": [],
    }
    outcome = ReviewAgentLoop(
        FakeClient(handler=lambda *args: json.dumps(response)),
        _config(),
        review_evidence(),
        MemoryTrace(),
        operation=operation,
    ).review_chunk(
        chapter=0,
        chunk_base=0,
        sources=["Ann arrived."],
        targets=["安到了。"],
        initial_issues=[
            {
                "index": 0,
                "type": "terminology",
                "detail": "Initial allegation",
                "suggestion": "Initial proposed correction",
            }
        ],
    )
    assert outcome.fallback_reason == "dismissal_without_reason"


def test_verification_schema_requires_guidance_by_issue_verdict():
    from wenyi_core.agents.review_schemas import verification_schema

    schema = verification_schema(["candidate-1"], 1, {"ref-1"})
    decisions = schema["properties"]["decisions"]["items"]["anyOf"]
    branches = {branch["properties"]["verdict"]["const"]: branch for branch in decisions}
    confirmed = branches["confirmed"]["properties"]
    dismissed = branches["dismissed"]["properties"]
    assert confirmed["detail"]["pattern"] == confirmed["suggestion"]["pattern"]
    assert dismissed["reason"]["pattern"] == confirmed["detail"]["pattern"]
    assert confirmed["candidate_id"]["enum"] == dismissed["candidate_id"]["enum"] == ["candidate-1"]
    assert confirmed["evidence_refs"] == dismissed["evidence_refs"]


@pytest.mark.parametrize(
    "value,expected",
    [
        ("原译准确地表达了原文的含义，不需要修改。", True),
        ("原文使用复数。\n现译漏掉数量信息，需补回多个对象。", True),
        (" \n保留当前译文。\n\t", True),
        ("", False),
        (" \t\n", False),
    ],
)
def test_nonblank_schema_patterns_allow_complete_single_and_multiline_guidance(value, expected):
    from wenyi_core.agents.review_schemas import verification_schema

    schema = verification_schema(["candidate-1"], 1, {"ref-1"})
    branches = {
        branch["properties"]["verdict"]["const"]: branch["properties"]
        for branch in schema["properties"]["decisions"]["items"]["anyOf"]
    }
    for field in [
        branches["confirmed"]["detail"],
        branches["confirmed"]["suggestion"],
        branches["dismissed"]["reason"],
    ]:
        pattern = field["pattern"]
        assert bool(re.search(pattern, value)) == expected
        # Guided decoders may match the entire string rather than search for a substring.
        assert bool(re.fullmatch(pattern, value)) == expected


def test_arbitration_schema_binds_proposals_and_conflict_without_changing_prompt():
    evidence = review_evidence()
    issues = normalize_review_issues(
        [
            {
                "chapter": 0,
                "index": index,
                "_chunk_id": f"chunk-{index}",
                "type": "terminology",
                "detail": "译名不同",
                "suggestion": value,
                "consistency": {"kind": "term", "subject_source": "Ann", "proposed_value": value},
            }
            for index, value in enumerate(["安", "安妮"])
        ],
        evidence,
    )
    conflict = build_conflict_groups(issues)[0]
    client = FakeClient(
        handler=lambda *args: json.dumps(
            {
                "action": "final",
                "complete": True,
                "conflict_id": conflict["conflict_id"],
                "status": "suggested",
                "recommended_value": "安",
                "reason": "原文支持",
                "evidence_refs": [],
            }
        )
    )
    config = _config()
    config.pipeline.review_agent_max_evidence_rounds = 0
    result = ReviewConflictArbiter(client, config, evidence, MemoryTrace()).arbitrate(conflict)
    assert result["status"] == "suggested"
    schema = client.calls[0]["response_schema"]
    assert schema is not None
    assert schema["properties"]["conflict_id"]["const"] == conflict["conflict_id"]
    assert schema["properties"]["recommended_value"]["enum"] == ["", "安", "安妮"]
    assert "supported_issue_ids" not in schema["properties"]
    assert "rejected_issue_ids" not in schema["properties"]


@pytest.mark.parametrize(
    "operation",
    [
        "review.scan",
        "review.verify",
        "review.arbitrate",
        "review.fix",
        "autofix.verify",
        "autofix.fix",
    ],
)
def test_review_schema_protocol_invalidates_pre_schema_inference(operation, monkeypatch):
    from wenyi_core.llm import operations, routing

    config = _config()
    current = inference_snapshot(config.llm, [operation])
    registry = dict(operations.OPERATIONS)
    previous_version = 2 if operation.endswith(".fix") else 1
    assert registry[operation].protocol_version > previous_version
    registry[operation] = replace(registry[operation], protocol_version=previous_version)
    with monkeypatch.context() as context:
        context.setattr(routing, "OPERATIONS", registry)
        context.setattr(operations, "OPERATIONS", registry)
        assert inference_snapshot(config.llm, [operation]) != current


@pytest.mark.parametrize("operation", ["review.verify", "autofix.verify"])
def test_pre_gate_confirmations_are_not_reused_after_protocol_upgrade(operation, monkeypatch):
    from wenyi_core.llm import operations, routing

    config = _config()
    previous = dict(operations.OPERATIONS)
    assert previous[operation].protocol_version == 4
    previous[operation] = replace(previous[operation], protocol_version=3)
    with monkeypatch.context() as context:
        context.setattr(routing, "OPERATIONS", previous)
        context.setattr(operations, "OPERATIONS", previous)
        old_inference = inference_snapshot(config.llm, [operation])
    trace = MemoryTrace()
    trace.data["chunk-ch0-base0-n1"] = {
        "agent_id": "chunk-ch0-base0-n1",
        "stage": operation,
        "inference": old_inference,
        "status": "finished",
        "turns": [],
        "result": {"issues": [{"index": 0, "detail": "Inherited allegation"}], "dismissed": []},
    }
    response = {
        "action": "final",
        "complete": True,
        "decisions": [
            {
                "candidate_id": "ch0-base0-candidate0",
                "verdict": "dismissed",
                "reason": "原译正确，无需修改。",
            }
        ],
        "new_issues": [],
    }
    client = FakeClient(handler=lambda *args: json.dumps(response))
    outcome = ReviewAgentLoop(
        client, config, review_evidence(), trace, operation=operation
    ).review_chunk(
        chapter=0,
        chunk_base=0,
        sources=["Ann arrived."],
        targets=["安到了。"],
        initial_issues=[
            {
                "index": 0,
                "type": "terminology",
                "detail": "Initial allegation",
                "suggestion": "No change",
            }
        ],
    )
    assert len(client.calls) == 1
    assert not outcome.issues and not outcome.fallback_reason
    assert len(outcome.dismissed) == 1
    assert any(event["event"] == "review_agent_cache_invalidated" for event in trace.events)


def test_schema_nested_objects_and_tool_arguments_are_closed():
    from wenyi_core.agents.review_schemas import action_schema, verification_schema

    schema = action_schema(verification_schema(["candidate-1"], 2, {"ref-1"}), allow_evidence=True)

    def check(value):
        if isinstance(value, dict):
            if value.get("type") == "object" and "anyOf" not in value:
                assert value["additionalProperties"] is False
                assert set(value["required"]) == set(value["properties"])
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)

    check(schema)
    requests = schema["anyOf"][0]["properties"]["requests"]
    assert (requests["minItems"], requests["maxItems"]) == (1, 4)
    tools = {item["properties"]["tool"]["const"]: item for item in requests["items"]["anyOf"]}
    assert set(tools) == {"glossary_term", "term_occurrences", "segment_context", "book_context"}
    context = tools["segment_context"]["properties"]["arguments"]["properties"]
    assert context["before"]["maximum"] == context["after"]["maximum"] == 6
