"""Fixed human translations reach verification and cannot be replaced by synonyms."""

import json

import pytest
from wenyi_core.agents.review_loop import ReviewAgentLoop
from wenyi_core.glossary.fixed import relevant_fixed_terms
from wenyi_core.llm.providers.fake import FakeClient
from wenyi_core.pipeline.orchestrator import Orchestrator

from tests.review_fixtures import MemoryTrace, _config, review_evidence


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Heart of the Tempest left.", {"Heart of the Tempest": "飓风之心号"}),
        (
            "Heart of the Tempest and Tempest left.",
            {"Heart of the Tempest": "飓风之心号", "Tempest": "飓风号"},
        ),
        ("A belt and a belter.", {}),
        (
            "Belt, Belter and Belters.",
            {"Belt": "小行星带", "Belter": "小行星带人", "Belters": "小行星带人"},
        ),
    ],
)
def test_fixed_context_respects_spelling_boundaries_and_longer_names(text, expected):
    mappings = {
        "Heart of the Tempest": "飓风之心号",
        "Tempest": "飓风号",
        "Belt": "小行星带",
        "Belter": "小行星带人",
        "Belters": "小行星带人",
    }
    assert relevant_fixed_terms(mappings, text) == expected


def test_verification_receives_only_relevant_fixed_mappings():
    config = _config()
    config.pipeline.glossary_fixed_terms = {
        "The Tempest": "飓风号",
        "Storm": "风暴号",
        "Belt": "小行星带",
    }
    client = FakeClient(
        handler=lambda *_: json.dumps(
            {
                "action": "final",
                "complete": True,
                "decisions": [],
                "new_issues": [],
            }
        )
    )
    ReviewAgentLoop(client, config, review_evidence(), MemoryTrace()).review_chunk(
        chapter=0,
        chunk_base=0,
        sources=["The Tempest left."],
        targets=["风暴号离开了。"],
        initial_issues=[],
    )
    user = client.calls[0]["messages"][1]["content"]
    assert "[Required human glossary mappings]" in user
    assert '"The Tempest": "飓风号"' in user
    assert '"Storm": "风暴号"' not in user
    assert '"Belt": "小行星带"' not in user


@pytest.mark.parametrize("operation", ["review.verify", "autofix.verify"])
@pytest.mark.parametrize("origin", ["initial", "agent"])
@pytest.mark.parametrize("proposed", ["暴风号", "飓风号"])
@pytest.mark.parametrize("subject", ["The Tempest", "the Tempest"])
def test_verification_checks_fixed_term_proposals(operation, origin, proposed, subject):
    config = _config()
    config.pipeline.glossary_fixed_terms = {"The Tempest": "飓风号"}
    candidate = {
        "index": 0,
        "type": "terminology",
        "detail": "Ship name differs.",
        "suggestion": "Use the fixed ship name.",
    }
    fields = {
        "detail": "The two ships need distinct names.",
        "suggestion": f"改为{proposed}。",
        "consistency": {
            "kind": "term",
            "subject_source": subject,
            "proposed_value": proposed,
        },
        "evidence_refs": [],
    }
    response = {"action": "final", "complete": True, "decisions": [], "new_issues": []}
    if origin == "initial":
        response["decisions"] = [
            {"candidate_id": "ch0-base0-candidate0", "verdict": "confirmed", **fields}
        ]
    else:
        response["new_issues"] = [{"index": 0, "type": "terminology", **fields}]
    outcome = ReviewAgentLoop(
        FakeClient(handler=lambda *_: json.dumps(response)),
        config,
        review_evidence(),
        MemoryTrace(),
        operation=operation,
    ).review_chunk(
        chapter=0,
        chunk_base=0,
        sources=["The Tempest left."],
        targets=["风暴号离开了。"],
        initial_issues=[candidate] if origin == "initial" else [],
    )
    if proposed == "暴风号":
        assert outcome.fallback_reason.startswith("fixed_term_conflict")
        assert all(issue.get("agent_fallback") for issue in outcome.issues)
    else:
        assert not outcome.fallback_reason
        assert outcome.issues[0]["consistency"]["proposed_value"] == "飓风号"


def test_changed_fixed_context_invalidates_finished_verification():
    config = _config()
    trace = MemoryTrace()
    client = FakeClient(
        handler=lambda *_: json.dumps(
            {
                "action": "final",
                "complete": True,
                "decisions": [],
                "new_issues": [],
            }
        )
    )
    for mappings in [{}, {"The Tempest": "飓风号"}, {"The Tempest": "飓风号"}]:
        config.pipeline.glossary_fixed_terms = mappings
        outcome = ReviewAgentLoop(client, config, review_evidence(), trace).review_chunk(
            chapter=0,
            chunk_base=0,
            sources=["The Tempest left."],
            targets=["风暴号离开了。"],
            initial_issues=[],
        )
        assert not outcome.fallback_reason
    assert len(client.calls) == 2
    assert any(e["event"] == "review_agent_cache_invalidated" for e in trace.events)


def test_fixed_value_in_consistency_does_not_hide_a_conflicting_suggestion():
    config = _config()
    config.pipeline.glossary_fixed_terms = {"The Tempest": "飓风号"}
    response = {
        "action": "final",
        "complete": True,
        "decisions": [],
        "new_issues": [
            {
                "index": 0,
                "type": "terminology",
                "detail": "Different ships.",
                "suggestion": "改为暴风号。",
                "consistency": {
                    "kind": "term",
                    "subject_source": "The Tempest",
                    "proposed_value": "飓风号",
                },
                "evidence_refs": [],
            }
        ],
    }
    outcome = ReviewAgentLoop(
        FakeClient(handler=lambda *_: json.dumps(response)),
        config,
        review_evidence(),
        MemoryTrace(),
    ).review_chunk(
        chapter=0,
        chunk_base=0,
        sources=["The Tempest left."],
        targets=["风暴号离开了。"],
        initial_issues=[],
    )
    assert outcome.fallback_reason.startswith("fixed_term_conflict")


def test_fixed_mapping_changes_are_bound_to_review_session_identity():
    config = _config()
    service = Orchestrator(config, FakeClient())._review
    before = service._review_config_snapshot()
    config.pipeline.glossary_fixed_terms["The Tempest"] = "飓风号"
    after = service._review_config_snapshot()
    assert before["fixed_terms"] == {}
    assert after["fixed_terms"] == {"The Tempest": "飓风号"}
    assert before != after
