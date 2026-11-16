"""Task-owned JSON Schemas for Review's existing output and evidence-action protocols."""

from __future__ import annotations

from typing import Any

from ..review.models import CONSISTENCY_KINDS

_ISSUE_TYPES = ["missing", "added", "mistranslation", "terminology", "pronoun"]


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _refs(allowed_refs: set[str]) -> dict[str, Any]:
    if not allowed_refs:
        return {"type": "array", "items": {"type": "string"}, "maxItems": 0}
    return {"type": "array", "items": {"type": "string", "enum": sorted(allowed_refs)}}


def _issue(segment_count: int) -> dict[str, Any]:
    return _object(
        {
            "index": {"type": "integer", "minimum": 0, "maximum": segment_count - 1},
            "type": {"type": "string", "enum": _ISSUE_TYPES},
            "detail": {"type": "string", "minLength": 1},
            "suggestion": {"type": "string", "minLength": 1},
        }
    )


def scanner_schema(segment_count: int) -> dict[str, Any]:
    """Require a complete screening receipt with block-local issue positions."""
    return _object(
        {
            "issues": {"type": "array", "items": _issue(segment_count)},
            "reviewed_segments": {"type": "integer", "const": segment_count},
            "complete": {"type": "boolean", "const": True},
        }
    )


def fixer_schema(
    segment_ref: str, before_hash: str, issue_ids: tuple[str, ...] | list[str]
) -> dict[str, Any]:
    """Bind a complete replacement to its exact baseline and issue identities."""
    return _object(
        {
            "segment_ref": {"type": "string", "const": segment_ref},
            "before_hash": {"type": "string", "const": before_hash},
            "issue_ids": {
                "type": "array",
                "items": {"type": "string", "enum": list(issue_ids)},
                "minItems": len(issue_ids),
                "maxItems": len(issue_ids),
                "uniqueItems": True,
            },
            "replacement": {"type": "string", "minLength": 1},
            "complete": {"type": "boolean", "const": True},
        }
    )


def verification_schema(
    candidate_ids: list[str], segment_count: int, allowed_refs: set[str]
) -> dict[str, Any]:
    """Describe final issue decisions; semantic identity checks remain local."""
    consistency = {
        "anyOf": [
            _object({}),
            _object(
                {
                    "subject_source": {"type": "string", "minLength": 1},
                    "kind": {"type": "string", "enum": sorted(CONSISTENCY_KINDS)},
                    "proposed_value": {"type": "string", "minLength": 1},
                }
            ),
        ]
    }
    candidate_id: dict[str, Any] = {"type": "string"}
    if candidate_ids:
        candidate_id["enum"] = candidate_ids
    decision_fields = {
        "candidate_id": candidate_id,
        "detail": {"type": "string"},
        "suggestion": {"type": "string"},
        "reason": {"type": "string"},
        "consistency": consistency,
        "evidence_refs": _refs(allowed_refs),
    }
    # Permit a full response even when a guided decoder applies full-match semantics.
    nonblank = {"type": "string", "minLength": 1, "pattern": r"[\s\S]*\S[\s\S]*"}
    decision = {
        "anyOf": [
            _object(
                {
                    **decision_fields,
                    "verdict": {
                        "type": "string",
                        "const": "confirmed",
                        "description": "The alleged translation error exists and needs correction.",
                    },
                    "detail": {
                        **nonblank,
                        "description": "Source evidence and the specific error in the current translation.",
                    },
                    "suggestion": {
                        **nonblank,
                        "description": "The minimum concrete correction supported by that evidence.",
                    },
                }
            ),
            _object(
                {
                    **decision_fields,
                    "verdict": {
                        "type": "string",
                        "const": "dismissed",
                        "description": "The alleged error is absent, uncertain, or needs no change.",
                    },
                    "reason": {
                        **nonblank,
                        "description": "Why the original translation should be retained for this allegation.",
                    },
                }
            ),
        ]
    }
    issue = _issue(segment_count)
    issue["properties"].update(consistency=consistency, evidence_refs=_refs(allowed_refs))
    issue["required"] = list(issue["properties"])
    return _object(
        {
            "action": {"type": "string", "const": "final"},
            "decisions": {
                "type": "array",
                "items": decision,
                "minItems": len(candidate_ids),
                "maxItems": len(candidate_ids),
            },
            "new_issues": {
                "type": "array",
                "items": issue,
                "maxItems": min(50, max(4, segment_count * 2)),
            },
            "complete": {"type": "boolean", "const": True},
        }
    )


def arbitration_schema(
    conflict_id: str, proposed_values: list[str], allowed_refs: set[str]
) -> dict[str, Any]:
    """Choose only an existing proposal; supporting issue IDs are derived locally."""
    return _object(
        {
            "action": {"type": "string", "const": "final"},
            "conflict_id": {"type": "string", "const": conflict_id},
            "status": {"type": "string", "enum": ["suggested", "unresolved"]},
            "recommended_value": {
                "type": "string",
                "enum": list(dict.fromkeys(["", *proposed_values])),
            },
            "reason": {"type": "string", "minLength": 1},
            "evidence_refs": _refs(allowed_refs),
            "complete": {"type": "boolean", "const": True},
        }
    )


def action_schema(final_schema: dict[str, Any], *, allow_evidence: bool) -> dict[str, Any]:
    """Retain distinct evidence/final payloads and require final when evidence is exhausted."""
    if not allow_evidence:
        return final_schema
    requests = []
    for tool, arguments in [
        ("glossary_term", _object({"term": {"type": "string", "minLength": 1, "maxLength": 128}})),
        (
            "term_occurrences",
            _object(
                {
                    "term": {"type": "string", "minLength": 1, "maxLength": 128},
                    "selectors": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 8,
                        "items": {
                            "anyOf": [
                                {"type": "string", "enum": ["first", "middle", "last"]},
                                {"type": "integer", "minimum": 1},
                            ]
                        },
                    },
                    "context_radius": {"type": "integer", "minimum": 0, "maximum": 2},
                }
            ),
        ),
        (
            "segment_context",
            _object(
                {
                    "chapter": {"type": "integer", "minimum": 0},
                    "index": {"type": "integer", "minimum": 0},
                    "before": {"type": "integer", "minimum": 0, "maximum": 6},
                    "after": {"type": "integer", "minimum": 0, "maximum": 6},
                }
            ),
        ),
        (
            "book_context",
            _object(
                {
                    "section": {"type": "string", "enum": ["style_guide", "book_synopsis"]},
                }
            ),
        ),
        (
            "book_context",
            _object(
                {
                    "section": {"type": "string", "const": "chapter_digest"},
                    "chapter": {"type": "integer", "minimum": 0},
                }
            ),
        ),
    ]:
        requests.append(
            _object(
                {
                    "request_id": {"type": "string", "minLength": 1},
                    "tool": {"type": "string", "const": tool},
                    "arguments": arguments,
                }
            )
        )
    evidence = _object(
        {
            "action": {"type": "string", "const": "request_evidence"},
            "requests": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": {"anyOf": requests},
            },
            "complete": {"type": "boolean", "const": False},
        }
    )
    return {"type": "object", "anyOf": [evidence, final_schema]}
