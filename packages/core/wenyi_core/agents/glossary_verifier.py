"""Review bounded new mapping candidates against source evidence before storage."""

from __future__ import annotations

import json
from dataclasses import replace

from ..errors import WenyiError
from ..llm.retrying import TruncatedResponseError
from ..models.glossary import GlossaryTerm
from .base import Agent


class GlossaryVerificationError(WenyiError):
    code = "glossary_verification_failed"


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value = {}
    for key, item in pairs:
        if key in value:
            raise GlossaryVerificationError("Duplicate glossary verification JSON key")
        value[key] = item
    return value


class GlossaryVerifier(Agent):
    def verify(self, terms: list[GlossaryTerm], evidence: list[dict]) -> list[GlossaryTerm]:
        """Require one explicit decision per caller-owned candidate identity."""
        ids = [item["id"] for item in evidence]
        schema = {
            "type": "object",
            "properties": {
                "decisions": {
                    "type": "array",
                    "minItems": len(ids),
                    "maxItems": len(ids),
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "enum": ids},
                            "accept": {"type": "boolean"},
                            "target": {"type": "string"},
                        },
                        "required": ["id", "accept", "target"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["decisions"],
            "additionalProperties": False,
        }
        messages = [
            {"role": "system", "content": self.render("glossary_verifier_system")},
            {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)},
        ]
        for attempt in range(self.config.pipeline.align_retry_limit + 1):
            try:
                raw = self.client.complete(
                    messages, operation="glossary.verify", json_mode=True, response_schema=schema
                )
                data = json.loads(raw, object_pairs_hook=_unique_object)
            except (json.JSONDecodeError, GlossaryVerificationError, TruncatedResponseError):
                data = None
            rows = (
                data.get("decisions")
                if isinstance(data, dict) and set(data) == {"decisions"}
                else None
            )
            if isinstance(rows, list) and len(rows) == len(ids):
                decisions = {}
                for row in rows:
                    if (
                        not isinstance(row, dict)
                        or set(row) != {"id", "accept", "target"}
                        or row.get("id") not in ids
                        or row["id"] in decisions
                        or type(row.get("accept")) is not bool
                        or not isinstance(row.get("target"), str)
                        or (row["accept"] and not row["target"].strip())
                    ):
                        break
                    decisions[row["id"]] = row
                else:
                    return [
                        replace(term, target=decisions[key]["target"].strip())
                        for key, term in zip(ids, terms)
                        if decisions[key]["accept"]
                    ]
            if attempt < self.config.pipeline.align_retry_limit:
                messages.append(
                    {
                        "role": "user",
                        "content": "Return exactly one decision for each supplied id. All targets must be strings.",
                    }
                )
        raise GlossaryVerificationError(
            "Invalid glossary verification output; no candidates admitted"
        )
