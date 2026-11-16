"""Polishing agent using the strong tier.
Improve literary quality in the target language without changing information or paragraph
count. Prefer appending a polish user turn to the translation conversation so shared
prompt prefixes stay cacheable. Invalid fresh polishing must stop publication.
"""

from __future__ import annotations

from typing import Any

from ..errors import WenyiError
from ..llm.json_parser import JsonParseError
from ..llm.retrying import TruncatedResponseError
from ..models.glossary import GlossaryTerm
from . import prompts
from .base import Agent, Messages


class PolishOutputError(WenyiError):
    """Polishing exhausted its structural retries; no batch may be published."""

    code = "polish_invalid_output"


def _polish_schema(n: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "polished": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": n,
                "maxItems": n,
            }
        },
        "required": ["polished"],
        "additionalProperties": False,
    }


def _polished_items(data: Any, n: int) -> list[str] | None:
    items = data.get("polished") if isinstance(data, dict) else data
    if (
        isinstance(items, list)
        and len(items) == n
        and all(isinstance(item, str) and item.strip() for item in items)
    ):
        return items
    return None


class Polisher(Agent):
    def polish(
        self,
        targets: list[str],
        *,
        glossary_terms: list[GlossaryTerm] | None = None,
        style: str = "",
        next_source: str = "",
        sources: list[str] | None = None,
    ) -> list[str]:
        """Polish against source references and stop on invalid output after bounded retries."""
        if not targets:
            return []
        n = len(targets)
        system = self.render(
            "polisher_system", src=self.src, tgt=self.tgt, n=n
        ) + prompts.protected_object_guidance(targets)
        user = self.render(
            "polisher_user",
            src=self.src,
            tgt=self.tgt,
            glossary=prompts.render_glossary(
                glossary_terms or [],
                source_lang=self.src,
                genre=self.language_policy.context.genre,
                note_source=(
                    "\n".join(sources or []) if self.config.pipeline.terminology_context else None
                ),
            ),
            source_context=(
                "[Current source paragraphs: reference only, not additional output items]\n"
                + prompts.numbered(sources)
                if sources
                else ""
            ),
            style=style or "(none)",
            n=n,
            numbered_target=prompts.numbered(targets),
            next_source=prompts.render_source_reference(next_source),
        )
        messages: Messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        for attempt in range(self.config.pipeline.align_retry_limit + 1):
            try:
                data, _ = self._complete_json_turn(
                    messages,
                    operation="polish.body",
                    response_schema=_polish_schema(n),
                )
            except (JsonParseError, TruncatedResponseError):
                data = None
            items = _polished_items(data, n)
            if items is not None:
                return items
            if attempt < self.config.pipeline.align_retry_limit:
                messages.append(
                    {
                        "role": "user",
                        "content": f'Return only {{"polished":[...]}} with exactly {n} nonempty strings. '
                        "Preserve the original paragraph order; do not return indices or objects.",
                    }
                )
        raise PolishOutputError(
            f"Polishing did not return {n} aligned nonempty paragraphs; batch not published"
        )

    def polish_continue(
        self,
        turn: Messages,
        *,
        n: int,
        next_source: str = "",
    ) -> list[str] | None:
        """Append a polish user turn to a successful translation transcript.

        Returns the polished list on success, or ``None`` when the continuation fails so the
        caller can fall back to a standalone polish call or keep the raw translations.
        """
        if n <= 0 or len(turn) < 3:
            return None
        continue_user = self.render(
            "polisher_continue_user",
            src=self.src,
            tgt=self.tgt,
            n=n,
            next_source=prompts.render_source_reference(next_source),
        ) + prompts.protected_object_guidance([message["content"] for message in turn])
        messages: Messages = [
            *turn,
            {"role": "user", "content": continue_user},
        ]
        try:
            data, _ = self._complete_json_turn(
                messages, operation="polish.body", response_schema=_polish_schema(n)
            )
        except (JsonParseError, TruncatedResponseError):
            return None
        return _polished_items(data, n)
