"""Select literal human mappings without confusing nested names or word fragments."""

from __future__ import annotations

import re
from collections.abc import Mapping


def relevant_fixed_terms(fixed: Mapping[str, str], text: str) -> dict[str, str]:
    """Match explicit spellings, taking longer phrases before overlapping short names.

    Case is significant: a fixed astronomical ``Belt`` must not capture a mechanical
    ``belt``. Users can supply additional spellings explicitly.
    """
    matches = []
    for source in fixed:
        left = r"(?<!\w)" if source.isascii() and source[0].isalnum() else ""
        right = r"(?!\w)" if source.isascii() and source[-1].isalnum() else ""
        for match in re.finditer(left + re.escape(source) + right, text):
            matches.append((match.start(), match.end(), source))
    accepted: list[tuple[int, int]] = []
    selected: set[str] = set()
    for start, end, source in sorted(matches, key=lambda item: (item[0] - item[1], item[0])):
        if any(
            start < previous_end and previous_start < end
            for previous_start, previous_end in accepted
        ):
            continue
        accepted.append((start, end))
        selected.add(source)
    return {source: target for source, target in fixed.items() if source in selected}
