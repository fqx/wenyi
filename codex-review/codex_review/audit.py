"""Seeded, stratified A/B payloads with a separately stored private mapping."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

from .freeze import read_json, write_json
from .storage import canonical_digest as digest


def create_audit(
    run: Path,
    results: list[dict[str, Any]],
    *,
    count: int | None = None,
    ratio: float | None = None,
    seed: int = 0,
) -> Path:
    frozen = read_json(run / "frozen.json")
    indexed = {row["id"]: row for row in frozen["segments"]}
    scope = frozen["scope_ids"]
    by_id = {row["id"]: row for row in results}
    if len(by_id) != len(results) or set(by_id) != set(scope):
        raise ValueError("Audit results must cover the frozen review scope exactly once")
    if count is not None and ratio is not None:
        raise ValueError("Choose count or ratio, not both")
    if count is None:
        fraction = 0.1 if ratio is None else ratio
        if not 0 < fraction <= 1:
            raise ValueError("Audit ratio must be in (0, 1]")
        count = max(1, math.ceil(len(scope) * fraction))
    if isinstance(count, bool) or count < 1:
        raise ValueError("Audit count must be positive")
    count = min(count, len(scope))
    rng = random.Random(seed)
    strata: dict[tuple[int, bool], list[str]] = {}
    for sid in scope:
        row, result = indexed[sid], by_id[sid]
        if not isinstance(result["target"], str):
            raise ValueError("Audit targets must be strings")
        strata.setdefault((row["chapter"], result["target"] != row["current"]), []).append(sid)
    groups = list(strata.values())
    for group in groups:
        rng.shuffle(group)
    rng.shuffle(groups)
    selected: list[str] = []
    while len(selected) < count:
        for group in groups:
            if group and len(selected) < count:
                selected.append(group.pop())
    payload, mapping = [], []
    for sid in selected:
        row, result = indexed[sid], by_id[sid]
        candidate_is_a = bool(rng.getrandbits(1))
        payload.append(
            {
                "id": sid,
                "source": row["source"],
                "A": result["target"] if candidate_is_a else row["current"],
                "B": row["current"] if candidate_is_a else result["target"],
                "context": row["context2"],
            }
        )
        mapping.append({"id": sid, "candidate": "A" if candidate_is_a else "B"})
    folder = run / "audit"
    documents = {
        "blind-payload.json": {"segments": payload, "required_terms": frozen["required_terms"]},
        "private-map.json": {
            "seed": seed,
            "input_digest": digest(frozen),
            "results_digest": digest(results),
            "mapping": mapping,
        },
    }
    for name, value in documents.items():
        path = folder / name
        if path.exists() and read_json(path) != value:
            raise ValueError("Existing audit differs; use a separate run or audit directory")
    for name, value in documents.items():
        write_json(folder / name, value)
    return folder
