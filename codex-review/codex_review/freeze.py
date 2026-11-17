"""Freeze an explicit Wenyi CLI FileStorage target without opening source databases."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

from .storage import atomic_json as write_json
from .storage import load_json as read_json

Json = dict[str, Any]


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tree_hashes(path: Path) -> dict[str, str]:
    return {
        p.relative_to(path).as_posix(): file_hash(p) for p in sorted(path.rglob("*")) if p.is_file()
    }


def source_metadata(source: Path) -> Json:
    with zipfile.ZipFile(source) as archive:
        container = ET.fromstring(archive.read("META-INF/container.xml"))
        package_path = container.find(".//{*}rootfile").attrib["full-path"]
        package = ET.fromstring(archive.read(package_path))
    ns = "{http://purl.org/dc/elements/1.1/}"
    return {
        "title": [x.text or "" for x in package.findall(".//" + ns + "title")],
        "creator": [x.text or "" for x in package.findall(".//" + ns + "creator")],
    }


def verify_frozen(frozen: Json) -> None:
    if file_hash(Path(frozen["source_epub"])) != frozen["source_sha256"]:
        raise ValueError("Source EPUB changed since freezing")
    if tree_hashes(Path(frozen["state_root"])) != frozen["state_hashes"]:
        raise ValueError("Source target state changed since freezing")
    if frozen.get("prior_run"):
        prior = Path(frozen["prior_run"])
        if any(
            not (prior / name).is_file() or file_hash(prior / name) != value
            for name, value in frozen["prior_hashes"].items()
        ):
            raise ValueError("Prior review fingerprints changed")
        if any(
            not Path(name).is_file() or file_hash(Path(name)) != value
            for name, value in frozen.get("prior_artifacts", {}).items()
        ):
            raise ValueError("Prior publication artifact changed")


def freeze(
    state: Path,
    source: Path,
    run: Path,
    *,
    terms: dict[str, str] | None = None,
    scope_ids: list[str] | None = None,
    prior_run: Path | None = None,
    backend: str = "cli",
) -> Json:
    state, source, run = state.resolve(), source.resolve(), run.resolve()
    if (
        backend != "cli"
        or not (state / "manifest.json").is_file()
        or not (state / "chapters").is_dir()
    ):
        raise ValueError(
            "Only Wenyi CLI FileStorage target directories are supported; Web/Desktop are unsupported"
        )
    if run == state or run.is_relative_to(state) or state.is_relative_to(run):
        raise ValueError("Review run and source target state must be separate directories")
    existing_names = {entry.name for entry in run.iterdir()} if run.exists() else set()
    known_names = {
        ".gitignore",
        "frozen.json",
        "manifest.json",
        "checkpoint.json",
        "tasks",
        "attempts",
        "runtime.json",
        "runtime.lock",
        "results.json",
        "audit",
        "change-index.json",
        "published-state",
        "publication-report.json",
    }
    frozen_path = run / "frozen.json"
    partial_private_run = (
        existing_names == {".gitignore"} and (run / ".gitignore").read_text() == "*\n"
    )
    if existing_names and not partial_private_run:
        if not frozen_path.is_file() or existing_names - known_names:
            raise ValueError("Run directory contains existing unknown data")
    manifest = read_json(state / "manifest.json")
    source_hash = file_hash(source)
    if manifest.get("fmt") != "epub" or manifest.get("source_sha256") != source_hash:
        raise ValueError("CLI target must be bound to the supplied source EPUB")
    language = manifest.get("target_lang")
    if not isinstance(language, str) or not language:
        raise ValueError("Target manifest requires target_lang")
    required_terms = dict(terms or {})
    if any(
        not isinstance(k, str) or not k or not isinstance(v, str) or not v
        for k, v in required_terms.items()
    ):
        raise ValueError("Required terms must map nonempty source strings to target strings")
    baseline = tree_hashes(state)
    segments: list[Json] = []
    chapters = manifest.get("chapters", [])
    if not chapters or len({c["index"] for c in chapters}) != len(chapters):
        raise ValueError("Manifest must contain unique chapters")
    for chapter in chapters:
        index = chapter["index"]
        data = read_json(state / "chapters" / f"ch{index}.json")
        if data["index"] != index:
            raise ValueError("Chapter identity differs from manifest")
        local = []
        for segment in data["segments"]:
            if segment["kind"] not in ("text", "heading"):
                continue
            if not isinstance(segment["source"], str) or not isinstance(segment.get("target"), str):
                raise ValueError(
                    "All reviewable segments require source and completed target strings"
                )
            local.append(
                {
                    "id": f"c{index}:s{segment['index']}",
                    "chapter": index,
                    "text_index": len(local),
                    "segment_index": segment["index"],
                    "kind": segment["kind"],
                    "source": segment["source"],
                    "current": segment["target"],
                    "resource_href": segment.get("resource_href"),
                }
            )
        for position, row in enumerate(local):
            for width in (2, 6):
                row[f"context{width}"] = [
                    {
                        "id": neighbor["id"],
                        "source": neighbor["source"],
                        "target": neighbor["current"],
                    }
                    for offset, neighbor in enumerate(local)
                    if offset != position and abs(offset - position) <= width
                ]
        segments.extend(local)
    ids = [row["id"] for row in segments]
    scope = ids if scope_ids is None else list(scope_ids)
    if (
        not ids
        or len(set(ids)) != len(ids)
        or not scope
        or len(set(scope)) != len(scope)
        or not set(scope) <= set(ids)
    ):
        raise ValueError("Segment identities and scope must be nonempty, unique and known")
    prior_hashes = {}
    prior_artifacts = {}
    if prior_run:
        prior_run = prior_run.resolve()
        for name in (
            "frozen.json",
            "manifest.json",
            "checkpoint.json",
            "results.json",
            "publication-report.json",
        ):
            if (prior_run / name).is_file():
                prior_hashes[name] = file_hash(prior_run / name)
        report = prior_run / "publication-report.json"
        if report.exists():
            old_output = Path(read_json(report)["output"])
            if not old_output.is_file():
                raise ValueError("Prior publication artifact is missing")
            prior_artifacts[str(old_output)] = file_hash(old_output)
        if not prior_hashes:
            raise ValueError("Prior run has no review artifacts")
    frozen = {
        "version": 1,
        "backend": "cli",
        "source_epub": str(source),
        "source_sha256": source_hash,
        "state_root": str(state),
        "state_hashes": baseline,
        "target_language": language,
        "source_metadata": source_metadata(source),
        "segments": segments,
        "scope_ids": scope,
        "required_terms": required_terms,
        "prior_run": str(prior_run) if prior_run else None,
        "prior_hashes": prior_hashes,
        "prior_artifacts": prior_artifacts,
    }
    verify_frozen(frozen)
    path = run / "frozen.json"
    if path.exists():
        if read_json(path) != frozen:
            raise ValueError("Existing frozen inputs differ; use a new run directory")
    else:
        run.mkdir(parents=True, exist_ok=True)
        if not existing_names:
            # Install privacy protection before writing any frozen book content.
            (run / ".gitignore").write_text("*\n", encoding="utf-8")
        write_json(path, frozen)
    return frozen
