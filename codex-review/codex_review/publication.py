"""Publish verified review changes into an independent CLI state and EPUB."""

from __future__ import annotations

import copy
import difflib
import hashlib
import os
import shutil
import sqlite3
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .freeze import file_hash, read_json, verify_frozen, write_json
from .storage import canonical_digest as digest

Json = dict[str, Any]
REVISION = 1


def _copy_state(source: Path, destination: Path) -> None:
    shutil.copytree(
        source, destination, ignore=lambda _path, names: [n for n in names if n.endswith("-shm")]
    )
    for original in source.rglob("*"):
        if original.is_file() and original.suffix in (".db", ".sqlite", ".sqlite3"):
            target = destination / original.relative_to(source)
            fd, name = tempfile.mkstemp(dir=target.parent)
            os.close(fd)
            with (
                closing(sqlite3.connect(target)) as copied,
                closing(sqlite3.connect(name)) as merged,
            ):
                copied.backup(merged)
            for suffix in ("-wal", "-shm"):
                target.with_name(target.name + suffix).unlink(missing_ok=True)
            os.replace(name, target)


def _pagebreaks(path: Path) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    with zipfile.ZipFile(path) as archive:
        for href in archive.namelist():
            if not href.lower().endswith((".html", ".xhtml", ".htm")):
                continue
            for node in ET.fromstring(archive.read(href)).iter():
                if node.tag.rsplit("}", 1)[-1] == "span" and node.get("role") == "doc-pagebreak":
                    identity = (href, node.get("id", ""))
                    if not identity[1] or identity in found:
                        raise ValueError("Pagebreak identities must be nonempty and unique")
                    found.add(identity)
    return found


def _map_point(before: str, after: str, point: Any) -> int | None:
    if not isinstance(point, int) or isinstance(point, bool) or not 0 <= point <= len(before):
        return None
    for tag, left, right, start, _ in difflib.SequenceMatcher(
        None, before, after, autojunk=False
    ).get_opcodes():
        if tag == "equal" and left <= point <= right:
            return start + point - left
    return None


def _annotation_changes(frozen: Json, changes: list[Json]) -> dict[str, Json]:
    state = Path(frozen["state_root"])
    registry = state / "annotation_contexts.json"
    contexts = read_json(registry).get("contexts", {}) if registry.exists() else {}
    amended = {}
    with zipfile.ZipFile(frozen["source_epub"]) as archive:
        for change in changes:
            chapter = read_json(state / "chapters" / f"ch{change['chapter']}.json")
            segment = [x for x in chapter["segments"] if x["kind"] in ("text", "heading")][
                change["text_index"]
            ]
            annotation = segment.get("meta", {}).get("epub_annotations", {})
            # Wenyi annotations hash raw target bytes, unlike task canonical JSON digests.
            if (
                annotation.get("target_digest")
                != hashlib.sha256(change["before"].encode()).hexdigest()
            ):
                continue
            items, placements = annotation.get("items", []), annotation.get("placements", [])
            if not items or len(items) != len(placements):
                continue
            ids = [item.get("id") for item in items]
            if len(set(ids)) != len(ids) or set(ids) != {p.get("id") for p in placements}:
                continue
            updated = copy.deepcopy(annotation)
            by_id = {p["id"]: p for p in updated["placements"]}
            valid = True
            for item in items:
                place = by_id[item["id"]]
                point = place.get("target_start")
                context = contexts.get(item.get("target_key"), {})
                blocks = context.get("source_blocks", [])
                href = context.get("resource_href")
                if (
                    item.get("relation") != "inline_object"
                    or item.get("mode") != "point"
                    or item.get("object_tag") != "span"
                    or item.get("source_text") != ""
                    or item.get("marker_text") != ""
                    or place.get("status") != "aligned"
                    or place.get("mode") != "point"
                    or point != place.get("target_end")
                    or len(blocks) != 1
                    or href != segment.get("resource_href")
                ):
                    valid = False
                    break
                block = ET.fromstring(blocks[0])
                if (
                    block.tag.rsplit("}", 1)[-1] != "span"
                    or block.get("role") != "doc-pagebreak"
                    or not block.get("id")
                    or len(block)
                    or (block.text or "")
                ):
                    valid = False
                    break
                source_nodes = [
                    n
                    for n in ET.fromstring(archive.read(href)).iter()
                    if n.get("id") == block.get("id")
                ]
                mapped = _map_point(change["before"], change["after"], point)
                if (
                    len(source_nodes) != 1
                    or source_nodes[0].tag.rsplit("}", 1)[-1] != "span"
                    or source_nodes[0].attrib != block.attrib
                    or len(source_nodes[0])
                    or (source_nodes[0].text or "")
                    or mapped is None
                ):
                    valid = False
                    break
                place["target_start"] = mapped
                place["target_end"] = mapped
            if valid:
                updated["target_digest"] = hashlib.sha256(change["after"].encode()).hexdigest()
                amended[change["id"]] = updated
    return amended


def prepare(run: Path, results: list[Json] | None = None) -> Json:
    from .workflow import Controller

    run = run.resolve()
    frozen = read_json(run / "frozen.json")
    verify_frozen(frozen)
    verified = Controller(run).validate_for_publish()
    if results is not None and results != verified:
        raise ValueError("Supplied publication results differ from verified controller results")
    indexed = {row["id"]: row for row in frozen["segments"]}
    if len({r["id"] for r in verified}) != len(verified) or {r["id"] for r in verified} != set(
        frozen["scope_ids"]
    ):
        raise ValueError("Publication must cover the complete frozen scope exactly once")
    changes = []
    for result in verified:
        row = indexed[result["id"]]
        if result["decision"] not in ("accepted", "retain"):
            raise ValueError("Resolve pending and needs_user results before publication")
        if result["decision"] == "accepted":
            if result["accepted_digest"] != digest(result["target"]):
                raise ValueError("Accepted target digest differs")
            if result["target"] != row["current"]:
                changes.append(
                    {
                        **{
                            k: row[k]
                            for k in ("id", "chapter", "text_index", "segment_index", "source")
                        },
                        "before": row["current"],
                        "after": result["target"],
                        "evidence": result["evidence"],
                    }
                )
        elif result["target"] != row["current"]:
            raise ValueError("Retained target differs from frozen translation")
    annotations = _annotation_changes(frozen, changes)
    index = {
        "version": REVISION,
        "frozen_digest": digest(frozen),
        "results_digest": digest(verified),
        "changes": changes,
        "annotations": annotations,
    }
    path = run / "change-index.json"
    if path.exists() and read_json(path) != index:
        raise ValueError("Existing publication index differs")
    if not path.exists():
        write_json(path, index)
    state = run / "published-state"
    if not state.exists():
        stage = Path(tempfile.mkdtemp(prefix=".state-", dir=run))
        stage.rmdir()
        _copy_state(Path(frozen["state_root"]), stage)
        verify_frozen(frozen)
        os.replace(stage, state)
    for name, expected_hash in frozen["state_hashes"].items():
        path = Path(name)
        if (
            path.parts[0] != "chapters"
            and path.suffix not in (".db", ".sqlite", ".sqlite3")
            and not name.endswith(("-wal", "-shm"))
        ):
            if not (state / path).is_file() or file_hash(state / path) != expected_hash:
                raise ValueError("Non-target publication state changed")
    by_position = {(r["chapter"], r["text_index"]): r for r in changes}
    for chapter in {r["chapter"] for r in frozen["segments"]}:
        relative = Path("chapters") / f"ch{chapter}.json"
        original = read_json(Path(frozen["state_root"]) / relative)
        current = read_json(state / relative)
        expected = copy.deepcopy(original)
        text_index = 0
        for segment in expected["segments"]:
            if segment["kind"] not in ("text", "heading"):
                continue
            change = by_position.get((chapter, text_index))
            if change:
                if (
                    segment["index"] != change["segment_index"]
                    or segment["source"] != change["source"]
                    or segment["target"] != change["before"]
                ):
                    raise ValueError("Frozen segment identity differs")
                segment["target"] = change["after"]
                if change["id"] in annotations:
                    segment["meta"]["epub_annotations"] = annotations[change["id"]]
            text_index += 1
        # An interrupted apply may leave either old or new values, but no unrelated changes.
        if current != original and current != expected:
            raise ValueError("Unexpected contents in publication state")
        if current != expected:
            write_json(state / relative, expected)
    verify_frozen(frozen)
    return index


def _normalize_metadata(source: Path, output: Path, language: str) -> None:
    with zipfile.ZipFile(source) as archive:
        container = ET.fromstring(archive.read("META-INF/container.xml"))
        source_package = ET.fromstring(
            archive.read(container.find(".//{*}rootfile").attrib["full-path"])
        )
    with zipfile.ZipFile(output) as archive:
        container = ET.fromstring(archive.read("META-INF/container.xml"))
        opf_path = container.find(".//{*}rootfile").attrib["full-path"]
        package = ET.fromstring(archive.read(opf_path))
        metadata = package.find("{*}metadata")
        ns = "{http://purl.org/dc/elements/1.1/}"
        for tag in ("title", "creator"):
            for node in list(metadata):
                if node.tag == ns + tag:
                    metadata.remove(node)
            for node in source_package.findall(".//" + ns + tag):
                metadata.append(copy.deepcopy(node))
        for node in list(metadata):
            if node.tag == ns + "language" or node.get("property") == "dcterms:modified":
                metadata.remove(node)
        ET.SubElement(metadata, ns + "language").text = language
        ET.SubElement(
            metadata, "{http://www.idpf.org/2007/opf}meta", {"property": "dcterms:modified"}
        ).text = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        replacement = ET.tostring(package, encoding="utf-8", xml_declaration=True)
        entries = [(info, archive.read(info.filename)) for info in archive.infolist()]
    stage = output.with_suffix(".metadata.tmp")
    with zipfile.ZipFile(stage, "w") as archive:
        for info, content in entries:
            archive.writestr(info, replacement if info.filename == opf_path else content)
    os.replace(stage, output)


def _render(state: Path, source: Path, output: Path) -> None:
    from wenyi_core.assemble.writer import assemble
    from wenyi_core.storage.file import FileStorage

    assemble(
        FileStorage(str(state), create=False),
        str(source),
        out_path=str(output),
        out_format="epub",
        about_page=False,
    )


def export_epub(
    run: Path,
    results: list[Json] | None,
    output: Path,
    *,
    renderer: Callable[[Path, Path, Path], None] | None = None,
) -> Json:
    run, output = run.resolve(), output.resolve()
    index = prepare(run, results)
    frozen = read_json(run / "frozen.json")
    source = Path(frozen["source_epub"])
    source_state = Path(frozen["state_root"])
    if (
        output == source
        or output.is_relative_to(source_state)
        or output.is_relative_to(run / "published-state")
    ):
        raise ValueError("Output must be a new artifact outside source and publication state")
    expected_pages = _pagebreaks(source)
    report_path = run / "publication-report.json"
    if output.exists():
        report = read_json(report_path) if report_path.exists() else {}
        if (
            report.get("index_digest") == digest(index)
            and report.get("output_sha256") == file_hash(output)
            and report.get("output") == str(output)
        ):
            if expected_pages - _pagebreaks(output):
                raise ValueError("Published EPUB lost source pagebreaks")
            return report
        raise ValueError("Output already exists without a matching publication intent")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".review-", suffix=".epub", dir=output.parent)
    os.close(fd)
    stage = Path(name)
    try:
        (renderer or _render)(run / "published-state", source, stage)
        _normalize_metadata(source, stage, frozen["target_language"])
        with zipfile.ZipFile(stage) as archive:
            if archive.testzip() is not None:
                raise ValueError("Rendered EPUB is corrupt")
        if expected_pages - _pagebreaks(stage):
            raise ValueError("Rendered EPUB lost source pagebreaks")
        if prepare(run, results) != index:
            raise ValueError("Publication inputs changed during rendering")
        report = {
            "version": REVISION,
            "index_digest": digest(index),
            "output": str(output),
            "output_sha256": file_hash(stage),
            "changes": len(index["changes"]),
        }
        write_json(report_path, report)
        os.replace(stage, output)
        return report
    finally:
        stage.unlink(missing_ok=True)
