"""Real Core ingestion and default EPUB publication using only synthetic data."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from codex_review.cli import main as cli_main
from codex_review.publication import export_epub
from codex_review.workflow import Controller
from wenyi_core.ingest.segmenter import load_document
from wenyi_core.storage.file import FileStorage


def write_epub(path: Path) -> None:
    container = """<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>
<rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
</rootfiles></container>"""
    opf = """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">
<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:identifier id="id">urn:test:review</dc:identifier>
<dc:title>Synthetic Review</dc:title><dc:creator>Example Author</dc:creator><dc:language>en</dc:language></metadata>
<manifest><item id="body" href="body.xhtml" media-type="application/xhtml+xml"/></manifest>
<spine><itemref idref="body"/></spine></package>"""
    body = """<html xmlns="http://www.w3.org/1999/xhtml"><body>
<p>Example text<span id="page1" role="doc-pagebreak" title="1"/>.</p>
</body></html>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("OEBPS/content.opf", opf)
        archive.writestr("OEBPS/body.xhtml", body)


class CoreSmokeTests(unittest.TestCase):
    def test_real_ingest_freeze_review_and_default_epub_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.epub"
            write_epub(source)
            state = root / "source-state"
            document = load_document(str(source), "en", "zh")
            store = FileStorage(str(state))
            manifest = store.stage_document(document)
            seen = 0
            for chapter_meta in manifest["chapters"]:
                chapter = store.load_chapter(chapter_meta["index"])
                for segment in chapter.segments:
                    if segment.kind not in ("text", "heading"):
                        continue
                    segment.target = "初译文字。"
                    annotations = segment.meta.get("epub_annotations")
                    if annotations:
                        annotations["target_digest"] = hashlib.sha256(
                            segment.target.encode()
                        ).hexdigest()
                        annotations["placements"] = [
                            {
                                "id": item["id"],
                                "mode": "point",
                                "target_start": len(segment.target) - 1,
                                "target_end": len(segment.target) - 1,
                                "status": "aligned",
                                "method": "test",
                            }
                            for item in annotations["items"]
                        ]
                        seen += len(annotations["items"])
                store.save_chapter(chapter)
            manifest["initialized"] = True
            store.save_manifest(manifest)
            store.close()
            self.assertEqual(seen, 1)
            run = root / "review-run"
            self.assertEqual(
                cli_main(
                    [
                        "freeze",
                        "--state",
                        str(state),
                        "--source",
                        str(source),
                        "--run-dir",
                        str(run),
                    ]
                ),
                0,
            )
            frozen = json.loads((run / "frozen.json").read_text())
            controller = Controller.initialize(run, "attention")
            triage = controller.next_tasks()[0]
            sid = frozen["scope_ids"][0]
            controller.submit(
                triage.id,
                {
                    "results": [
                        {
                            "id": sid,
                            "status": "issue",
                            "issues": [
                                {
                                    "detail": "Synthetic issue",
                                    "source_quote": "Example",
                                    "suggestion": "Repair",
                                }
                            ],
                        }
                    ]
                },
                "triage-1",
            )
            fix = controller.next_tasks()[0]
            controller.submit(
                fix.id, {"results": [{"id": sid, "replacement": "修订后的中文。"}]}, "fix-1"
            )
            acceptance = controller.next_tasks()[0]
            controller.submit(
                acceptance.id,
                {"results": [{"id": sid, "decision": "accept", "issues": []}]},
                "accept-1",
            )
            output = root / "reviewed.epub"
            report = export_epub(run, controller.validate_for_publish(), output)
            self.assertEqual(report["changes"], 1)
            with zipfile.ZipFile(output) as archive:
                self.assertIsNone(archive.testzip())
                body = ET.fromstring(archive.read("OEBPS/body.xhtml"))
                self.assertIn("修订后的中文。", "".join(body.itertext()))
                self.assertTrue(
                    any(
                        node.get("id") == "page1" and node.get("role") == "doc-pagebreak"
                        for node in body.iter()
                    )
                )
                opf = ET.fromstring(archive.read("OEBPS/content.opf"))
                ns = "{http://purl.org/dc/elements/1.1/}"
                self.assertEqual(
                    [x.text for x in opf.findall(".//" + ns + "title")], ["Synthetic Review"]
                )
                self.assertEqual(
                    [x.text for x in opf.findall(".//" + ns + "creator")], ["Example Author"]
                )


if __name__ == "__main__":
    unittest.main()
