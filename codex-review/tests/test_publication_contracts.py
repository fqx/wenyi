"""Synthetic-only freeze, publication and blind-audit contracts."""

from __future__ import annotations

import copy
import hashlib
import shutil
import sqlite3
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from codex_review.audit import create_audit
from codex_review.freeze import file_hash, freeze, read_json, tree_hashes, write_json
from codex_review.publication import _annotation_changes, _copy_state, export_epub, prepare
from codex_review.workflow import Controller


class PublicationContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.state = self.base / "source-state"
        self.run = self.base / "review"
        self.source = self.base / "source.epub"
        package = (
            '<package xmlns="http://www.idpf.org/2007/opf" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/"><metadata>'
            "<dc:title>Sample Title</dc:title><dc:creator>Sample Author</dc:creator>"
            "<dc:language>en</dc:language></metadata></package>"
        )
        with zipfile.ZipFile(self.source, "w") as archive:
            archive.writestr(
                "META-INF/container.xml",
                '<container><rootfiles><rootfile full-path="content.opf"/></rootfiles></container>',
            )
            archive.writestr("content.opf", package)
            archive.writestr(
                "chapter.xhtml",
                '<html xmlns="http://www.w3.org/1999/xhtml"><body><span id="p1" role="doc-pagebreak"/></body></html>',
            )
        self.chapter = {
            "index": 0,
            "segments": [
                {
                    "index": i,
                    "kind": "text",
                    "source": f"Source paragraph {i}.",
                    "target": f"Current paragraph {i}.",
                    "resource_href": "chapter.xhtml",
                }
                for i in range(8)
            ],
        }
        write_json(self.state / "chapters/ch0.json", self.chapter)
        write_json(
            self.state / "manifest.json",
            {
                "fmt": "epub",
                "source_sha256": file_hash(self.source),
                "target_lang": "fr",
                "chapters": [{"index": 0}],
            },
        )

    def tearDown(self):
        self.temp.cleanup()

    def initialize(self, scope=None):
        frozen = freeze(
            self.state, self.source, self.run, scope_ids=scope, terms={"Source": "Equivalent"}
        )
        Controller.initialize(self.run, "attention")
        return frozen

    def finish(self, changed=True):
        while True:
            controller = Controller(self.run)
            tasks = controller.next_tasks()
            if not tasks:
                return controller.results()
            task = tasks[0]
            rows = []
            for row in task.payload["segments"]:
                sid = row["id"]
                if task.stage in ("triage", "expanded"):
                    issue = changed and sid == "c0:s0"
                    rows.append(
                        {
                            "id": sid,
                            "status": "issue" if issue else "clear",
                            "issues": [
                                {
                                    "detail": "Missing meaning",
                                    "source_quote": "Source",
                                    "suggestion": "Restore meaning",
                                }
                            ]
                            if issue
                            else [],
                        }
                    )
                elif task.stage == "fix":
                    rows.append({"id": sid, "replacement": "Repaired paragraph."})
                elif task.stage == "accept":
                    rows.append({"id": sid, "decision": "accept", "issues": []})
                else:
                    self.fail("Unexpected synthetic task stage")
            controller.submit(task.id, {"results": rows}, "attempt-1")

    def test_freeze_stable_scope_and_context_is_idempotent(self):
        frozen = self.initialize(["c0:s0", "c0:s4"])
        self.assertEqual(len(frozen["segments"]), 8)
        self.assertEqual([r["id"] for r in frozen["segments"][0]["context2"]], ["c0:s1", "c0:s2"])
        self.assertEqual(len(frozen["segments"][0]["context6"]), 6)
        self.assertEqual(
            freeze(
                self.state,
                self.source,
                self.run,
                scope_ids=["c0:s0", "c0:s4"],
                terms={"Source": "Equivalent"},
            ),
            frozen,
        )
        with self.assertRaises(ValueError):
            freeze(self.state, self.source, self.base / "other", backend="desktop")

    def test_custom_run_ignores_every_runtime_file_in_real_git_repository(self):
        repository = self.base / "synthetic-repository"
        repository.mkdir()
        subprocess.run(["git", "init", "--quiet", str(repository)], check=True, capture_output=True)
        run = repository / "custom-private-runtime"
        freeze(self.state, self.source, run)
        write_json(run / "checkpoint.json", {"synthetic": "runtime data"})
        write_json(run / "attempts/task/response.json", {"synthetic": "response"})
        paths = [
            "custom-private-runtime/.gitignore",
            "custom-private-runtime/frozen.json",
            "custom-private-runtime/checkpoint.json",
            "custom-private-runtime/attempts/task/response.json",
        ]
        ignored = subprocess.run(
            ["git", "-C", str(repository), "check-ignore", "--", *paths],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(set(ignored.stdout.splitlines()), set(paths))
        self.assertEqual((run / ".gitignore").read_text(), "*\n")
        subprocess.run(["git", "-C", str(repository), "add", "-A"], check=True, capture_output=True)
        staged = subprocess.run(
            ["git", "-C", str(repository), "ls-files"], check=True, capture_output=True, text=True
        )
        self.assertEqual(staged.stdout, "")

    def test_freeze_never_adds_ignore_rules_to_existing_user_data_or_source(self):
        occupied = self.base / "occupied"
        occupied.mkdir()
        (occupied / "notes.txt").write_text("synthetic user note")
        with self.assertRaisesRegex(ValueError, "unknown data"):
            freeze(self.state, self.source, occupied)
        self.assertFalse((occupied / ".gitignore").exists())
        for run in (self.state, self.state / "nested-review"):
            with self.assertRaisesRegex(ValueError, "separate directories"):
                freeze(self.state, self.source, run)
        self.assertFalse((self.state / ".gitignore").exists())
        self.assertFalse((self.state / "nested-review").exists())

    def test_source_state_and_prior_artifacts_are_bound(self):
        self.initialize()
        self.source.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            prepare(self.run)

    def test_publish_only_accepted_scope_and_repeat(self):
        self.initialize(["c0:s0", "c0:s1"])
        results = self.finish()
        baseline = tree_hashes(self.state)
        first = prepare(self.run, results)
        self.assertEqual(prepare(self.run, results), first)
        self.assertEqual(len(first["changes"]), 1)
        copied = read_json(self.run / "published-state/chapters/ch0.json")
        self.assertEqual(copied["segments"][0]["target"], "Repaired paragraph.")
        self.assertEqual(copied["segments"][1:], self.chapter["segments"][1:])
        self.assertEqual(tree_hashes(self.state), baseline)
        fake = copy.deepcopy(results)
        fake[0]["target"] = "Unverified replacement"
        with self.assertRaises(ValueError):
            prepare(self.run, fake)

    def test_pending_and_tampered_artifacts_block_publication(self):
        self.initialize()
        with self.assertRaises(ValueError):
            prepare(self.run)
        results = self.finish()
        checkpoint = read_json(self.run / "checkpoint.json")
        task = next(t for t in checkpoint["tasks"].values() if t["task"]["stage"] == "accept")
        task["response"]["results"][0]["decision"] = "reject"
        write_json(self.run / "checkpoint.json", checkpoint)
        with self.assertRaises(ValueError):
            prepare(self.run, results)
        self.assertFalse((self.run / "published-state").exists())

    def test_copied_wal_includes_committed_rows_without_source_writes(self):
        database = self.state / "glossary.db"
        source = sqlite3.connect(database)
        source.execute("pragma journal_mode=wal")
        source.execute("create table terms(value text)")
        source.execute("insert into terms values ('example')")
        source.commit()
        try:
            baseline = tree_hashes(self.state)
            _copy_state(self.state, self.base / "copied")
            self.assertEqual(tree_hashes(self.state), baseline)
            copied = sqlite3.connect(self.base / "copied/glossary.db")
            try:
                self.assertEqual(copied.execute("select value from terms").fetchone()[0], "example")
            finally:
                copied.close()
        finally:
            source.close()

    def test_pagebreak_rebase_is_all_or_nothing_and_never_activates_stale(self):
        annotation = {
            "target_digest": hashlib.sha256(b"Current paragraph 0.").hexdigest(),
            "items": [
                {
                    "id": "point",
                    "relation": "inline_object",
                    "mode": "point",
                    "object_tag": "span",
                    "source_text": "",
                    "marker_text": "",
                    "target_key": "key",
                }
            ],
            "placements": [
                {
                    "id": "point",
                    "status": "aligned",
                    "mode": "point",
                    "target_start": 8,
                    "target_end": 8,
                }
            ],
        }
        self.chapter["segments"][0]["meta"] = {"epub_annotations": annotation}
        write_json(self.state / "chapters/ch0.json", self.chapter)
        write_json(
            self.state / "annotation_contexts.json",
            {
                "contexts": {
                    "key": {
                        "resource_href": "chapter.xhtml",
                        "source_blocks": ['<span id="p1" role="doc-pagebreak"/>'],
                    }
                }
            },
        )
        frozen = freeze(self.state, self.source, self.run)
        change = {
            "id": "c0:s0",
            "chapter": 0,
            "text_index": 0,
            "before": "Current paragraph 0.",
            "after": "Current new paragraph 0.",
        }
        rebased = _annotation_changes(frozen, [change])
        self.assertEqual(len(rebased), 1)
        annotation["items"].append({"id": "link", "relation": "link"})
        annotation["placements"].append({"id": "link", "status": "aligned"})
        write_json(self.state / "chapters/ch0.json", self.chapter)
        self.assertEqual(_annotation_changes(frozen, [change]), {})
        annotation["items"].pop()
        annotation["placements"].pop()
        annotation["target_digest"] = "stale"
        write_json(self.state / "chapters/ch0.json", self.chapter)
        self.assertEqual(_annotation_changes(frozen, [change]), {})

    def test_export_metadata_pagebreaks_and_cached_retry(self):
        self.initialize()
        results = self.finish()
        calls = []

        def render(_state, source, output):
            calls.append(True)
            shutil.copyfile(source, output)

        output = self.base / "new.epub"
        report = export_epub(self.run, results, output, renderer=render)
        self.assertEqual(export_epub(self.run, results, output, renderer=render), report)
        self.assertEqual(len(calls), 1)
        with zipfile.ZipFile(output) as archive:
            package = ET.fromstring(archive.read("content.opf"))
        self.assertEqual(package.find(".//{*}title").text, "Sample Title")
        self.assertEqual(package.find(".//{*}creator").text, "Sample Author")
        self.assertEqual(package.find(".//{*}language").text, "fr")
        self.assertTrue(any(n.get("property") == "dcterms:modified" for n in package.iter()))

    def test_missing_pagebreak_cannot_publish(self):
        self.initialize()
        results = self.finish()
        output = self.base / "missing-page.epub"

        def render(_state, source, destination):
            with zipfile.ZipFile(source) as old, zipfile.ZipFile(destination, "w") as new:
                for info in old.infolist():
                    content = old.read(info.filename)
                    if info.filename == "chapter.xhtml":
                        content = b"<html><body><p>Body text</p></body></html>"
                    new.writestr(info, content)

        with self.assertRaisesRegex(ValueError, "pagebreak"):
            export_epub(self.run, results, output, renderer=render)
        self.assertFalse(output.exists())
        self.assertFalse((self.run / "publication-report.json").exists())

    def test_interrupted_prepare_reapplies_and_unrelated_meta_is_rejected(self):
        self.initialize()
        results = self.finish()
        prepare(self.run, results)
        copied = self.run / "published-state/chapters/ch0.json"
        write_json(copied, self.chapter)
        prepare(self.run, results)
        self.assertEqual(read_json(copied)["segments"][0]["target"], "Repaired paragraph.")
        value = read_json(copied)
        value["segments"][4]["meta"] = {"unexpected": "mutation"}
        write_json(copied, value)
        with self.assertRaisesRegex(ValueError, "Unexpected contents"):
            prepare(self.run, results)

    def test_prior_artifact_fingerprint_blocks_changed_predecessor(self):
        prior = self.base / "prior-review"
        old_output = self.base / "prior-output.epub"
        shutil.copyfile(self.source, old_output)
        write_json(prior / "publication-report.json", {"output": str(old_output)})
        freeze(self.state, self.source, self.run, prior_run=prior)
        Controller.initialize(self.run, "attention")
        results = self.finish()
        old_output.write_bytes(b"changed predecessor")
        with self.assertRaisesRegex(ValueError, "Prior publication"):
            prepare(self.run, results)

    def test_seeded_blind_audit_is_separate_and_reproducible(self):
        self.initialize()
        results = self.finish()
        folder = create_audit(self.run, results, count=4, seed=13)
        blind = read_json(folder / "blind-payload.json")
        private = read_json(folder / "private-map.json")
        self.assertEqual(len(blind["segments"]), 4)
        self.assertTrue(
            all(set(row) == {"id", "source", "A", "B", "context"} for row in blind["segments"])
        )
        self.assertIn("c0:s0", {row["id"] for row in blind["segments"]})
        self.assertEqual(create_audit(self.run, results, count=4, seed=13), folder)
        self.assertEqual(read_json(folder / "private-map.json"), private)
        with self.assertRaises(ValueError):
            create_audit(self.run, results, count=4, ratio=0.5)


if __name__ == "__main__":
    unittest.main()
