"""Offline behavioral tests for the review controller."""

import json
import tempfile
import unittest
from pathlib import Path

from codex_review.workflow import Controller, ResponseError


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def create(self, mode="attention", count=4):
        rows = [
            {
                "id": f"s{i}",
                "source": f"Original sentence {i}.",
                "current": f"Current {i}",
                "chapter": 1,
                "context2": [],
                "context6": [{"id": "far", "source": "far source", "target": "far"}],
            }
            for i in range(count)
        ]
        rows.append(
            {
                "id": "outside",
                "source": "The distant reference is specific.",
                "current": "outside",
                "chapter": 2,
                "context2": [],
                "context6": [],
            }
        )
        frozen = {
            "version": 1,
            "source_sha256": "fake",
            "segments": rows,
            "scope_ids": [r["id"] for r in rows[:-1]],
            "required_terms": {"sample": "示例"},
        }
        (self.run / "frozen.json").write_text(json.dumps(frozen), encoding="utf-8")
        return Controller.initialize(self.run, mode)

    def test_run_rejects_model_config_changes(self):
        from codex_review.cli import main

        self.create()
        configuration = self.run / "config.local.json"
        configuration.write_text(json.dumps({"models": {"triage": "different-model"}}))
        with self.assertRaisesRegex(ValueError, "frozen at init"):
            main(["run", "--run-dir", str(self.run), "--config", str(configuration)])

    def test_saturated_role_skips_existing_and_new_tasks(self):
        c = self.create(mode="full")
        scan = c.next_tasks()[0]
        c.submit(scan.id, self.response(scan), "scan")
        assert c.next_tasks(exclude_roles={"repair"}) == []
        assert len(c.state["tasks"]) == 1
        verify = c.next_tasks()[0]
        assert verify.stage == "verify"
        assert c.next_tasks(exclude_roles={"repair"}) == []
        assert c.next_tasks()[0].id == verify.id

    @staticmethod
    def response(task, status="clear", issue_id=None):
        return {
            "results": [
                {
                    "id": row["id"],
                    "status": "issue" if row["id"] == issue_id else status,
                    "issues": [
                        {"detail": "Wrong term", "source_quote": "Original", "suggestion": "Fix"}
                    ]
                    if row["id"] == issue_id
                    else [],
                }
                for row in task.payload["segments"]
            ]
        }

    def test_partial_quote_retry_and_duplicate_attempt(self):
        c = self.create()
        task = c.next_tasks()[0]
        bad = self.response(task, issue_id="s1")
        bad["results"][1]["issues"][0]["source_quote"] = "original"
        with self.assertRaises(ResponseError):
            c.submit(task.id, bad, "one")
        self.assertEqual(c.fail(task.id, "one", "format", "bad quote")["status"], "retry")
        self.assertEqual(c.fail(task.id, "one", "format", "bad quote")["status"], "retry")
        self.assertEqual(c.state["tasks"][task.id]["format_failures"], 1)
        partial = c.next_tasks()[0]
        self.assertEqual([x["id"] for x in partial.payload["segments"]], ["s1"])
        self.assertEqual(
            partial.schema["properties"]["results"]["items"]["properties"]["id"]["enum"], ["s1"]
        )
        fixed = {
            "results": [
                {
                    "id": "s1",
                    "status": "issue",
                    "issues": [
                        {"detail": "Wrong term", "source_quote": "Original", "suggestion": "Fix"}
                    ],
                }
            ]
        }
        self.assertEqual(c.submit(task.id, fixed, "two")["status"], "completed")
        self.assertEqual(len(c.state["tasks"][task.id]["response"]["results"]), 4)
        self.assertEqual(c.submit(task.id, fixed, "two")["status"], "completed")

    def test_full_64_row_partial_retry_retains_63_rows(self):
        c = self.create(mode="full", count=64)
        task = c.next_tasks()[0]
        self.assertEqual(len(task.payload["segments"]), 64)
        bad = self.response(task, issue_id="s1")
        bad["results"][1]["issues"][0]["source_quote"] = "original"
        with self.assertRaises(ResponseError):
            c.submit(task.id, bad, "first")
        c.fail(task.id, "first", "format", "quote")
        partial = c.next_tasks()[0]
        self.assertEqual([x["id"] for x in partial.payload["segments"]], ["s1"])
        bad_row = {"results": [bad["results"][1]]}
        with self.assertRaises(ResponseError):
            c.submit(task.id, bad_row, "second")
        c.fail(task.id, "second", "format", "quote")
        self.assertEqual(c.status()["blocked"], [task.id])
        c.resume(task.id)
        self.assertEqual([x["id"] for x in c.next_tasks()[0].payload["segments"]], ["s1"])
        corrected = self.response(partial, issue_id="s1")
        c.submit(task.id, corrected, "third")
        self.assertEqual(len(c.state["tasks"][task.id]["response"]["results"]), 64)

    def test_adjudicator_must_quote_supplied_source_and_replacement_is_rechecked(self):
        c = self.create(count=1)
        triage = c.next_tasks()[0]
        c.submit(triage.id, self.response(triage, "needs_evidence"), "1")
        expanded = c.next_tasks()[0]
        c.submit(expanded.id, self.response(expanded, "needs_evidence"), "1")
        adjudicate = c.next_tasks()[0]
        c.submit(
            adjudicate.id,
            {
                "results": [
                    {
                        "id": "s0",
                        "decision": "search",
                        "reason": "",
                        "replacement": "",
                        "search_queries": ["distant reference"],
                        "evidence": [],
                    }
                ]
            },
            "1",
        )
        adjudicate = c.next_tasks()[0]
        response = {
            "results": [
                {
                    "id": "s0",
                    "decision": "replace",
                    "reason": "The distant reference settles it",
                    "replacement": "Reviewed",
                    "search_queries": [],
                    "evidence": [{"source_id": "outside", "source_quote": "distant reference"}],
                }
            ]
        }
        c.submit(adjudicate.id, response, "1")
        acceptance = c.next_tasks()[0]
        self.assertEqual(acceptance.stage, "accept")
        self.assertNotIn("search_results", acceptance.payload["segments"][0])
        self.assertIn(
            {"id": "outside", "source": "The distant reference is specific."},
            acceptance.payload["segments"][0]["context"],
        )
        self.assertNotIn("confirmed_issues", acceptance.payload["segments"][0])
        c.submit(
            acceptance.id, {"results": [{"id": "s0", "decision": "accept", "issues": []}]}, "1"
        )
        self.assertEqual(c.validate_for_publish()[0]["evidence"]["basis"], "adjudicator")

    def test_missing_or_duplicate_result_forces_whole_batch(self):
        c = self.create()
        task = c.next_tasks()[0]
        bad = self.response(task)
        bad["results"][-1] = bad["results"][0]
        with self.assertRaises(ResponseError):
            c.submit(task.id, bad, "bad")
        c.fail(task.id, "bad", "format", "duplicate")
        self.assertEqual(len(c.next_tasks()[0].payload["segments"]), 4)

    def test_ambiguous_context_search_and_user_replacement_rechecked(self):
        c = self.create(count=1)
        task = c.next_tasks()[0]
        c.submit(task.id, self.response(task, "needs_evidence"), "1")
        expanded = c.next_tasks()[0]
        self.assertEqual(expanded.stage, "expanded")
        self.assertEqual(len(expanded.payload["segments"][0]["context"]), 1)
        c.submit(expanded.id, self.response(expanded, "needs_evidence"), "1")
        adjudicate = c.next_tasks()[0]
        self.assertEqual(adjudicate.stage, "adjudicate")
        c.submit(
            adjudicate.id,
            {
                "results": [
                    {
                        "id": "s0",
                        "decision": "search",
                        "reason": "",
                        "replacement": "",
                        "search_queries": ["distant reference"],
                        "evidence": [],
                    }
                ]
            },
            "1",
        )
        adjudicate = c.next_tasks()[0]
        self.assertEqual(adjudicate.payload["segments"][0]["search_results"][0]["id"], "outside")
        c.submit(
            adjudicate.id,
            {
                "results": [
                    {
                        "id": "s0",
                        "decision": "needs_user",
                        "reason": "Uncertain",
                        "replacement": "",
                        "search_queries": [],
                        "evidence": [],
                    }
                ]
            },
            "1",
        )
        self.assertEqual(c.results()[0]["decision"], "needs_user")
        with self.assertRaises(ValueError):
            c.validate_for_publish()
        c.decide("s0", replacement="Human proposed", reason="Reviewed context")
        accept = c.next_tasks()[0]
        self.assertEqual(accept.stage, "accept")
        self.assertIsNone(accept.session_key)
        c.submit(accept.id, {"results": [{"id": "s0", "decision": "accept", "issues": []}]}, "1")
        result = c.validate_for_publish()[0]
        self.assertEqual(result["target"], "Human proposed")
        self.assertEqual(result["evidence"]["basis"], "user")

    def test_repairs_preserve_original_findings_and_budget(self):
        c = self.create(count=1)
        task = c.next_tasks()[0]
        c.submit(task.id, self.response(task, issue_id="s0"), "1")
        for n in range(3):
            fix = c.next_tasks()[0]
            self.assertEqual(fix.stage, "fix")
            self.assertEqual(
                fix.payload["segments"][0]["confirmed_issues"][0]["detail"], "Wrong term"
            )
            c.submit(fix.id, {"results": [{"id": "s0", "replacement": f"Fixed {n}"}]}, "1")
            accept = c.next_tasks()[0]
            self.assertEqual(accept.stage, "accept")
            self.assertNotIn("confirmed_issues", accept.payload["segments"][0])
            c.submit(
                accept.id,
                {
                    "results": [
                        {
                            "id": "s0",
                            "decision": "reject",
                            "issues": [
                                {
                                    "detail": "Another issue",
                                    "source_quote": "Original",
                                    "suggestion": "Fix",
                                }
                            ],
                        }
                    ]
                },
                "1",
            )
        self.assertEqual(c.next_tasks()[0].stage, "adjudicate")

    def test_full_sample_failure_expands_and_publish_detects_tamper(self):
        c = self.create(mode="full", count=20)
        task = c.next_tasks()[0]
        self.assertEqual(task.stage, "scan")
        c.submit(task.id, self.response(task), "1")
        task = c.next_tasks()[0]
        self.assertEqual(task.stage, "verify")
        c.submit(task.id, self.response(task), "1")
        sample = c.next_tasks()[0]
        self.assertEqual(sample.stage, "accept")
        self.assertEqual(len(sample.payload["segments"]), 2)
        ids = [x["id"] for x in sample.payload["segments"]]
        c.submit(
            sample.id,
            {
                "results": [
                    {
                        "id": sid,
                        "decision": "reject" if i == 0 else "accept",
                        "issues": [
                            {"detail": "Error", "source_quote": "Original", "suggestion": "Fix"}
                        ]
                        if i == 0
                        else [],
                    }
                    for i, sid in enumerate(ids)
                ]
            },
            "1",
        )
        expanded = c.next_tasks()[0]
        self.assertEqual(expanded.stage, "accept")
        self.assertEqual(len(expanded.payload["segments"]), 18)
        c.submit(
            expanded.id,
            {
                "results": [
                    {"id": r["id"], "decision": "accept", "issues": []}
                    for r in expanded.payload["segments"]
                ]
            },
            "1",
        )
        fix = c.next_tasks()[0]
        c.submit(fix.id, {"results": [{"id": ids[0], "replacement": "Fixed"}]}, "1")
        accept = c.next_tasks()[0]
        c.submit(accept.id, {"results": [{"id": ids[0], "decision": "accept", "issues": []}]}, "1")
        self.assertEqual(len(c.validate_for_publish()), 20)
        checkpoint = json.loads((self.run / "checkpoint.json").read_text())
        checkpoint["tasks"][accept.id]["response"]["results"][0]["decision"] = "reject"
        (self.run / "checkpoint.json").write_text(json.dumps(checkpoint))
        with self.assertRaises(ValueError):
            Controller(self.run).validate_for_publish()

    def test_adjudicator_retain_explicitly_binds_frozen_current(self):
        c = self.create(count=1)
        triage = c.next_tasks()[0]
        c.submit(triage.id, self.response(triage, issue_id="s0"), "1")
        for n in range(3):
            fix = c.next_tasks()[0]
            c.submit(fix.id, {"results": [{"id": "s0", "replacement": f"Candidate {n}"}]}, "1")
            accept = c.next_tasks()[0]
            c.submit(
                accept.id,
                {
                    "results": [
                        {
                            "id": "s0",
                            "decision": "reject",
                            "issues": [
                                {"detail": "Wrong", "source_quote": "Original", "suggestion": "Fix"}
                            ],
                        }
                    ]
                },
                "1",
            )
        adjudicate = c.next_tasks()[0]
        row = adjudicate.payload["segments"][0]
        self.assertEqual(row["current"], "Current 0")
        self.assertEqual(row["proposed"], "Candidate 2")
        self.assertIn("Retain means retain current", adjudicate.payload["instruction"])
        c.submit(
            adjudicate.id,
            {
                "results": [
                    {
                        "id": "s0",
                        "decision": "retain",
                        "reason": "Original is preferable",
                        "replacement": "",
                        "search_queries": [],
                        "evidence": [{"source_id": "s0", "source_quote": "Original"}],
                    }
                ]
            },
            "1",
        )
        result = c.validate_for_publish()[0]
        self.assertEqual(result["decision"], "retain")
        self.assertEqual(result["target"], "Current 0")

    def test_transport_block_and_explicit_resume(self):
        c = self.create(count=1)
        task = c.next_tasks()[0]
        c.fail(task.id, "a", "transport", "timeout")
        self.assertEqual(c.fail(task.id, "b", "transport", "timeout")["status"], "blocked")
        self.assertFalse(c.status()["complete"])
        self.assertEqual(c.next_tasks(), [])
        Controller(self.run).resume(task.id)
        self.assertEqual(Controller(self.run).next_tasks()[0].id, task.id)


if __name__ == "__main__":
    unittest.main()
