"""Additional independent runtime and terminal-decision contracts."""

from __future__ import annotations

import os
import signal
from pathlib import Path

import pytest
from codex_review import runtime
from codex_review.storage import atomic_json, load_json
from codex_review.workflow import Controller
from test_runtime import FakeController, config, task
from test_runtime import executable as executable


def test_sigint_during_dispatch_launches_no_additional_worker(
    tmp_path: Path, executable: Path, monkeypatch
):
    controller = FakeController(tmp_path / "run", [task("one"), task("two"), task("three")])
    real_popen = runtime.subprocess.Popen
    calls = []

    def interrupt_after_first_launch(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        if args[0][1:3] == ["login", "status"]:
            return child
        calls.append(child.pid)
        if len(calls) == 1:
            os.kill(os.getpid(), signal.SIGINT)
        return child

    monkeypatch.setattr(runtime.subprocess, "Popen", interrupt_after_first_launch)
    result = runtime.run(controller, config(executable))
    assert result["stopping"]
    assert len(calls) == 1
    assert controller.done == {"one"}
    assert not result["active"]


def test_ready_repair_tasks_with_one_session_are_serialized(tmp_path: Path, executable: Path):
    controller = FakeController(
        tmp_path / "run",
        [
            task("one", "repair", session_key="same-batch", payload={"delay": 0.1}),
            task("two", "repair", session_key="same-batch", payload={"delay": 0.1}),
        ],
    )
    result = runtime.run(controller, config(executable))
    assert result["status"]["complete"]
    first = load_json(controller.run_dir / "attempts/one/attempt-0001/worker.json")
    second = load_json(controller.run_dir / "attempts/two/attempt-0001/worker.json")
    assert first["end"] <= second["start"]
    assert second["args"][:2] == ["exec", "resume"]


def test_recovery_replays_one_transport_attempt_without_double_count(
    tmp_path: Path, executable: Path
):
    controller = FakeController(tmp_path / "run", [task("recover")])
    folder = controller.run_dir / "attempts/recover/attempt-0001"
    folder.mkdir(parents=True)
    controller.fail("recover", "attempt-0001", "transport", "Worker interrupted")
    atomic_json(folder / "worker-result.json", {"failed": True, "reason": "Worker interrupted"})
    atomic_json(
        controller.run_dir / "runtime.json",
        {
            "active": {"recover": {"attempt": "attempt-0001", "pid": 999999999}},
        },
    )
    result = runtime.run(controller, config(executable))
    assert result["status"]["complete"]
    assert controller.failures["recover"] == ["attempt-0001"]
    assert len(list((controller.run_dir / "attempts/recover").glob("attempt-*"))) == 2


def test_orphan_completed_repair_recovers_its_session(tmp_path: Path, executable: Path):
    controller = FakeController(
        tmp_path / "run",
        [
            task("first", "repair", session_key="same-batch"),
            task("second", "repair", session_key="same-batch", dependencies=["first"]),
        ],
    )
    folder = controller.run_dir / "attempts/first/attempt-0001"
    folder.mkdir(parents=True)
    atomic_json(folder / "model-final.json", {"ok": True})
    atomic_json(folder / "metadata.json", {"session_key": "same-batch", "persistent_session": True})
    (folder / "events.jsonl").write_text(
        '{"type":"thread.started","thread_id":"recovered-thread"}\n'
    )
    atomic_json(
        controller.run_dir / "runtime.json",
        {
            "active": {"first": {"attempt": "attempt-0001", "pid": 999999999}},
        },
    )
    runtime.run(controller, config(executable))
    arguments = load_json(controller.run_dir / "attempts/second/attempt-0001/worker.json")["args"]
    assert arguments[:3] == ["exec", "resume", "recovered-thread"]


def attention_controller(path: Path) -> Controller:
    rows = [
        {
            "id": "local",
            "chapter": 0,
            "source": "The sibling followed.",
            "current": "Old target.",
            "context2": [],
            "context6": [],
        },
        {
            "id": "distant",
            "chapter": 1,
            "source": "The sibling was younger.",
            "current": "Distant target.",
            "context2": [],
            "context6": [],
        },
    ]
    atomic_json(
        path / "frozen.json",
        {
            "version": 1,
            "source_sha256": "synthetic",
            "segments": rows,
            "scope_ids": ["local"],
            "required_terms": {},
        },
    )
    return Controller.initialize(path, "attention")


def submit_uncertain(controller: Controller):
    for stage in ("triage", "expanded"):
        work = controller.next_tasks()[0]
        assert work.stage == stage
        controller.submit(
            work.id, {"results": [{"id": "local", "status": "needs_evidence", "issues": []}]}, "one"
        )
    return controller.next_tasks()[0]


def adjudication(decision: str, **kwargs) -> dict:
    row = {
        "id": "local",
        "decision": decision,
        "reason": "Source reviewed",
        "replacement": "",
        "search_queries": [],
        "evidence": [],
    }
    row.update(kwargs)
    return {"results": [row]}


def test_adjudicator_replacement_rejection_finishes_needs_user(tmp_path: Path):
    controller = attention_controller(tmp_path)
    judge = submit_uncertain(controller)
    controller.submit(judge.id, adjudication("search", search_queries=["younger"]), "one")
    judge = controller.next_tasks()[0]
    assert judge.payload["segments"][0]["search_results"] == [
        {"id": "distant", "source": "The sibling was younger."}
    ]
    controller.submit(
        judge.id,
        adjudication(
            "replace",
            replacement="New target.",
            evidence=[{"source_id": "distant", "source_quote": "younger"}],
        ),
        "one",
    )
    accept = controller.next_tasks()[0]
    assert accept.stage == "accept" and accept.session_key is None
    assert "confirmed_issues" not in accept.payload["segments"][0]
    controller.submit(
        accept.id,
        {
            "results": [
                {
                    "id": "local",
                    "decision": "reject",
                    "issues": [
                        {"detail": "Still wrong", "source_quote": "sibling", "suggestion": "Review"}
                    ],
                }
            ]
        },
        "one",
    )
    assert controller.status()["complete"]
    assert controller.results()[0]["decision"] == "needs_user"
    assert controller.next_tasks() == []
    with pytest.raises(ValueError):
        controller.validate_for_publish()


def test_repeated_source_search_is_bounded(tmp_path: Path):
    controller = attention_controller(tmp_path)
    judge = submit_uncertain(controller)
    controller.submit(judge.id, adjudication("search", search_queries=["younger"]), "one")
    judge = controller.next_tasks()[0]
    controller.submit(judge.id, adjudication("search", search_queries=["younger"]), "one")
    assert controller.results()[0]["decision"] == "needs_user"
    assert controller.next_tasks() == []


def test_frozen_tampering_is_not_a_format_retry(tmp_path: Path):
    controller = attention_controller(tmp_path)
    work = controller.next_tasks()[0]
    frozen = load_json(tmp_path / "frozen.json")
    frozen["segments"][0]["source"] = "Altered source."
    atomic_json(tmp_path / "frozen.json", frozen)
    with pytest.raises(RuntimeError):
        controller.submit(
            work.id, {"results": [{"id": "local", "status": "clear", "issues": []}]}, "one"
        )
    checkpoint = load_json(tmp_path / "checkpoint.json")
    assert checkpoint["tasks"][work.id]["format_failures"] == 0
    assert checkpoint["tasks"][work.id]["attempts"] == {}
