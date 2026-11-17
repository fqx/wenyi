"""Exercise real local child processes without invoking a model or a service."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from codex_review.runtime import RuntimeConfig, run
from codex_review.storage import atomic_json, exclusive_lock, load_json
from codex_review.workflow import ResponseError


@pytest.fixture
def executable(tmp_path: Path) -> Path:
    path = tmp_path / "fake-codex"
    path.write_text(
        f"#!{sys.executable}\n"
        + """import json, os, pathlib, sys, time
args = sys.argv[1:]
if args[:2] == ["login", "status"]:
    print("Logged in using ChatGPT")
    sys.exit(0)
folder = pathlib.Path.cwd()
metadata = json.loads((folder / "metadata.json").read_text())
payload = json.loads((folder / "payload.json").read_text())
record = {"start": time.time(), "args": args, "role": payload["role"],
          "secret_present": any(os.environ.get(k) for k in ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL"))}
(folder / "worker.json").write_text(json.dumps(record))
behavior = payload.get("behavior", "ok")
if behavior == "timeout":
    time.sleep(30)
else:
    sys.stdin.read()
    time.sleep(payload.get("delay", 0.05))
    output = pathlib.Path(args[args.index("-o") + 1])
    invalid = behavior == "invalid" or (behavior == "invalid_once" and folder.name.endswith("0001"))
    output.write_text("{" if invalid else json.dumps({"ok": True}))
    print(json.dumps({"type": "thread.started", "thread_id": "synthetic-thread"}), flush=True)
    record["end"] = time.time()
    (folder / "worker.json").write_text(json.dumps(record))
""",
        encoding="utf-8",
    )
    path.chmod(0o700)
    return path


class FakeController:
    def __init__(self, directory: Path, tasks: list[SimpleNamespace], failure_limit: int = 2):
        self.run_dir = directory
        directory.mkdir(exist_ok=True)
        self.tasks = tasks
        self.done: set[str] = set()
        self.failures: dict[str, list[str]] = {}
        self.failure_limit = failure_limit

    def reload(self) -> None:
        pass

    def status(self) -> dict:
        blocked = [key for key, value in self.failures.items() if len(value) >= self.failure_limit]
        return {"complete": len(self.done) == len(self.tasks), "blocked": blocked}

    def next_tasks(self, limit: int, exclude_ids: set[str]) -> list[SimpleNamespace]:
        return [
            item
            for item in self.tasks
            if item.id not in self.done | exclude_ids
            and all(parent in self.done for parent in item.dependencies)
        ][:limit]

    def submit(self, task_id: str, response: dict, attempt: str) -> dict:
        if response != {"ok": True}:
            raise ResponseError("Invalid synthetic result")
        duplicate = task_id in self.done
        self.done.add(task_id)
        return {"status": "duplicate" if duplicate else "completed"}

    def fail(self, task_id: str, attempt: str, kind: str, reason: str) -> dict:
        values = self.failures.setdefault(task_id, [])
        if attempt not in values:
            values.append(attempt)
        return {"status": "blocked" if len(values) >= self.failure_limit else "retry"}


def task(identifier: str, role: str = "judge", **kwargs) -> SimpleNamespace:
    return SimpleNamespace(
        id=identifier,
        stage="fix" if role == "repair" else "accept",
        role=role,
        model="synthetic-model",
        payload={"role": role, **kwargs.pop("payload", {})},
        schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
        dependencies=kwargs.pop("dependencies", []),
        session_key=kwargs.pop("session_key", None),
    )


def config(executable: Path, **kwargs) -> RuntimeConfig:
    return RuntimeConfig(codex=str(executable), poll_seconds=0.01, **kwargs)


def test_dynamic_role_limits_and_private_environment(tmp_path: Path, executable: Path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-value")
    monkeypatch.setenv("CODEX_API_KEY", "synthetic-value")
    monkeypatch.setenv("OPENAI_BASE_URL", "synthetic-value")
    tasks = [
        task(f"task-{index}", role, payload={"delay": 0.25})
        for index, role in enumerate(["judge", "judge", "repair", "repair", "scan", "scan"] * 2)
    ]
    controller = FakeController(tmp_path / "run", tasks)
    result = run(controller, config(executable, role_limits={"judge": 2, "repair": 2, "scan": 2}))
    assert result["status"]["complete"]
    intervals = [load_json(path) for path in controller.run_dir.glob("attempts/*/*/worker.json")]
    assert len(intervals) == len(tasks)
    assert not any(item["secret_present"] for item in intervals)
    maximum = 0
    for item in intervals:
        running = [other for other in intervals if other["start"] <= item["start"] < other["end"]]
        maximum = max(maximum, len(running))
        assert len(running) <= 6
        for role in ("scan", "repair", "judge"):
            assert sum(other["role"] == role for other in running) <= 2
    assert maximum >= 4


def test_persistent_repair_and_fresh_acceptance(tmp_path: Path, executable: Path):
    items = [
        task("first", "repair", session_key="batch"),
        task("second", "repair", session_key="batch", dependencies=["first"]),
        task("judge", dependencies=["second"]),
    ]
    controller = FakeController(tmp_path / "run", items)
    run(controller, config(executable))
    arguments = {}
    for item in items:
        arguments[item.id] = load_json(
            controller.run_dir / "attempts" / item.id / "attempt-0001" / "worker.json"
        )["args"]
    assert "--ephemeral" not in arguments["first"]
    assert arguments["second"][:2] == ["exec", "resume"]
    assert "--ephemeral" in arguments["judge"] and "resume" not in arguments["judge"]


def test_format_retry_and_block_drains_workers(tmp_path: Path, executable: Path):
    controller = FakeController(
        tmp_path / "retry", [task("recover", payload={"behavior": "invalid_once"})]
    )
    assert run(controller, config(executable))["status"]["complete"]
    assert len(controller.failures["recover"]) == 1
    blocked = FakeController(
        tmp_path / "blocked",
        [
            task("bad", payload={"behavior": "invalid", "delay": 0.01}),
            task("inflight", payload={"delay": 0.3}),
            task("not-dispatched"),
        ],
        failure_limit=1,
    )
    result = run(blocked, config(executable, max_workers=2))
    assert result["pause"] and "inflight" in blocked.done
    assert not (blocked.run_dir / "attempts" / "not-dispatched").exists()
    before = list(blocked.run_dir.glob("attempts/*/*"))
    run(blocked, config(executable))
    assert list(blocked.run_dir.glob("attempts/*/*")) == before


def test_timeout_does_not_block_on_stdin(tmp_path: Path, executable: Path):
    controller = FakeController(
        tmp_path / "run",
        [
            task(
                "slow",
                payload={
                    "behavior": "timeout",
                    "large_input": "x" * 200_000,
                },
            )
        ],
        failure_limit=1,
    )
    start = time.monotonic()
    result = run(controller, config(executable, timeout_seconds=0.2))
    assert time.monotonic() - start < 5
    assert result["pause"] and not controller.done


def test_surviving_worker_and_lock_are_not_duplicated(tmp_path: Path, executable: Path):
    controller = FakeController(tmp_path / "run", [task("held")])
    atomic_json(
        controller.run_dir / "runtime.json",
        {
            "active": {"held": {"attempt": "attempt-0001", "pid": os.getpid()}},
        },
    )
    result = run(controller, config(executable), resume=True)
    assert result["pause"] and not list(controller.run_dir.glob("attempts/*"))
    with exclusive_lock(controller.run_dir / "runtime.lock"), pytest.raises(RuntimeError):
        run(controller, config(executable))


def test_api_login_is_rejected_without_dispatch(tmp_path: Path, executable: Path):
    executable.write_text(
        executable.read_text().replace("Logged in using ChatGPT", "Logged in using an API key")
    )
    controller = FakeController(tmp_path / "run", [task("not-dispatched")])
    with pytest.raises(ValueError, match="ChatGPT login is required"):
        run(controller, config(executable))
    assert not list(controller.run_dir.glob("attempts/*"))
