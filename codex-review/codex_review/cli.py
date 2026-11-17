"""Command-line entry point for standalone Codex review runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .storage import exclusive_lock
from .workflow import Controller


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def config(path: Path | None) -> dict[str, Any]:
    value = read_json(path) if path else {}
    if not isinstance(value, dict) or set(value) - {"models", "runtime"}:
        raise ValueError("Configuration permits only models and runtime sections")
    if any(not isinstance(value.get(key, {}), dict) for key in ("models", "runtime")):
        raise ValueError("Configuration sections must be objects")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Review a frozen Wenyi CLI EPUB snapshot.")
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--run-dir", type=Path, required=True)
    freeze.add_argument("--state", type=Path, required=True)
    freeze.add_argument("--source", type=Path, required=True)
    freeze.add_argument("--terms-json", type=Path)
    freeze.add_argument("--scope-json", type=Path)
    freeze.add_argument("--prior-run", type=Path)
    init = commands.add_parser("init")
    init.add_argument("--run-dir", type=Path, required=True)
    init.add_argument("--mode", choices=("full", "attention"), required=True)
    init.add_argument("--config", type=Path)
    run = commands.add_parser("run")
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--config", type=Path)
    run.add_argument("--resume", action="store_true")
    for name in ("status", "results"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
    decide = commands.add_parser("decide")
    decide.add_argument("--run-dir", type=Path, required=True)
    decide.add_argument("--id", required=True)
    choice = decide.add_mutually_exclusive_group(required=True)
    choice.add_argument("--retain", action="store_true")
    choice.add_argument("--replacement-file", type=Path)
    decide.add_argument("--reason", required=True)
    resume = commands.add_parser("resume")
    resume.add_argument("--run-dir", type=Path, required=True)
    resume.add_argument("--task-id", required=True)
    publish = commands.add_parser("publish")
    publish.add_argument("--run-dir", type=Path, required=True)
    publish.add_argument("--output", type=Path, required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--run-dir", type=Path, required=True)
    amount = audit.add_mutually_exclusive_group()
    amount.add_argument("--count", type=int)
    amount.add_argument("--ratio", type=float)
    audit.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    directory = args.run_dir.resolve()
    if args.command == "freeze":
        from .freeze import freeze as freeze_snapshot

        frozen = freeze_snapshot(
            args.state,
            args.source,
            directory,
            terms=read_json(args.terms_json) if args.terms_json else None,
            scope_ids=read_json(args.scope_json) if args.scope_json else None,
            prior_run=args.prior_run,
        )
        outcome = {"status": "frozen", "segments": len(frozen["segments"])}
    elif args.command == "run":
        from .runtime import RuntimeConfig
        from .runtime import run as run_tasks

        settings = config(args.config)
        controller = Controller(directory)
        if any(
            controller.manifest["models"].get(stage) != model
            for stage, model in settings.get("models", {}).items()
        ):
            raise ValueError("Model assignments are frozen at init; use a new run to change them.")
        outcome = run_tasks(
            controller, RuntimeConfig(**settings.get("runtime", {})), resume=args.resume
        )
    elif args.command in ("status", "results"):
        controller = Controller(directory)
        outcome = controller.status() if args.command == "status" else controller.results()
    else:
        with exclusive_lock(directory / "runtime.lock"):
            if args.command == "init":
                controller = Controller.initialize(
                    directory, args.mode, config(args.config).get("models")
                )
                outcome = controller.status()
            elif args.command == "decide":
                controller = Controller(directory)
                replacement = (
                    args.replacement_file.read_text(encoding="utf-8")
                    if args.replacement_file
                    else None
                )
                controller.decide(
                    args.id, retain=args.retain, replacement=replacement, reason=args.reason
                )
                outcome = controller.status()
            elif args.command == "resume":
                controller = Controller(directory)
                controller.resume(args.task_id)
                outcome = controller.status()
            elif args.command == "publish":
                from .publication import export_epub

                outcome = export_epub(directory, None, args.output)
            else:
                from .audit import create_audit

                controller = Controller(directory)
                results = controller.validate_for_publish()
                path = create_audit(
                    directory, results, count=args.count, ratio=args.ratio, seed=args.seed
                )
                outcome = {"status": "prepared", "path": str(path)}
    print(json.dumps(outcome, ensure_ascii=False, indent=2, default=str))
    return 0


def entrypoint() -> None:
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
