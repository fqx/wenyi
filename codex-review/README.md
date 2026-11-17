# Codex Review

Codex Review is a local, resumable review runner for translation projects. It freezes a source snapshot, delegates review stages to Codex CLI, records decisions under a run directory, and publishes only after every required review decision is resolved.

## Requirements

- Python 3.10 or newer.
- macOS or Linux; the runner uses POSIX locks and process groups.
- Codex CLI installed and already signed in with your ChatGPT account.
- A local Wenyi project and its source EPUB.

The runner checks that Codex CLI is signed in with ChatGPT and rejects API-key authentication. It has no API-key or endpoint setting. Keep the frozen input and all run data in a private local directory.

## Quick start

Run `uv sync --locked` once. Execute the examples from the repository root in its virtual environment, or replace `python` with `uv run --no-sync python`. `--state` selects a CLI target directory containing `manifest.json` and `chapters/`, not the top-level `state_dir` or a Web/Desktop workspace.

Replace the placeholder inputs with local project paths, freeze a run, then initialize and run it:

```bash
python codex-review/review.py freeze --run-dir codex-review/runs/example --state ./private-project/state --source ./private-project/source.epub
python codex-review/review.py init --run-dir codex-review/runs/example --mode full --config codex-review/config.example.json
python codex-review/review.py run --run-dir codex-review/runs/example --config codex-review/config.example.json
python codex-review/review.py status --run-dir codex-review/runs/example
```

`full` reviews every segment. Use `attention` to triage a selected set of segments from an existing review. Inspect the run with `results`; unresolved attention cases must be decided before publication. Publish to a separate EPUB path with `publish`. See [the workflow guide](docs/workflow.md) for stages, recovery, decisions, auditing, and publication.

The config example selects model IDs and worker limits. You can copy it to a private `*.local.json` file and edit it; that filename pattern is ignored by Git. Do not add real keys or endpoint URLs to the configuration.

## Data and privacy

The recommended local run directory is `codex-review/runs/`, which is ignored by Git. The runner requires an explicit `--run-dir` and adds a self-ignoring `.gitignore` to newly created run directories, including custom locations. Run directories contain source excerpts, proposed and repaired translations, task results, checkpoints, and publication artifacts. Treat them as private project data.

Publication writes a separate review state and EPUB output. It does not overwrite the source EPUB or source project state. Review steps and their output contracts are documented in [the workflow guide](docs/workflow.md).
