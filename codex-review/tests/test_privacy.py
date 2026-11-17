"""Check that synthetic local runtime and configuration paths are ignored."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_local_runtime_paths_are_ignored() -> None:
    paths = (
        "codex-review/runs/synthetic-run/checkpoint.json",
        "codex-review/artifacts/synthetic-export.json",
        "codex-review/local/synthetic-cache.json",
        "codex-review/config.local.json",
        "codex-review/.env.synthetic",
    )
    for path in paths:
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "--quiet", "--", path],
            cwd=ROOT,
            check=False,
        )
        assert result.returncode == 0, f"Expected synthetic local path to be ignored: {path}"
