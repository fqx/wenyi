# Review workflow

Codex Review operates on a frozen input snapshot. A run directory owns its manifest, checkpoint, worker payloads and responses, decisions, audit material, and publication output. Resume the same run directory to continue; do not replace its frozen input after initialization.

## Modes

### Full review

1. **Scan:** review every requested source/translation pair for substantive fidelity.
2. **Verify:** independently check every pair, including pairs the scan marked clear.
3. **Fix:** make a minimal repair for confirmed issues, preserving already-correct text.
4. **Blind acceptance:** independently assess every repaired paragraph and a fixed 10% sample of unchanged paragraphs without upstream judgments. If the sample exposes an issue, expand acceptance to the remaining paragraphs. Unresolved cases enter the same bounded adjudication path described below.
5. **Publish:** validate that all required decisions are complete, then write a separate review state and EPUB.

### Attention review

1. **Triage:** independently classify each selected case using its initial context.
2. **Expand:** gather wider neutral context for cases that need it and reassess them.
3. **Fix:** repair confirmed issues. The workflow permits the initial repair and up to two additional repair attempts.
4. **Accept:** independently assess the repaired text.
5. **Adjudicate uncertainty:** for unresolved cases, an adjudicator can request searches against the full-text index, retain the current text, choose a proposed replacement, or return `needs_user`. Search results are evidence to inspect, not an automatic decision.
6. **Independent acceptance:** changes selected during adjudication receive a separate acceptance review.
7. **Publish:** any remaining `needs_user`, pending, or blocked decision prevents publication. Use `decide` to record a user's retain-or-replacement decision with a reason, then resume the run.

Stage dependencies are fixed. Model assignments are configurable at initialization and then frozen; worker limits may change when resuming. Worker caps are per role and cannot exceed the global worker limit. The example uses six workers overall, with limits of two scan, two repair, and four judge workers.

For a completed one-off review, freeze its latest accepted CLI state into a new run directory. Legacy prototype checkpoints are not this tool's checkpoint format; keep their private directories intact rather than overwriting them with `init`.

## Commands

Run commands from the repository root. Replace example paths and IDs with synthetic or local values; do not put private project paths or source text into committed examples.

```bash
python codex-review/review.py freeze --run-dir codex-review/runs/example --state ./private-project/state --source ./private-project/source.epub
python codex-review/review.py init --run-dir codex-review/runs/example --mode full --config codex-review/config.example.json
python codex-review/review.py run --run-dir codex-review/runs/example --config codex-review/config.example.json
python codex-review/review.py status --run-dir codex-review/runs/example
python codex-review/review.py results --run-dir codex-review/runs/example
python codex-review/review.py publish --run-dir codex-review/runs/example --output ./private-project/reviewed.epub
```

For an attention run, freeze a selected scope with `--scope-json PATH`, then initialize with `--mode attention`. Supply optional terminology and prior-run context with `--terms-json PATH` and `--prior-run PATH`. Use `decide` only for a case awaiting user input; provide either a retain decision or a replacement file, plus a reason. Use `resume` to requeue a blocked task after inspecting its failure, then use `run --resume` to restart the worker runtime. The separate `audit` command runs only after all review decisions are resolved. It creates a seeded, stratified blind A/B sample and a separate private mapping; it does not change review decisions. Keep both files private.

```bash
python codex-review/review.py decide --run-dir codex-review/runs/example --id synthetic-segment --retain --reason "Reviewed against the source."
python codex-review/review.py resume --run-dir codex-review/runs/example --task-id synthetic-task
python codex-review/review.py run --run-dir codex-review/runs/example --config codex-review/config.example.json --resume
python codex-review/review.py audit --run-dir codex-review/runs/example --ratio 0.1 --seed 0
```

## Configuration and login

The JSON configuration has `models` and `runtime` sections. `models` maps workflow roles (`scan`, `verify`, `triage`, `expanded`, `fix`, `accept`, and `adjudicate`) to Codex model IDs. `runtime` sets the Codex executable, global worker cap, per-role worker caps, task timeout, and poll interval. See [`config.example.json`](../config.example.json).

Sign in with ChatGPT through Codex CLI before starting a run. The runner checks the login mode and rejects API-key authentication, then removes common API key and endpoint environment variables from worker processes. Do not add credentials to configuration, command arguments, frozen data, or the repository.

## Run data and publication

The recommended `codex-review/runs/` path is ignored by Git. The CLI requires an explicit `--run-dir`; a custom run directory may be outside the repository, where this `.gitignore` does not apply. Run directories may contain source excerpts and translations as well as private review feedback. Do not commit run data, audit maps, or exports.

Publication creates a new review state and EPUB at a separately selected output path; it does not modify the source EPUB or source project state. Publication is blocked while any case is pending, blocked, or `needs_user`. Keep the run directory private even after publishing because it retains the review evidence and decisions.

## Recovery

The workflow records each task and checkpoint so interrupted work can resume from the same run directory. Inspect `status` and the task result before using `resume`; a resumed worker must not silently discard a completed task or change the frozen source. Validation failures are retried within the configured policy. If a task remains blocked, preserve its diagnostic output and resolve the input or task issue before continuing.
