# python-rope-refactor

A small Codex CLI skill that nudges the agent to use `rope` (via a wrapper script) for *mechanical* Python refactors: renames, moves, import updates, etc.

## Disclaimer

This skill is under development!
- it is intentionally pragmatic
- it may have sharp edges
- please treat it like a helpful power tool, not a polished library

## Prerequisites

Make sure [uv](https://docs.astral.sh/uv/getting-started/installation/) is installed and available.

## What's in here

- `SKILL.md`: the skill instructions Codex reads (includes a rope-first workflow + examples)
- `scripts/rope_refactor.py`: a small CLI wrapper around common Rope refactorings

The wrapper is designed around a safer loop:
1) run with `--dry-run`
2) review the diff-like output
3) re-run with `--apply`
4) review `git diff` and run tests/typechecks

## What it can do (via `scripts/rope_refactor.py`)

- Move a module into another package (`move-module`)
- Rename a module file (`rename-module`)
- Rename a symbol (class/function/variable) (`rename-symbol`)
- Extract function / method (`extract-function`, `extract-method`)
- Inline variable / method (`inline-variable`, `inline-method`)
- Organize imports (`organize-imports`)
- Run a JSON-driven batch of operations (`batch`)

Run `uv run scripts/rope_refactor.py --help` for the full command list.

## Installing / using as a Codex skill

This repo is meant to be copied into your Codex skills directory.

Typical location:
- `~/.codex/skills/python-rope-refactor/`

Minimum required files:
- `SKILL.md`
- `scripts/rope_refactor.py`

Once installed, Codex will discover it at startup and (when prompted for mechanical Python refactors) prefer Rope over hand-editing imports/usages.

## Limitations / gotchas

- Rope cannot reliably update dynamic/string-based references (for example `importlib.import_module("pkg.mod")` or config strings like `"pkg.mod:Class"`).
- If scan roots are too narrow, Rope may miss references; widen scanning or scan the whole project (slower).
- Rope APIs can differ across versions; this was built around Rope 1.14.x.
