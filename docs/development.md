# Development

This guide covers code quality checks for PanTEon contributors. Everything runs inside Docker — you do not need conda, a local Python environment, or `pip install` on your machine.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running
- [Git](https://git-scm.com/)

## Overview

| Component | Purpose |
|-----------|---------|
| `Dockerfile.lint` | Minimal image with pre-commit and Git |
| `scripts/pre-commit.sh` | Wrapper that builds and runs checks in Docker |
| `scripts/install-git-hook.sh` | Installs a Git hook that calls the wrapper on commit |
| `.pre-commit-config.yaml` | Hook definitions (Ruff linter + formatter) |
| `pyproject.toml` | Ruff configuration (line length, rules, exclusions) |

The lint image (`panteon-lint`) is separate from the runtime image (`panteon:cpu`). The runtime container does not include development tools.

## Quick start

From the repository root:

```bash
# Run checks on all Python files
./scripts/pre-commit.sh run --all-files

# Run checks only on staged files (same as a commit)
./scripts/pre-commit.sh run

# Optional: run Ruff checks automatically before every commit
./scripts/install-git-hook.sh
```

The first run builds the `panteon-lint` image (~150 MB). Subsequent runs reuse it. Ruff and its hook environments are cached under `.cache/pre-commit/` (gitignored).

## What gets checked

pre-commit runs two Ruff hooks (see `.pre-commit-config.yaml`):

1. **ruff-check** — linting with auto-fix (`--fix`)
2. **ruff-format** — code formatting

Configuration lives in `pyproject.toml`:

- **Line length**: 120 characters
- **Python target**: 3.10
- **Excluded paths**: `features/kanalyze-2.0.0/`, `Custom_classifiers/*_unused`
- **Rules**: pycodestyle errors/warnings, pyflakes, isort, pyupgrade

## Git hook

`install-git-hook.sh` writes a pre-commit hook to `.git/hooks/pre-commit` that delegates to the Docker wrapper. On each `git commit`, only staged files are checked.

If a hook modifies files (e.g. Ruff auto-fix or format), stage the changes and commit again:

```bash
git add -u
git commit
```

## Manual image build

The wrapper builds the image automatically when needed. To build explicitly:

```bash
docker build -f Dockerfile.lint -t panteon-lint .
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PANTEON_LINT_IMAGE` | `panteon-lint` | Docker image name for lint runs |
| `PANTEON_PRE_COMMIT_CACHE` | `.cache/pre-commit` | Host path for pre-commit/Ruff cache |

Example:

```bash
PANTEON_LINT_IMAGE=panteon-lint:dev ./scripts/pre-commit.sh run --all-files
```

## Local install (optional)

If you prefer running pre-commit directly on the host (without Docker), install from `requirements-dev.txt`:

```bash
pip install -r requirements-dev.txt
pre-commit install
pre-commit run --all-files
```

This is optional. The Docker workflow above is the recommended approach when you want to avoid local Python dependencies.

## Troubleshooting

**`git failed. Is it installed...`**

The lint container includes Git. If you see ownership errors, ensure you run checks via `scripts/pre-commit.sh` (it runs the container as your user and sets `safe.directory`).

**Hook fails on commit**

Run manually to see full output:

```bash
./scripts/pre-commit.sh run --all-files
```

**Slow first run**

The first execution downloads Ruff and creates hook environments in `.cache/pre-commit/`. Later runs are much faster.

**Modified files after a check**

Ruff may reformat or auto-fix code. Review the diff, stage the changes, and commit again.
