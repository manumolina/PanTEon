#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${PANTEON_LINT_IMAGE:-panteon-lint}"
CACHE_DIR="${PANTEON_PRE_COMMIT_CACHE:-$ROOT/.cache/pre-commit}"

mkdir -p "$CACHE_DIR"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Building lint image ($IMAGE)..."
    docker build -f "$ROOT/Dockerfile.lint" -t "$IMAGE" "$ROOT"
fi

docker run --rm \
    -u "$(id -u):$(id -g)" \
    -v "${ROOT}:/repo" \
    -v "${CACHE_DIR}:/cache/pre-commit" \
    -e PRE_COMMIT_HOME=/cache/pre-commit \
    -w /repo \
    "$IMAGE" "$@"
