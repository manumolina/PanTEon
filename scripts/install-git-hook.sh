#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOOK_SCRIPT="$ROOT/scripts/pre-commit.sh"
HOOK_PATH="$(git -C "$ROOT" rev-parse --git-dir)/hooks/pre-commit"

cat >"$HOOK_PATH" <<EOF
#!/usr/bin/env bash
exec "$HOOK_SCRIPT" run
EOF

chmod +x "$HOOK_PATH" "$HOOK_SCRIPT"
echo "Git hook installed at $HOOK_PATH"
