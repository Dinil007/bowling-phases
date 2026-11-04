#!/usr/bin/env bash
# ScoutEdge bowling evidence pipeline - fetch weights wrapper
# Runs scripts/fetch_weights.py with verified SHA-256 checksums
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

python "$SCRIPT_DIR/fetch_weights.py" "$@"
