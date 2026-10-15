#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
VENV="${VENV:-/home/public/tbiom/.venv}"
PYTHON="${PYTHON:-$VENV/bin/python}"
"$PYTHON" -m pip install -q -r requirements.txt
echo "Env ready: $PYTHON"

