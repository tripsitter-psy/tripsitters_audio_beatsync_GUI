#!/usr/bin/env bash
set -euo pipefail
REQ=${1:-scripts/requirements.txt}
if ! command -v python >/dev/null 2>&1; then
  echo "python not found on PATH"
  exit 1
fi
python -m venv .venv
. ./.venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r "$REQ"
cat <<'EOF'
Done. Activate with: source .venv/bin/activate
EOF
