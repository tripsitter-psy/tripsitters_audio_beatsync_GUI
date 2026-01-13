#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "Starting tracing collector (docker compose up -d)..."
docker compose up -d
echo "Collector and Jaeger should be available. Jaeger UI: http://localhost:16686"
