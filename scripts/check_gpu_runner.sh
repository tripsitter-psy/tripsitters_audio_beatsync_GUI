#!/usr/bin/env bash
set -euo pipefail

echo "== GPU Runner Validation Script =="

# NVIDIA visibility
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi found — output:"
  nvidia-smi || true
else
  echo "WARNING: nvidia-smi not found. Install NVIDIA driver and ensure nvidia-smi is on PATH." >&2
fi

# Docker GPU test
CUDA_IMAGE=${CUDA_IMAGE:-nvidia/cuda:12.0-base}
if command -v docker >/dev/null 2>&1; then
  echo "Docker found — testing container GPU access (this will pull a small image)..."
  if timeout 120s docker run --rm --gpus all "$CUDA_IMAGE" nvidia-smi >/dev/null 2>&1; then
    echo "Docker GPU test succeeded."
  else
    echo "WARNING: Docker GPU test failed or timed out. Ensure nvidia-docker / NVIDIA Container Toolkit is installed." >&2
  fi
else
  echo "Docker not found — skipping Docker GPU test."
fi

# nvcc presence
if command -v nvcc >/dev/null 2>&1; then
  echo "nvcc found — version:"
  nvcc --version || true
else
  echo "WARNING: nvcc not found. Install CUDA toolkit if you plan to build with CUDA." >&2
fi

# TEMP disk space check
TMPDIR=${TMPDIR:-/tmp}
if command -v python3 >/dev/null 2>&1; then
  available_gb=$(python3 -c 'import shutil,sys; print(shutil.disk_usage(sys.argv[1]).free // (1024**3))' "$TMPDIR" 2>/dev/null)
  py_status=$?
  if [ $py_status -ne 0 ] || ! [[ "$available_gb" =~ ^[0-9]+$ ]]; then
    echo "Temp path: $TMPDIR (Free: unknown)"
    echo "WARNING: Could not determine free space in TEMP (python3 error or non-numeric output). Skipping disk space check." >&2
  else
    echo "Temp path: $TMPDIR (Free: ${available_gb}GB)"
    if [ "$available_gb" -lt 10 ]; then
      echo "WARNING: Less than 10GB free in TEMP. nvcc and builds may fail with 'ptxas' or 'No space left' errors." >&2
    fi
  fi
else
  echo "Temp path: $TMPDIR (python3 not found — skipping disk space check)"
fi


echo "Validation done. Review warnings above and fix as needed."

# Optional GitHub runner check (requires GITHUB_TOKEN and GITHUB_REPOSITORY in OWNER/REPO form)
if [ -n "${GITHUB_TOKEN:-}" ] && [ -n "${GITHUB_REPOSITORY:-}" ]; then
  if ! command -v jq >/dev/null 2>&1; then
    echo "WARNING: jq is not installed. Skipping GitHub runner check for GPU label."
  else
    echo "Checking GitHub self-hosted runners for repo: $GITHUB_REPOSITORY"
    api="https://api.github.com/repos/$GITHUB_REPOSITORY/actions/runners"
    RUNNERS_JSON=$(mktemp) || { echo "Failed to create temp file"; exit 1; }
    trap "rm -f \"$RUNNERS_JSON\"" EXIT
    GITHUB_API_OK=true
    if curl -fS --max-time 20 -H "Authorization: token $GITHUB_TOKEN" -H "Accept: application/vnd.github+json" "$api" >"$RUNNERS_JSON"; then
      echo "Runners in repo (name => labels):"
      jq -r '.runners[] | "\(.name) => \(.labels | map(.name) | join(", "))"' "$RUNNERS_JSON" || true
      if jq -e '.runners | any(.labels[]?.name == "gpu")' "$RUNNERS_JSON" >/dev/null 2>&1; then
        echo "Found runner with 'gpu' label."
      else
        echo "WARNING: No runner with 'gpu' label found in repo runners." >&2
      fi
    else
      echo "WARNING: Failed to query GitHub API for runners." >&2
      GITHUB_API_OK=false
    fi
  fi
else
  echo "No GITHUB_TOKEN/GITHUB_REPOSITORY set; skipping GitHub runner label check. Set GITHUB_REPOSITORY='owner/repo' and export GITHUB_TOKEN to enable."
fi


echo "Validation done. Review warnings above and fix as needed."

# Exit with failure if GitHub API check was attempted but failed
if [ "${GITHUB_API_OK:-}" = false ]; then
  echo "ERROR: GitHub API validation failed." >&2
  exit 1
fi
