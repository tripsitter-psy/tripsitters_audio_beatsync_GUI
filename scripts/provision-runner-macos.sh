#!/usr/bin/env bash
# Safe helper to provision a self-hosted GitHub Actions runner on macOS.
# Installs common packages via Homebrew and downloads the runner artifact.
# Requires a registration token to complete runner configuration (printed at end).

set -euo pipefail
REPO_URL="${1:-}"        # e.g. https://github.com/<owner>/<repo>
RUNNER_NAME="${2:-ue5-mac-runner}"
LABELS="${3:-self-hosted,macos,ue5-5.3}"
WORKDIR="${4:-/opt/actions-runner}"
# Optional arg 5: github PAT used to request ephemeral registration token. Falls back to env GITHUB_PAT
GITHUB_PAT="${5:-${GITHUB_PAT:-}}"
# Optional arg 6: pass 1 to dispatch the smoke workflow after registration
RUN_SMOKE="${6:-0}"
# Optional arg 7: upload provision-result.json as Gist (pass 1 to enable)
RUN_UPLOAD_GIST="${7:-0}"
UE_ROOT="${UE5_ROOT:-/Applications/Epic Games/UE_5.3}"

if [[ -z "$REPO_URL" ]]; then
  echo "Usage: $0 <repo_url> [runner_name] [labels] [workdir] [github_pat]"
  exit 2
fi

# Auto-register helper: if GITHUB_PAT provided, request registration token and run config
function auto_register() {
  if [[ -z "$GITHUB_PAT" ]]; then
    return
  fi
  REPO_PATH=$(echo "$REPO_URL" | sed -E 's#https?://github.com/##; s#\.git$##')
  echo "Requesting ephemeral registration token for $REPO_PATH..."
  REG_JSON=$(curl -s -X POST -H "Authorization: token $GITHUB_PAT" -H "User-Agent: provision-runner" "https://api.github.com/repos/$REPO_PATH/actions/runners/registration-token")
  REG_TOKEN=$(python - <<PY
import sys,json
obj=json.load(sys.stdin)
print(obj.get('token',''))
PY
  <<<"$REG_JSON")
  if [[ -z "$REG_TOKEN" ]]; then
    echo "Failed to get registration token from GitHub"; return 1
  fi
  echo "Configuring runner (unattended)..."
  ./config.sh --url "$REPO_URL" --token "$REG_TOKEN" --name "$RUNNER_NAME" --labels "$LABELS" --work _work --unattended --replace
  echo "Installing runner as service..."
  sudo ./svc.sh install
  sudo ./svc.sh start

  # Verify runner appears online in GitHub (poll API)
  echo "Verifying runner is online (this may take 30-60s)..."
  for i in {1..12}; do
    sleep 5
    RUNNERS_JSON=$(curl -s -H "Authorization: token $GITHUB_PAT" -H "User-Agent: provision-runner" "https://api.github.com/repos/$REPO_PATH/actions/runners")
    ONLINE=$(python - <<PY
import sys,json
obj=json.load(sys.stdin)
for r in obj.get('runners',[]):
    if r.get('name') == "$RUNNER_NAME" and r.get('status') == 'online':
        print('1')
        sys.exit(0)
print('0')
PY
<<<"$RUNNERS_JSON")
    if [[ "$ONLINE" == "1" ]]; then
      echo "Runner '$RUNNER_NAME' is online on GitHub."
      break
    fi
  done
  if [[ "$ONLINE" != "1" ]]; then
    echo "Warning: Runner did not appear online within the expected time. Check runner service logs and GitHub UI."
  fi

  # Optionally dispatch a smoke workflow to verify runner can accept jobs
  if [[ -n "$GITHUB_PAT" && -n "$RUNNER_NAME" && -n "$REPO_PATH" && "${RUN_SMOKE:-0}" == "1" ]]; then
    echo "Dispatching runner smoke workflow..."
    DISPATCH_PAYLOAD=$(python - <<PY
import json
print(json.dumps({"ref":"ci/nsis-smoke-test","inputs":{"expected_runner_name":"%s"}}))
PY
    )
    curl -s -X POST -H "Authorization: token $GITHUB_PAT" -H "Accept: application/vnd.github+json" -H "Content-Type: application/json" -d "$DISPATCH_PAYLOAD" "https://api.github.com/repos/$REPO_PATH/actions/workflows/runner-smoke.yml/dispatches"
    echo "Smoke workflow dispatched; checking run status and attempting to download artifact..."

    # Poll for run completion and download artifact
    for i in {1..30}; do
      sleep 5
      RUNS_JSON=$(curl -s -H "Authorization: token $GITHUB_PAT" -H "User-Agent: provision-runner" "https://api.github.com/repos/$REPO_PATH/actions/workflows/runner-smoke.yml/runs?per_page=10")
      RUN_ID=$(python - <<PY
import sys,json
obj=json.load(sys.stdin)
for r in obj.get('workflow_runs',[]):
    if r.get('created_at'):
        print(r.get('id'))
        # stop at first completed run
        if r.get('status')=='completed':
            print('COMPLETED:'+str(r.get('id')))
            sys.exit(0)
print('')
PY
<<<"$RUNS_JSON")
      if [[ -n "$RUN_ID" ]]; then
        # find completed run id
        COMPLETED_ID=$(python - <<PY
import sys,json
obj=json.load(sys.stdin)
for r in obj.get('workflow_runs',[]):
    if r.get('status')=='completed':
        print(r.get('id'))
        sys.exit(0)
print('')
PY
<<<"$RUNS_JSON")
        if [[ -n "$COMPLETED_ID" ]]; then
          echo "Found completed run: $COMPLETED_ID. Fetching artifacts..."
          ART_JSON=$(curl -s -H "Authorization: token $GITHUB_PAT" "https://api.github.com/repos/$REPO_PATH/actions/runs/$COMPLETED_ID/artifacts")
          ART_URL=$(python - <<PY
import sys,json
obj=json.load(sys.stdin)
for a in obj.get('artifacts',[]):
    if a.get('name')=='runner-smoke':
        print(a.get('archive_download_url'))
        sys.exit(0)
print('')
PY
<<<"$ART_JSON")
          if [[ -n "$ART_URL" ]]; then
            echo "Downloading artifact..."
            curl -L -H "Authorization: token $GITHUB_PAT" -o "$WORKDIR/runner-smoke.zip" "$ART_URL"
            mkdir -p "$WORKDIR/runner-smoke" && unzip -o "$WORKDIR/runner-smoke.zip" -d "$WORKDIR/runner-smoke"
            echo "Downloaded and extracted runner-smoke to $WORKDIR/runner-smoke"

            # Discover and record artifacts of interest
            ARTIFACTS_FILE="$WORKDIR/provision-artifacts.txt"
            rm -f "$ARTIFACTS_FILE" || true
            if [[ -f "$WORKDIR/runner-smoke/wallpaper_check.txt" ]]; then
              echo "$WORKDIR/runner-smoke/wallpaper_check.txt" >> "$ARTIFACTS_FILE"
              echo "Found artifact: $WORKDIR/runner-smoke/wallpaper_check.txt"
            fi
            if [[ -f "$WORKDIR/provision-gist-url.txt" ]]; then
              echo "$WORKDIR/provision-gist-url.txt" >> "$ARTIFACTS_FILE"
              echo "Found artifact: $WORKDIR/provision-gist-url.txt"
            fi
            if [[ -f "$ARTIFACTS_FILE" ]]; then
              echo "Provision artifacts recorded in: $ARTIFACTS_FILE"
              cat "$ARTIFACTS_FILE"
            fi

            # Validate artifact contents (smoke.txt must contain 'Smoke OK')
            SMOKE_FILE="$WORKDIR/runner-smoke/smoke.txt"
            if [[ -f "$SMOKE_FILE" ]]; then
              if grep -q "Smoke OK" "$SMOKE_FILE"; then
                echo "Smoke artifact validation passed."
                cat > "$WORKDIR/provision-result.json" <<EOF
{"status":"success","timestamp":"$(date -u +%Y-%m-%dT%H:%M:%SZ)","runner":"$RUNNER_NAME","artifact":"$WORKDIR/runner-smoke","message":"Smoke artifact validated"}
EOF
                if [[ -f "$ARTIFACTS_FILE" ]]; then echo "Provision artifacts recorded in: $ARTIFACTS_FILE"; cat "$ARTIFACTS_FILE"; fi
                if [[ "$RUN_UPLOAD_GIST" == "1" ]]; then
                  if [[ -z "$GITHUB_PAT" ]]; then echo "No GITHUB_PAT provided; cannot upload gist"; else
                    BODY=$(python - <<PY
import json
print(json.dumps({"description":"Provision result: %s"%("$RUNNER_NAME",),"public":False,"files":{"provision-result.json":{"content":open("$WORKDIR/provision-result.json").read()}}}))
PY
)
                    RES=$(curl -s -X POST -H "Authorization: token $GITHUB_PAT" -H "Content-Type: application/json" -d "$BODY" "https://api.github.com/gists")
                    GIST_URL=$(python - <<PY
import sys,json
obj=json.load(sys.stdin)
print(obj.get('html_url',''))
PY
<<<"$RES")
                    if [[ -n "$GIST_URL" ]]; then echo "$GIST_URL" > "$WORKDIR/provision-gist-url.txt"; echo "Provision result uploaded as Gist: $GIST_URL"; fi
                  fi
                fi
              else
                echo "Smoke artifact validation FAILED: 'Smoke OK' not found in $SMOKE_FILE" >&2
                cat > "$WORKDIR/provision-result.json" <<EOF
{"status":"failure","timestamp":"$(date -u +%Y-%m-%dT%H:%M:%SZ)","runner":"$RUNNER_NAME","artifact":"$SMOKE_FILE","message":"Smoke artifact content mismatch"}
EOF
              if [[ "$RUN_UPLOAD_GIST" == "1" && -n "$GITHUB_PAT" ]]; then
                BODY=$(python - <<PY
import json
print(json.dumps({"description":"Provision result: %s"%("$RUNNER_NAME",),"public":False,"files":{"provision-result.json":{"content":open("$WORKDIR/provision-result.json").read()}}}))
PY
)
                RES=$(curl -s -X POST -H "Authorization: token $GITHUB_PAT" -H "Content-Type: application/json" -d "$BODY" "https://api.github.com/gists")
                GIST_URL=$(python - <<PY
import sys,json
obj=json.load(sys.stdin)
print(obj.get('html_url',''))
PY
<<<"$RES")
                if [[ -n "$GIST_URL" ]]; then echo "$GIST_URL" > "$WORKDIR/provision-gist-url.txt"; echo "Provision result uploaded as Gist: $GIST_URL"; fi
              fi              if [[ -f "$ARTIFACTS_FILE" ]]; then echo "Provision artifacts recorded in: $ARTIFACTS_FILE"; cat "$ARTIFACTS_FILE"; fi              exit 1
              fi
            else
              echo "Smoke artifact validation FAILED: $SMOKE_FILE not found" >&2
              cat > "$WORKDIR/provision-result.json" <<EOF
{"status":"failure","timestamp":"$(date -u +%Y-%m-%dT%H:%M:%SZ)","runner":"$RUNNER_NAME","artifact":"$SMOKE_FILE","message":"Smoke artifact missing"}
EOF
              exit 1
            fi
          else
            echo "runner-smoke artifact not found for run $COMPLETED_ID"
          fi
          break
        fi
      fi
    done
  fi
}


# Install Homebrew if missing (non-interactive)
if ! command -v brew >/dev/null 2>&1; then
  echo "Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  eval "$(/opt/homebrew/bin/brew shellenv)" || true
fi

echo "Installing packages: git cmake ninja pkg-config ffmpeg"
brew install git cmake ninja pkg-config ffmpeg

# Create workdir
sudo mkdir -p "$WORKDIR"
sudo chown "$USER" "$WORKDIR"
cd "$WORKDIR"

# Download latest actions runner for macOS
ASSET_URL=$(curl -s https://api.github.com/repos/actions/runner/releases/latest | jq -r '.assets[] | select(.name|test("actions-runner-osx-x64-")) | .browser_download_url' | head -n1)
if [[ -z "$ASSET_URL" ]]; then
  echo "Could not determine latest runner URL; please download manually from https://github.com/actions/runner/releases/latest"; exit 1
fi

echo "Downloading runner: $ASSET_URL"
curl -Lo actions-runner.tar.gz "$ASSET_URL"
tar xzf actions-runner.tar.gz

cat <<EOF
Runner downloaded to $WORKDIR
Next steps (manual):
  1) Create a registration token on GitHub (Repo Settings -> Actions -> Runners -> New self-hosted runner).
  2) From $WORKDIR run:
     ./config.sh --url $REPO_URL --token <TOKEN> --name $RUNNER_NAME --labels $LABELS --work _work
  3) Install as a service (to run at startup):
     sudo ./svc.sh install
     sudo ./svc.sh start
  4) Optionally set UE5_ROOT in /etc/profile or your runner environment to $UE_ROOT
EOF

echo "Provision helper finished. Run the configure command with a valid token to complete registration."
