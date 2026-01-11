param(
    [Parameter(Mandatory=$true)][string]$Project,
    [string]$TestName = "TripSitter.Beatsync.EditorSmoke",
    [string]$UEPath
)

# Resolve UE path
if (-not $UEPath) {
    if ($env:UE5_ROOT) { $UEPath = $env:UE5_ROOT }
    else {
        # Try common Windows install dirs and allow wildcard for minor versions (e.g., UE_5.7, UE_5.7.1)
        $candidates = Get-ChildItem "C:\Program Files\Epic Games\" -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -like 'UE_5.7*' }
        if ($candidates -and $candidates.Count -gt 0) { $UEPath = $candidates[0].FullName }
        elseif (Test-Path "C:\Program Files\Epic Games\UE_5.7") { $UEPath = "C:\Program Files\Epic Games\UE_5.7" }

    }
}

if (-not $UEPath) {
    Write-Error "UE5 install path not provided and could not be inferred. Set -UEPath or UE5_ROOT environment variable." ; exit 2
}

$EditorCmd = Join-Path -Path $UEPath -ChildPath "Engine\Binaries\Win64\UE5Editor-Cmd.exe"
if (-not (Test-Path $EditorCmd)) { Write-Error "Could not find UE5Editor-Cmd.exe at $EditorCmd"; exit 2 }

$absProject = Resolve-Path $Project
$log = Join-Path -Path (Split-Path $absProject -Parent) -ChildPath "BeatsyncAutomation.log"

$exec = "`"$absProject`" -ExecCmds=\"Automation RunTests $TestName; Quit\" -unattended -nopause -nullrhi -abslog=\"$log\""
Write-Host "Running: $EditorCmd $exec"

$proc = Start-Process -FilePath $EditorCmd -ArgumentList $exec -Wait -PassThru
if ($proc.ExitCode -ne 0) {
    Write-Error "Editor automation process exited with code $($proc.ExitCode). Check $log for details." ; exit $proc.ExitCode
}

Write-Host "Automation run complete. See $log for details."; exit 0
