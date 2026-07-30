# diagnose_gpu_build.ps1
# Diagnose nvidia-cutlass build failure

Write-Host "`n=== GPU Build Diagnostics ===" -ForegroundColor Cyan

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $ProjectRoot "build\vcpkg_installed\vcpkg\buildtrees\nvidia-cutlass"
$OutLog = Join-Path $ProjectRoot "build\vcpkg_installed\vcpkg\buildtrees\nvidia-cutlass\config-x64-windows-rel-CMakeConfigureLog.yaml.log"

Write-Host "Looking for nvidia-cutlass build logs..." -ForegroundColor Yellow

# Find the actual error log
$ErrorLogs = @(
    "config-x64-windows-out.log",
    "config-x64-windows-rel-CMakeConfigureLog.yaml.log",
    "config-x64-windows-dbg-CMakeConfigureLog.yaml.log"
)

foreach ($log in $ErrorLogs) {
    $LogPath = Join-Path $ProjectRoot "build\vcpkg_installed\vcpkg\buildtrees\nvidia-cutlass\$log"
    if (Test-Path $LogPath) {
        Write-Host "`nFound log: $log" -ForegroundColor Green
        Write-Host "======================================" -ForegroundColor Gray
        
        # Read and display error context
        $content = Get-Content $LogPath -Tail 100 -ErrorAction SilentlyContinue
        $content | ForEach-Object {
            if ($_ -match "error|Error|ERROR|FAILED|failed") {
                Write-Host $_ -ForegroundColor Red
            } elseif ($_ -match "warning|Warning|WARNING") {
                Write-Host $_ -ForegroundColor Yellow
            } else {
                Write-Host $_
            }
        }
        Write-Host "======================================" -ForegroundColor Gray
    }
}

Write-Host "`n=== System Information ===" -ForegroundColor Cyan

# Check CUDA/CUDNN installation
$CudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
if (Test-Path $CudaPath) {
    Write-Host "✓ CUDA 12.6 found at: $CudaPath" -ForegroundColor Green
    $CudaBinPath = Join-Path $CudaPath "bin"
    if (Test-Path (Join-Path $CudaBinPath "nvcc.exe")) {
        Write-Host "✓ NVCC compiler found" -ForegroundColor Green
    }
} else {
    Write-Host "✗ CUDA 12.6 not found at standard location" -ForegroundColor Red
}

# Check CMake version
try {
    $cmakeVersion = cmake --version 2>&1 | Select-Object -First 1
    Write-Host "✓ $cmakeVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ CMake not found in PATH" -ForegroundColor Red
}

# Check Ninja
$ninjaPath = Join-Path $ProjectRoot "vcpkg\downloads\tools\ninja-1.13.2-windows\ninja.exe"
if (Test-Path $ninjaPath) {
    Write-Host "✓ Ninja found at: $ninjaPath" -ForegroundColor Green
} else {
    Write-Host "✗ Ninja not found at: $ninjaPath" -ForegroundColor Red
}

Write-Host "`n=== Recommendations ===" -ForegroundColor Cyan
Write-Host "1. Try updating vcpkg submodule:" -ForegroundColor Gray
Write-Host "   cd vcpkg && git pull && cd .." -ForegroundColor Gray
Write-Host "2. Try using CPU-only ONNX Runtime:" -ForegroundColor Gray
Write-Host "   Edit vcpkg.json: change 'onnxruntime-gpu' to 'onnxruntime'" -ForegroundColor Gray
Write-Host "3. Check CUDA/CUDNN compatibility with ONNX Runtime 1.23.2" -ForegroundColor Gray
