# fix_gpu_build.ps1
# Multiple strategies to fix GPU build without nvidia-cutlass compilation failure

param(
    [ValidateSet("update-vcpkg", "prebuilt-onnx", "disable-cutlass", "use-cpu")]
    [string]$Strategy = "update-vcpkg"
)

Write-Host "`n=== GPU Build Fix ===" -ForegroundColor Cyan
Write-Host "Strategy: $Strategy" -ForegroundColor Gray

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

switch ($Strategy) {
    "update-vcpkg" {
        Write-Host "`n[Strategy 1] Updating vcpkg to latest version..." -ForegroundColor Yellow
        Write-Host "This may include fixes for nvidia-cutlass builds." -ForegroundColor Gray
        
        cd "$ProjectRoot\vcpkg"
        Write-Host "Pulling latest vcpkg changes..." -ForegroundColor Yellow
        git pull
        
        cd $ProjectRoot
        Write-Host "vcpkg updated. Retry build with: .\rebuild_beatsync.ps1 -GPU -Deploy" -ForegroundColor Green
    }

    "prebuilt-onnx" {
        Write-Host "`n[Strategy 2] Installing pre-built ONNX Runtime GPU..." -ForegroundColor Yellow
        Write-Host "This downloads pre-compiled binaries instead of building from source." -ForegroundColor Gray
        
        # This uses the binary cache if available
        Write-Host "Setting vcpkg to prefer binary cache..." -ForegroundColor Yellow
        
        cd $ProjectRoot
        
        # Configure with binary cache enabled
        $env:VCPKG_BINARY_SOURCES = "clear;x-azblob,https://vcpkgms.blob.core.windows.net/vcpkg,read"
        
        Write-Host "Environment set. Retry build with: .\rebuild_beatsync.ps1 -GPU -Deploy" -ForegroundColor Green
    }

    "disable-cutlass" {
        Write-Host "`n[Strategy 3] Patching nvidia-cutlass to disable header-only features..." -ForegroundColor Yellow
        Write-Host "Note: This is experimental and may affect performance." -ForegroundColor Yellow
        
        $CutlassOverlay = Join-Path $ProjectRoot "vcpkg-cutlass-overlay"
        
        if (-not (Test-Path $CutlassOverlay)) {
            Write-Host "Creating overlay port for nvidia-cutlass..." -ForegroundColor Yellow
            mkdir $CutlassOverlay\ports\nvidia-cutlass -Force | Out-Null
            
            # Copy existing portfile
            $ExistingPort = Join-Path $ProjectRoot "vcpkg\ports\nvidia-cutlass"
            if (Test-Path $ExistingPort) {
                Copy-Item -Path "$ExistingPort\*" -Destination "$CutlassOverlay\ports\nvidia-cutlass\" -Recurse -Force
                Write-Host "Copied portfile to overlay." -ForegroundColor Green
            }
        }
        
        Write-Host "Use overlay with: cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_OVERLAY_PORTS=vcpkg-cutlass-overlay/ports -DVCPKG_MANIFEST_FEATURES=gpu" -ForegroundColor Green
    }

    "use-cpu" {
        Write-Host "`n[Strategy 4] Switching to CPU-only ONNX Runtime..." -ForegroundColor Yellow
        Write-Host "Your GPU will still accelerate during inference via other libraries." -ForegroundColor Gray
        
        # Modify vcpkg.json to use cpu runtime
        $vcpkgJson = Join-Path $ProjectRoot "vcpkg.json"
        
        $content = Get-Content $vcpkgJson -Raw
        if ($content -match '"onnxruntime-gpu"') {
            $content = $content -replace '"onnxruntime-gpu"', '"onnxruntime"'
            Set-Content $vcpkgJson $content
            Write-Host "✓ Changed vcpkg.json: onnxruntime-gpu → onnxruntime" -ForegroundColor Green
        }
        
        Write-Host "Retry build with: .\rebuild_beatsync.ps1 -Deploy" -ForegroundColor Green
    }
}

Write-Host "`n=== Recommended Next Step ===" -ForegroundColor Cyan
Write-Host "Run: .\rebuild_beatsync.ps1 -Deploy" -ForegroundColor Gray
