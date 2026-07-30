# DLL Verification and Deployment Guide

## Quick Verification

To verify that all required DLLs are correctly deployed to TripSitter:

```powershell
cd c:\Users\samue\Desktop\BeatSyncEditor
.\scripts\deploy_tripsitter.ps1 -Verify
```

**Expected output if all DLLs are correct:**
```
=== Checking DLLs in D:\UnrealEngine\Engine\Binaries\Win64 ===

  [OK] beatsync_backend_shared.dll (0.35 MB)
  [OK] abseil_dll.dll (2.1 MB)
  [OK] libprotobuf.dll (12.5 MB)
  [OK] libprotobuf-lite.dll (1.8 MB)
  [OK] re2.dll (1.3 MB)
  [OK] onnxruntime.dll (15.2 MB)
  [OK] onnxruntime_providers_shared.dll (0.02 MB)
  [OK] onnxruntime_providers_cuda.dll (350.5 MB)
  [OK] avcodec-62.dll (106.2 MB)
  [OK] avformat-62.dll (22.1 MB)
  [OK] avutil-60.dll (3.5 MB)
  [OK] avfilter-11.dll (89.3 MB)
  [OK] avdevice-62.dll (4.2 MB)
  [OK] swresample-6.dll (0.7 MB)
  [OK] swscale-9.dll (2.1 MB)

=== Summary ===
OK: 15 / 15

All DLLs verified successfully! TripSitter.exe can run.
```

## If Verification Fails

### Missing DLLs
If the script reports `[MISSING]` DLLs:

1. Build the backend library:
```powershell
cd c:\Users\samue\Desktop\BeatSyncEditor\build
cmake --build . --config Release --target beatsync_backend_shared
```

2. Deploy the DLLs:
```powershell
cd c:\Users\samue\Desktop\BeatSyncEditor
.\scripts\deploy_tripsitter.ps1
```

3. Verify again:
```powershell
.\scripts\deploy_tripsitter.ps1 -Verify
```

### Wrong File Size
If the script reports `[WRONG SIZE]` DLLs:

This indicates the wrong version of a DLL was deployed. Common causes:
- **FFmpeg**: vcpkg version (~13MB) was used instead of the correct ThirdParty version (~100MB+)
- **ONNX Runtime**: CPU-only version was used instead of GPU version

**Fix:**
```powershell
# Clean and redeploy
Remove-Item 'D:\UnrealEngine\Engine\Binaries\Win64\*.dll' -ErrorAction SilentlyContinue

# Rebuild backend
cd c:\Users\samue\Desktop\BeatSyncEditor\build
cmake --build . --config Release --target beatsync_backend_shared

# Deploy again
cd c:\Users\samue\Desktop\BeatSyncEditor
.\scripts\deploy_tripsitter.ps1
```

## Alternative: Manual Check

List DLLs in the target directory:

```powershell
Get-ChildItem 'D:\UnrealEngine\Engine\Binaries\Win64\*.dll' | 
    Select-Object Name, @{Name="SizeMB";Expression={[math]::Round($_.Length/1MB,2)}} |
    Format-Table -AutoSize
```

Expected sizes (approximate):
| DLL | Min Size |
|-----|----------|
| beatsync_backend_shared.dll | 0.3 MB |
| avcodec-62.dll | 100 MB |
| avformat-62.dll | 20 MB |
| avutil-60.dll | 2 MB |
| avfilter-11.dll | 80 MB |
| avdevice-62.dll | 3 MB |
| swresample-6.dll | 0.5 MB |
| swscale-9.dll | 2 MB |
| onnxruntime.dll | 14 MB |
| onnxruntime_providers_shared.dll | 0.01 MB |
| onnxruntime_providers_cuda.dll | 5 MB |
| abseil_dll.dll | 1 MB |
| libprotobuf.dll | 10 MB |
| libprotobuf-lite.dll | 1 MB |
| re2.dll | 1 MB |

## Running TripSitter After Verification

Once verification passes:

```powershell
D:\UnrealEngine\Engine\Binaries\Win64\TripSitter.exe
```

If TripSitter crashes with a missing DLL error, re-run the verification to identify which DLL is missing or has the wrong size.
