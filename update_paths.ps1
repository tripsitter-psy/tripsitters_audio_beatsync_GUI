$oldPath = 'D:\UnrealEngine'
$newPath = 'D:\UnrealEngine'
$searchPath = 'C:\Users\samue\Desktop\BeatSyncEditor'
$files = Get-ChildItem -Path $searchPath -Recurse -Include '*.md','*.ps1','*.bat','*.cs' -ErrorAction SilentlyContinue
$updatedCount = 0
foreach ($file in $files) {
    $content = Get-Content $file.FullName -Raw
    if ($content -match [regex]::Escape($oldPath)) {
        $newContent = $content -replace [regex]::Escape($oldPath), $newPath
        Set-Content -Path $file.FullName -Value $newContent -NoNewline
        Write-Host "Updated: $($file.FullName)"
        $updatedCount++
    }
}
Write-Host "Path update complete. Updated $updatedCount files."
# Also update bash style paths
$oldBash = '/mnt/d/UnrealEngine'
$newBash = '/mnt/d/UnrealEngine'
$updatedBash = 0
foreach ($file in $files) {
    $content = Get-Content $file.FullName -Raw
    if ($content -match [regex]::Escape($oldBash)) {
        $newContent = $content -replace [regex]::Escape($oldBash), $newBash
        Set-Content -Path $file.FullName -Value $newContent -NoNewline
        if (-not ($file.FullName -match 'update_paths')) {
            Write-Host "Updated bash path in: $($file.FullName)"
        }
        $updatedBash++
    }
}
Write-Host "Bash path update complete. Updated $updatedBash files."