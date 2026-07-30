Get-ChildItem 'C:\Users\samue' -Filter 'MyProject.exe' -Recurse -Depth 5 -ErrorAction SilentlyContinue |
    Where-Object { $_.Length -gt 100MB } |
    Select-Object FullName, @{N='SizeMB';E={[math]::Round($_.Length/1MB,0)}}
