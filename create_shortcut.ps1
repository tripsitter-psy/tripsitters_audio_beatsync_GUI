$desktopPath = [Environment]::GetFolderPath('Desktop')
$s = (New-Object -ComObject WScript.Shell).CreateShortcut("$desktopPath\BeatSync Backend Tests.lnk")
$s.TargetPath = "$env:USERPROFILE\Desktop\BeatSyncEditor\build\tests\Release\test_backend_api.exe"
$s.WorkingDirectory = "$env:USERPROFILE\Desktop\BeatSyncEditor"
$s.Save()
