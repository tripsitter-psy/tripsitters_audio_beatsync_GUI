$s = (New-Object -ComObject WScript.Shell).CreateShortcut("C:\Users\samue\OneDrive\Desktop\BeatSync Backend Tests.lnk")
$s.TargetPath = "C:\Users\samue\Desktop\BeatSyncEditor\build\tests\Release\test_backend_api.exe"
$s.WorkingDirectory = "C:\Users\samue\Desktop\BeatSyncEditor"
$s.Save()
