@echo off
echo Fixing MTV TripSitter installation...
echo.

:: Check for admin privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: This script requires administrator privileges.
    echo Right-click and select "Run as administrator"
    pause
    exit /b 1
)

:: Rename folder
if exist "C:\Program Files\MTV TripSitter\TripSitter" (
    if exist "C:\Program Files\MTV TripSitter\MyProject" (
        echo Removing old MyProject folder...
        rmdir /s /q "C:\Program Files\MTV TripSitter\MyProject"
    )
    echo Renaming TripSitter to MyProject...
    rename "C:\Program Files\MTV TripSitter\TripSitter" MyProject
    if %errorLevel% equ 0 (
        echo SUCCESS: Folder renamed
    ) else (
        echo FAILED: Could not rename folder
        pause
        exit /b 1
    )
) else (
    echo Folder already named correctly or not found
)

:: Verify
if exist "C:\Program Files\MTV TripSitter\MyProject\Binaries\Win64\MyProject.exe" (
    echo.
    echo Installation fixed! You can now launch TripSitter.exe
) else (
    echo.
    echo ERROR: MyProject.exe not found at expected path
)

echo.
pause
