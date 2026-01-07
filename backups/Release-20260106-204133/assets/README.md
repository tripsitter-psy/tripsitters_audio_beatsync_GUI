Assets directory

Place GUI assets here:
- asset - this one.png — current primary MTV Trip Sitter background (PNG)
- asset for top hedder alpha_2.png — header banner (alpha background) shown at the top of the GUI (PNG)
- button asset alpha.png — start sync button artwork with transparent background (PNG)
- background.png — legacy background image fallback (PNG)
- icon.ico — Windows application icon (ICO)
- icon.rc — resource script referencing the icon (used by CMake when present)

Notes:
- The project will use the default gradient background if `background.png` is missing or invalid.
- Replace the placeholder files with real images before creating release builds.

Importing assets from an external folder
- You can copy assets into this directory using the `scripts/import_assets.ps1` script.

Example:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\import_assets.ps1 -Source "C:\Users\samue\Downloads\assets for GUI aesthetics" -Force
```
The script backs up any existing assets into `assets/backup-YYYYMMDD-HHMMSS` before copying files.
Supported image formats: `.png`, `.jpg`, `.jpeg`, `.ico`. Ensure `icon.ico` is present if you want an embedded Windows icon.
