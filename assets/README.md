Assets directory

Place GUI assets here:
- **alpha_2.png** — Top header transparency asset (PNG with alpha channel)
- **alpha.png** — Button transparency/artwork asset (PNG with alpha channel)
- **background.png** — Background image used by the GUI (PNG, recommended 1920x1080)
- **ComfyUI_03324_.png** — Upscaled background image (optional, PNG)
- **icon.ico** — Windows application icon (ICO format)
- **icon.rc** — Resource script referencing the icon (used by CMake when present)

Notes:
- The project will use the default gradient background if `background.png` is missing or invalid.
- Alpha channel assets (alpha_2.png, alpha.png) are used for transparency overlays in the GUI.
- Replace the placeholder files with real images before creating release builds.
- On macOS, assets are copied to `TripSitter.app/Contents/Resources/assets/` during build.
- On Windows, assets are copied to the `bin/Release/assets/` directory.

Importing assets from an external folder
- You can copy assets into this directory using the `scripts/import_assets.ps1` script.

Example:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\import_assets.ps1 -Source "C:\Users\samue\Downloads\assets for GUI aesthetics" -Force
```
The script backs up any existing assets into `assets/backup-YYYYMMDD-HHMMSS` before copying files.
Supported image formats: `.png`, `.jpg`, `.jpeg`, `.ico`. Ensure `icon.ico` is present if you want an embedded Windows icon.

Required Assets Checklist:
- [ ] alpha_2.png - Header transparency
- [ ] alpha.png - Button artwork
- [ ] background.png - Main background
- [ ] icon.ico - Windows icon (Windows only)
