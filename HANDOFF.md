# Handoff — Windows → Mac (Apple Silicon port)

Last updated: 2026-06-17 (end of Windows session)

## Where things stand

Windows/UE work is parked clean. Branch `feat/vertical-output-and-branding`
is pushed to `origin`. Recent commits:

- `15e7b5e` docs: add configurable per-clip speed ramps to roadmap (§3.3)
- `45d9bac` chore: gitignore UE DerivedDataCache
- `6c54261` refactor: move Vertical 9:16 checkbox to Analysis section (built + tested live)

No PR opened yet.

## Direction

Build the **Apple Silicon version**:
- Keep the portable C++ backend (FFmpeg + ONNX Runtime). All video work is
  FFmpeg-CLI driven, no UE dependency — ports cleanly.
- Drop the Unreal Engine GUI.
- Build a cross-platform frontend for Mac first, then Linux.

## Frontend framework — OPEN decision (ImGui vs Qt)

Leaning ImGui, but not locked. Summary of the tradeoff:

- **ImGui** — fastest to integrate with the C++ backend; DrawList is great for
  the custom waveform/effect-timeline rendering (already hand-drawn in Slate, so
  it ports ~1:1). Weakest on bespoke branded chrome.
- **Qt** — most flexible / most capable for a polished branded app; QSS can skin
  custom widget shapes declaratively. Cost: learning curve, heavier build,
  LGPL/commercial licensing.
- **Slint** — modern middle ground, lighter than Qt, C++ API, younger ecosystem.

**Recommended approach:** prototype in ImGui first to prove the backend +
waveform/timeline on arm64 (the risky part). Starting in ImGui does NOT lock the
backend, since all heavy lifting stays in the portable C++ core. Qt is the upgrade
path if ImGui's look ends up bugging you.

### Can ImGui match the current branded look?

Mostly yes, with little pain — the asset files reuse directly:
- `Corpta.otf` → `AddFontFromFileTTF` (same file). Neon cyan/magenta headers are
  just flat tinted text in that font.
- `wallpaper.png` + `TitleHeader.png` → load as textures, draw via DrawList.
- Dark theme + cyan/magenta accents → ImGui's style system, trivial.
- Waveform / effect timeline → DrawList, ports cleanly from Slate.

The ONE real cost: the angular/beveled sci-fi **button shapes** (ANALYZE,
APPLY BEATS, BROWSE). ImGui buttons are rectangles; matching bespoke shapes means
custom-draw per widget (`InvisibleButton` + DrawList). Bounded one-time styling
pass, not a fundamental fight. If that chrome is sacred, it's the strongest single
argument for Qt.

## Early decisions to make on the Mac

1. Repo path on Mac; check out `feat/vertical-output-and-branding` or branch fresh
   off `main-new`?
2. Dependency strategy for arm64: vcpkg vs Homebrew/native for FFmpeg + ONNX Runtime.
3. First job: get the backend configuring under CMake on arm64 — **no UI yet**.

## Read first on the Mac

- `CLAUDE.md` — build/architecture (note: heavily Windows/UE-specific; treat the
  UE/TensorRT/DLL sections as Windows-only).
- `ROADMAP.md` — feature roadmap; §3.3 is the new configurable per-clip speed-ramp idea.
- `git log` on the current branch.
