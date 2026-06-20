// Minimal native file dialogs. macOS uses osascript (no extra deps); Windows
// and Linux paths are stubbed for now and will be filled in when those targets
// are built. Returns an empty string if the user cancels.
#pragma once

#include <string>
#include <vector>

namespace FileDialog
{
    // Open a single file. `filterDesc` is a human label (e.g. "Audio"); `exts`
    // is a list of bare extensions without dots (e.g. {"wav","mp3"}). Empty exts
    // means any file.
    std::string OpenFile(const std::string& title,
                         const std::string& filterDesc,
                         const std::vector<std::string>& exts);

    // Open multiple files (for multi-clip video selection).
    std::vector<std::string> OpenFiles(const std::string& title,
                                       const std::string& filterDesc,
                                       const std::vector<std::string>& exts);

    // Choose a destination file path for saving.
    std::string SaveFile(const std::string& title,
                         const std::string& defaultName,
                         const std::vector<std::string>& exts);

    // Choose a folder.
    std::string PickFolder(const std::string& title);
}
