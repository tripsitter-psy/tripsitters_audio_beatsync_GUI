#pragma once

#include <string>
#include <vector>

namespace BeatSync {

// Create a ZIP archive containing the given files (stored, no compression).
// Returns true on success, false on error; 'error' will contain a message.
bool createZip(const std::vector<std::string>& files, const std::string& dest, std::string& error);

} // namespace BeatSync
