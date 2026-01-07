#pragma once

#include <string>

namespace BeatSync {

// Run a command hidden (on Windows) or via shell (on POSIX) and capture stdout+stderr.
// Returns exit code; output is appended to 'output'.
int runHiddenCommand(const std::string& cmdLine, std::string& output);

} // namespace BeatSync
