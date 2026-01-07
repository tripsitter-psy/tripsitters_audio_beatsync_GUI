#pragma once

#include <string>

namespace BeatSync {

// Cross-platform function to run a command and capture output
// Returns exit code, output is captured in the string parameter
int runHiddenCommand(const std::string& command, std::string& output);

} // namespace BeatSync