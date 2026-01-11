#pragma once

#include <string>

namespace BeatSync {

// Initialize tracing for the process. serviceName is used as resource.service.name.
// Returns true if tracing was initialized successfully (or no-op stub when disabled).
bool InitializeTracing(const std::string& serviceName);

// Shutdown tracing and flush any pending spans. Safe to call even if tracing was not enabled.
void ShutdownTracing();

} // namespace BeatSync
