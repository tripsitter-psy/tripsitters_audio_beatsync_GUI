#include "tracing_helpers.h"

// Deprecated compatibility implementation - prefer using the project's
// tracing facility in src/tracing/Tracing.h (TRACE_FUNC / TRACE_SCOPE macros).

// No-op implementations kept for compatibility; all functionality should be
// implemented via ::BeatSync::tracing::Span instead.

struct BeatSync::TraceSpan::Impl {};

BeatSync::TraceSpan::TraceSpan(const std::string& name) { (void)name; }
BeatSync::TraceSpan::~TraceSpan() {}
void BeatSync::TraceSpan::SetError(const std::string& msg) { (void)msg; }

