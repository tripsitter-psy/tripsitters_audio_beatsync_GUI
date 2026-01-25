#include "tracing_helpers.h"

// Deprecated compatibility implementation - prefer using the project's
// tracing facility in src/backend/tracing.h (TRACE_FUNC / TRACE_SCOPE macros)
// and the ::BeatSync::tracing API.

// No-op implementations kept for compatibility; all functionality should be
// implemented via ::BeatSync::tracing::Span instead.

// Empty struct satisfying the pImpl idiom for TraceSpan. The header forward-declares
// this type; a concrete definition is required even though TraceSpan is now a no-op.
// Intentionally empty and never instantiated—exists only to allow compilation.
struct BeatSync::TraceSpan::Impl {};

BeatSync::TraceSpan::TraceSpan(const std::string& name) { (void)name; }
BeatSync::TraceSpan::~TraceSpan() {}
void BeatSync::TraceSpan::SetError(const std::string& msg) { (void)msg; }

