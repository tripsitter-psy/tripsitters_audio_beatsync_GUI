#pragma once

// Compiler deprecation warnings
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC warning "tracing_helpers.h is deprecated. Use src/tracing/Tracing.h and ::BeatSync::tracing::Span with TRACE_FUNC()/TRACE_SCOPE() instead."
#elif defined(_MSC_VER)
#pragma message("tracing_helpers.h is deprecated. Use src/tracing/Tracing.h and ::BeatSync::tracing::Span with TRACE_FUNC()/TRACE_SCOPE() instead.")
#endif

// This header is deprecated: use src/tracing/Tracing.h and the existing
// ::BeatSync::tracing::Span type with TRACE_FUNC()/TRACE_SCOPE() macros.

#include "../tracing/Tracing.h"

// Provide compatibility aliases
#define TRACE_SCOPE_FN() TRACE_FUNC()

// No other helpers here; the project-wide tracing facility should be used.
