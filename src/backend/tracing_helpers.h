#pragma once

// This header is deprecated: use src/tracing/Tracing.h and the existing
// ::BeatSync::tracing::Span type with TRACE_FUNC()/TRACE_SCOPE() macros.

#include "../tracing/Tracing.h"

// Provide compatibility aliases
#define TRACE_SCOPE_FN() TRACE_FUNC()

// No other helpers here; the project-wide tracing facility should be used.
