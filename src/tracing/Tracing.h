#pragma once
#include <string>
#include <memory>

namespace BeatSync {
namespace tracing {

void InitTracing(const std::string& outfile = "");
void ShutdownTracing();

class Span {
public:
    explicit Span(const char* name);
    ~Span();
    void End();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tracing
} // namespace BeatSync

// Macros for convenience
#ifdef USE_TRACING
#define TRACE_CONCAT_IMPL(a, b) a##b
#define TRACE_CONCAT(a, b) TRACE_CONCAT_IMPL(a, b)
#define TRACE_FUNC() ::BeatSync::tracing::Span TRACE_CONCAT(__trace_span_, __LINE__)(__func__)
#define TRACE_SCOPE(name) ::BeatSync::tracing::Span TRACE_CONCAT(__trace_span_, __LINE__)(name)
#else
#define TRACE_FUNC()
#define TRACE_SCOPE(name)
#endif
